from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


class BoneXrayInputGate:
    POSITIVE_LABELS = {
        "bone_xray",
        "bonexray",
        "bone",
        "xray_bone",
        "musculoskeletal",
        "mura",
        "positive",
        "yes",
        "1",
    }

    BONE_FILENAME_HINTS = {
        "bone",
        "hand",
        "wrist",
        "finger",
        "elbow",
        "shoulder",
        "humerus",
        "forearm",
        "mura",
        "xray",
        "x-ray",
    }

    def __init__(
        self,
        model_path: str = "reference_data/input_gate/bone_xray_input_gate_model.pth",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device)

        if "class_names" not in checkpoint:
            raise ValueError("Bone X-ray gate checkpoint is missing class_names.")

        if "model_state_dict" not in checkpoint:
            raise ValueError("Bone X-ray gate checkpoint is missing model_state_dict.")

        self.class_names = checkpoint["class_names"]

        self.model = models.resnet18(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        image_size = int(checkpoint.get("image_size", 224))

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _normalize_label(self, label: str) -> str:
        return (
            str(label)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

    def _is_positive_bone_label(self, label: str) -> bool:
        normalized = self._normalize_label(label)

        if normalized in self.POSITIVE_LABELS:
            return True

        if "bone" in normalized and ("xray" in normalized or "x_ray" in normalized):
            return True

        return False

    def _has_filename_hint(self, filename: str | None) -> bool:
        if not filename:
            return False

        filename_lower = filename.lower()
        return any(hint in filename_lower for hint in self.BONE_FILENAME_HINTS)

    def _get_bone_probability(self, probs) -> float:
        bone_probability = 0.0

        for index, label in enumerate(self.class_names):
            if self._is_positive_bone_label(label):
                bone_probability = max(bone_probability, float(probs[index]))

        return bone_probability

    def _visual_sanity_check(self, image: Image.Image) -> dict:
        gray = image.convert("L").resize((224, 224))
        arr = np.asarray(gray).astype(np.float32) / 255.0

        global_std = float(arr.std())
        foreground_ratio = float((arr > 0.12).mean())

        center = arr[56:168, 56:168]
        center_foreground_ratio = float((center > 0.12).mean())

        passed = (
            global_std >= 0.04
            and 0.03 <= foreground_ratio <= 0.85
            and center_foreground_ratio >= 0.03
        )

        return {
            "passed": passed,
            "global_std": global_std,
            "foreground_ratio": foreground_ratio,
            "center_foreground_ratio": center_foreground_ratio,
        }

    def evaluate(self, raw_bytes: bytes, filename: str | None = None) -> dict:
        try:
            image = Image.open(BytesIO(raw_bytes)).convert("RGB")
        except UnidentifiedImageError:
            return {
                "is_bone_xray_like": False,
                "predicted_label": None,
                "confidence": 0.0,
                "bone_probability": 0.0,
                "margin": 0.0,
                "hard_reject": True,
                "visual_sanity": {},
                "probabilities": {},
                "reason": "File could not be opened as an image.",
            }

        visual_sanity = self._visual_sanity_check(image)
        has_filename_hint = self._has_filename_hint(filename)

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(probs.argmax())
        pred_label = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        bone_probability = self._get_bone_probability(probs)

        sorted_probs = sorted([float(p) for p in probs], reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

        is_positive_label = self._is_positive_bone_label(pred_label)

        base_accept = (
            is_positive_label
            and confidence >= 0.85
            and bone_probability >= 0.85
            and margin >= 0.40
            and visual_sanity["passed"]
        )

        filename_assisted_accept = (
            has_filename_hint
            and is_positive_label
            and confidence >= 0.70
            and bone_probability >= 0.70
            and margin >= 0.25
            and visual_sanity["passed"]
        )

        is_bone_xray_like = base_accept or filename_assisted_accept

        return {
            "is_bone_xray_like": is_bone_xray_like,
            "predicted_label": pred_label,
            "confidence": confidence,
            "bone_probability": bone_probability,
            "margin": margin,
            "hard_reject": False,
            "visual_sanity": visual_sanity,
            "filename_hint": has_filename_hint,
            "probabilities": {
                self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))
            },
            "reason": (
                "Accepted as bone X-ray by model confidence and visual sanity check."
                if is_bone_xray_like
                else "Rejected by Bone X-ray gate because confidence, margin, or visual sanity check failed."
            ),
        }