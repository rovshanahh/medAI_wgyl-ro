from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


class ChestXrayInputGate:
    POSITIVE_LABELS = {
        "chest_xray",
        "chestxray",
        "chest",
        "cxr",
        "xray_chest",
        "positive",
        "yes",
        "1",
    }

    CHEST_FILENAME_HINTS = {
        "chest",
        "cxr",
        "thorax",
        "lung",
        "pneumonia",
        "xray",
        "x-ray",
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
    }

    def __init__(self, model_path: str = "reference_data/input_gate/input_gate_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device)

        if "class_names" not in checkpoint:
            raise ValueError("Chest X-ray gate checkpoint is missing class_names.")

        if "model_state_dict" not in checkpoint:
            raise ValueError("Chest X-ray gate checkpoint is missing model_state_dict.")

        self.class_names = checkpoint["class_names"]

        self.model = models.resnet18(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
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

    def _is_positive_chest_label(self, label: str) -> bool:
        normalized = self._normalize_label(label)

        if normalized in self.POSITIVE_LABELS:
            return True

        if "chest" in normalized and ("xray" in normalized or "x_ray" in normalized):
            return True

        return False

    def _has_chest_filename_hint(self, filename: str | None) -> bool:
        if not filename:
            return False

        filename_lower = filename.lower()
        return any(hint in filename_lower for hint in self.CHEST_FILENAME_HINTS)

    def _has_bone_filename_hint(self, filename: str | None) -> bool:
        if not filename:
            return False

        filename_lower = filename.lower()
        return any(hint in filename_lower for hint in self.BONE_FILENAME_HINTS)

    def _get_chest_probability(self, probs) -> float:
        chest_probability = 0.0

        for index, label in enumerate(self.class_names):
            if self._is_positive_chest_label(label):
                chest_probability = max(chest_probability, float(probs[index]))

        return chest_probability

    def _visual_sanity_check(self, image: Image.Image) -> dict:
        gray = image.convert("L").resize((224, 224))
        arr = np.asarray(gray).astype(np.float32) / 255.0

        h, w = arr.shape

        left = arr[:, : w // 2]
        right = arr[:, w // 2 :]

        left_mean = float(left.mean())
        right_mean = float(right.mean())
        symmetry_gap = abs(left_mean - right_mean)

        center = arr[h // 5 : 4 * h // 5, w // 5 : 4 * w // 5]
        center_std = float(center.std())
        global_std = float(arr.std())
        foreground_ratio = float((arr > 0.12).mean())

        passed = (
            symmetry_gap <= 0.18
            and center_std >= 0.04
            and global_std >= 0.04
            and 0.20 <= foreground_ratio <= 0.95
        )

        return {
            "passed": passed,
            "symmetry_gap": symmetry_gap,
            "center_std": center_std,
            "global_std": global_std,
            "foreground_ratio": foreground_ratio,
        }

    def evaluate(self, raw_bytes: bytes, filename: str | None = None) -> dict:
        try:
            image = Image.open(BytesIO(raw_bytes)).convert("RGB")
        except UnidentifiedImageError:
            return {
                "is_chest_xray_like": False,
                "predicted_label": None,
                "confidence": 0.0,
                "chest_probability": 0.0,
                "margin": 0.0,
                "hard_reject": True,
                "visual_sanity": {},
                "probabilities": {},
                "reason": "File could not be opened as an image.",
            }

        visual_sanity = self._visual_sanity_check(image)
        has_chest_hint = self._has_chest_filename_hint(filename)
        has_bone_hint = self._has_bone_filename_hint(filename)

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(probs.argmax())
        pred_label = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        chest_probability = self._get_chest_probability(probs)

        sorted_probs = sorted([float(p) for p in probs], reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

        is_positive_label = self._is_positive_chest_label(pred_label)

        is_chest_xray_like = (
            is_positive_label
            and confidence >= 0.90
            and chest_probability >= 0.90
            and margin >= 0.45
            and visual_sanity["passed"]
            and not has_bone_hint
        )

        if has_chest_hint and not has_bone_hint:
            is_chest_xray_like = (
                is_positive_label
                and confidence >= 0.80
                and chest_probability >= 0.80
                and margin >= 0.30
                and visual_sanity["passed"]
            )

        hard_reject = (
            not is_chest_xray_like
            and confidence >= 0.80
        )

        return {
            "is_chest_xray_like": is_chest_xray_like,
            "predicted_label": pred_label,
            "confidence": confidence,
            "chest_probability": chest_probability,
            "margin": margin,
            "hard_reject": hard_reject,
            "visual_sanity": visual_sanity,
            "filename_hint": has_chest_hint,
            "bone_filename_hint": has_bone_hint,
            "probabilities": {
                self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))
            },
            "reason": (
                "Accepted as chest X-ray by model confidence and visual sanity check."
                if is_chest_xray_like
                else "Rejected by Chest X-ray gate because confidence, margin, visual sanity check, or bone filename conflict failed."
            ),
        }