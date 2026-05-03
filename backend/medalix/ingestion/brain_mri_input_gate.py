from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


class BrainMriInputGate:
    POSITIVE_LABELS = {
        "brain_mri",
        "brainmri",
        "brain",
        "mri",
        "mr",
        "positive",
        "yes",
        "1",
    }

    def __init__(
        self,
        model_path: str = "reference_data/input_gate/brain_mri_input_gate_model.pth",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device)

        if "class_names" not in checkpoint:
            raise ValueError("Brain MRI gate checkpoint is missing class_names.")

        if "model_state_dict" not in checkpoint:
            raise ValueError("Brain MRI gate checkpoint is missing model_state_dict.")

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

    def _is_positive_brain_label(self, label: str) -> bool:
        normalized = self._normalize_label(label)

        if normalized in self.POSITIVE_LABELS:
            return True

        if "brain" in normalized and ("mri" in normalized or "mr" in normalized):
            return True

        return False

    def _get_brain_probability(self, probs) -> float:
        brain_probability = 0.0

        for index, label in enumerate(self.class_names):
            if self._is_positive_brain_label(label):
                brain_probability = max(brain_probability, float(probs[index]))

        return brain_probability

    def _visual_sanity_check(self, image: Image.Image) -> dict:
        gray = image.convert("L").resize((224, 224))
        arr = np.asarray(gray).astype(np.float32) / 255.0

        h, w = arr.shape

        border_pixels = np.concatenate(
            [
                arr[:15, :].reshape(-1),
                arr[-15:, :].reshape(-1),
                arr[:, :15].reshape(-1),
                arr[:, -15:].reshape(-1),
            ]
        )

        center = arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

        border_mean = float(border_pixels.mean())
        center_mean = float(center.mean())
        global_std = float(arr.std())

        foreground = arr > 0.12
        foreground_ratio = float(foreground.mean())

        center_foreground_ratio = float((center > 0.12).mean())

        passed = (
            border_mean <= 0.35
            and center_mean > border_mean
            and 0.05 <= foreground_ratio <= 0.75
            and center_foreground_ratio >= 0.10
            and global_std >= 0.05
        )

        return {
            "passed": passed,
            "border_mean": border_mean,
            "center_mean": center_mean,
            "foreground_ratio": foreground_ratio,
            "center_foreground_ratio": center_foreground_ratio,
            "global_std": global_std,
        }

    def evaluate(self, raw_bytes: bytes) -> dict:
        try:
            image = Image.open(BytesIO(raw_bytes)).convert("RGB")
        except UnidentifiedImageError:
            return {
                "is_brain_mri_like": False,
                "predicted_label": None,
                "confidence": 0.0,
                "brain_probability": 0.0,
                "hard_reject": True,
                "visual_sanity": {},
                "probabilities": {},
                "reason": "File could not be opened as an image.",
            }

        visual_sanity = self._visual_sanity_check(image)

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(probs.argmax())
        pred_label = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        brain_probability = self._get_brain_probability(probs)

        sorted_probs = sorted([float(p) for p in probs], reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

        is_positive_label = self._is_positive_brain_label(pred_label)

        is_brain_mri_like = (
            is_positive_label
            and confidence >= 0.90
            and brain_probability >= 0.90
            and margin >= 0.50
            and visual_sanity["passed"]
        )

        return {
            "is_brain_mri_like": is_brain_mri_like,
            "predicted_label": pred_label,
            "confidence": confidence,
            "brain_probability": brain_probability,
            "margin": margin,
            "hard_reject": False,
            "visual_sanity": visual_sanity,
            "probabilities": {
                self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))
            },
            "reason": (
                "Accepted as brain MRI by model confidence and visual sanity check."
                if is_brain_mri_like
                else "Rejected by Brain MRI gate because confidence, margin, or visual sanity check failed."
            ),
        }