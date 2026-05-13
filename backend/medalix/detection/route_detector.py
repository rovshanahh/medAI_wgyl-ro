from io import BytesIO

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

from medalix.config.route_metadata import ROUTE_TO_REGION_MODALITY


class RouteDetector:
    MIN_CONFIDENCE = 0.80
    MIN_MARGIN = 0.30

    def __init__(
        self,
        model_path: str = "reference_data/route_detector/route_detector_model.pth",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device)

        if "class_names" not in checkpoint:
            raise ValueError("Route detector checkpoint is missing class_names.")

        if "model_state_dict" not in checkpoint:
            raise ValueError("Route detector checkpoint is missing model_state_dict.")

        self.class_names = checkpoint["class_names"]

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        image_size = int(checkpoint.get("image_size", 224))

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _unknown_result(self, reason: str, raw_route_label: str = "unknown") -> dict:
        return {
            "route_label": "unknown",
            "raw_route_label": raw_route_label,
            "attempted_route_label": raw_route_label,
            "region": None,
            "modality": None,
            "confidence": 0.0,
            "margin": 0.0,
            "requires_confirmation": True,
            "supported": False,
            "route_decision": "UNKNOWN",
            "decision_reasons": [reason],
            "probabilities": {},
            "reason": reason,
        }

    def predict(self, raw_bytes: bytes) -> dict:
        try:
            image = Image.open(BytesIO(raw_bytes)).convert("RGB")
        except UnidentifiedImageError:
            return self._unknown_result("File could not be opened as an image.")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(probs.argmax())
        raw_route_label = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        sorted_probs = sorted([float(p) for p in probs], reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

        raw_region, raw_modality = ROUTE_TO_REGION_MODALITY.get(
            raw_route_label,
            (None, None),
        )

        probability_map = {
            self.class_names[i]: float(probs[i])
            for i in range(len(self.class_names))
        }

        decision_reasons = []

        if raw_route_label == "unknown":
            decision_reasons.append("Top route class was unknown.")

        if confidence < self.MIN_CONFIDENCE:
            decision_reasons.append(
                f"Route confidence {confidence:.3f} is below threshold {self.MIN_CONFIDENCE:.2f}."
            )

        if margin < self.MIN_MARGIN:
            decision_reasons.append(
                f"Route margin {margin:.3f} is below threshold {self.MIN_MARGIN:.2f}."
            )

        if raw_region is None or raw_modality is None:
            decision_reasons.append(
                f"Raw route '{raw_route_label}' is not mapped to an active region/modality."
            )

        if decision_reasons:
            return {
                "route_label": "unknown",
                "raw_route_label": raw_route_label,
                "attempted_route_label": raw_route_label,
                "region": None,
                "modality": None,
                "confidence": confidence,
                "margin": margin,
                "requires_confirmation": True,
                "supported": False,
                "route_decision": "NEEDS_CONFIRMATION",
                "decision_reasons": decision_reasons,
                "probabilities": probability_map,
                "reason": "Route was not trusted enough for automatic model selection.",
            }

        return {
            "route_label": raw_route_label,
            "raw_route_label": raw_route_label,
            "attempted_route_label": raw_route_label,
            "region": raw_region,
            "modality": raw_modality,
            "confidence": confidence,
            "margin": margin,
            "requires_confirmation": False,
            "supported": True,
            "route_decision": "ACCEPTED",
            "decision_reasons": [
                "Route confidence and margin passed automatic selection thresholds."
            ],
            "probabilities": probability_map,
            "reason": "Route selected by multi-class route detector.",
        }
