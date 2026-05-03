from io import BytesIO

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


class RouteDetector:
    ROUTE_TO_REGION_MODALITY = {
        "brain_mri": ("brain", "mri"),
        "bone_xray": ("bone", "xray"),
        "chest_xray": ("chest", "xray"),
        "unknown": (None, None),
    }

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
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict(self, raw_bytes: bytes) -> dict:
        try:
            image = Image.open(BytesIO(raw_bytes)).convert("RGB")
        except UnidentifiedImageError:
            return {
                "route_label": "unknown",
                "raw_route_label": "unknown",
                "region": None,
                "modality": None,
                "confidence": 0.0,
                "margin": 0.0,
                "requires_confirmation": True,
                "supported": False,
                "probabilities": {},
                "reason": "File could not be opened as an image.",
            }

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(probs.argmax())
        raw_route_label = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        sorted_probs = sorted([float(p) for p in probs], reverse=True)
        margin = (
            sorted_probs[0] - sorted_probs[1]
            if len(sorted_probs) > 1
            else sorted_probs[0]
        )

        raw_region, raw_modality = self.ROUTE_TO_REGION_MODALITY.get(
            raw_route_label,
            (None, None),
        )

        low_confidence = confidence < self.MIN_CONFIDENCE
        low_margin = margin < self.MIN_MARGIN
        raw_unknown = raw_route_label == "unknown"

        if raw_unknown or low_confidence or low_margin:
            safe_route_label = "unknown"
            region = None
            modality = None
            supported = False
            requires_confirmation = True
            reason = (
                "Route detector confidence or margin was insufficient, "
                "so input was treated as unknown."
            )
        else:
            safe_route_label = raw_route_label
            region = raw_region
            modality = raw_modality
            supported = region is not None and modality is not None
            requires_confirmation = False
            reason = "Route selected by multi-class route detector."

        return {
            "route_label": safe_route_label,
            "raw_route_label": raw_route_label,
            "region": region,
            "modality": modality,
            "confidence": confidence,
            "margin": margin,
            "requires_confirmation": requires_confirmation,
            "supported": supported,
            "probabilities": {
                self.class_names[i]: float(probs[i])
                for i in range(len(self.class_names))
            },
            "reason": reason,
        }