from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


class TrainedMedicalImageGate:
    def __init__(
        self,
        model_path: str = "reference_data/input_gate/medical_image_gate_model.pth",
    ):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        self.class_names = checkpoint["class_names"]
        self.threshold = float(checkpoint.get("threshold", 0.75))
        image_size = int(checkpoint.get("image_size", 224))

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

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

    def evaluate(self, raw_bytes: bytes) -> dict:
        try:
            image = Image.open(BytesIO(raw_bytes)).convert("RGB")
        except UnidentifiedImageError:
            return {
                "accepted": False,
                "confidence": 0.0,
                "predicted_label": "unreadable",
                "reason": "File could not be opened as a readable image.",
                "probabilities": {},
            }

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

        probabilities = {
            self.class_names[i]: float(probs[i])
            for i in range(len(self.class_names))
        }

        medical_probability = probabilities.get("medical", 0.0)
        predicted_label = max(probabilities, key=probabilities.get)
        accepted = medical_probability >= self.threshold

        return {
            "accepted": accepted,
            "confidence": medical_probability,
            "predicted_label": predicted_label,
            "threshold": self.threshold,
            "probabilities": probabilities,
            "reason": (
                "Input accepted as likely medical imaging."
                if accepted
                else "Input rejected as likely non-medical or unsupported."
            ),
        }
