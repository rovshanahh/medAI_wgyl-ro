from io import BytesIO

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class ChestXrayInputGate:
    def __init__(self, model_path: str = "reference_data/input_gate/input_gate_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint["class_names"]

        self.model = models.resnet18(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def evaluate(self, raw_bytes: bytes) -> dict:
        image = Image.open(BytesIO(raw_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(probs.argmax())
        pred_label = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        hard_reject = (
            pred_label == "not_chest_xray"
            and confidence > 0.80
        )

        return {
            "is_chest_xray_like": pred_label == "chest_xray",
            "predicted_label": pred_label,
            "confidence": confidence,
            "hard_reject": hard_reject,
            "probabilities": {
                self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))
            },
        }