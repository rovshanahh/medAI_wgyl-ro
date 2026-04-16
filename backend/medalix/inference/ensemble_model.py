import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision import models

LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

REPO_ID = "itsomk/chexpert-densenet121"
FILENAME = "pytorch_model.safetensors"


class DenseNet121CheXpert(nn.Module):
    def __init__(self, num_labels: int = 14, dropout_p: float = 0.2):
        super().__init__()
        self.densenet = models.densenet121(weights=None)
        num_features = self.densenet.classifier.in_features

        self.dropout_p = dropout_p
        self.densenet.classifier = nn.Linear(num_features, num_labels)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.densenet.features(x)
        features = F.relu(features, inplace=False)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.densenet.classifier(features)
        return logits

    def forward_with_mc_dropout(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        dropped = F.dropout(features, p=self.dropout_p, training=True)
        logits = self.densenet.classifier(dropped)
        return logits, features


def load_base_model() -> DenseNet121CheXpert:
    local_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    state = load_file(local_path)

    model = DenseNet121CheXpert(num_labels=14, dropout_p=0.2)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


class EnsembleModel:
    def __init__(self, mc_passes: int = 10):
        self.model = load_base_model()
        self.mc_passes = mc_passes

    def predict(self, tensor: torch.Tensor) -> dict:
        mc_probs = []

        with torch.no_grad():
            feature_vector = None

            for i in range(self.mc_passes):
                logits, features = self.model.forward_with_mc_dropout(tensor)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                mc_probs.append(probs)

                if i == 0:
                    feature_vector = features.squeeze(0).cpu().numpy()

        mc_probs = np.array(mc_probs, dtype=np.float32)
        mean_probs = mc_probs.mean(axis=0)
        var_probs = mc_probs.var(axis=0)

        top_idx = int(np.argmax(mean_probs))

        positive_findings = [
            LABELS[i] for i in range(len(LABELS)) if mean_probs[i] >= 0.5
        ]

        return {
            "top_label": LABELS[top_idx],
            "top_probability": float(mean_probs[top_idx]),
            "positive_findings": positive_findings,
            "probabilities": {
                LABELS[i]: float(mean_probs[i]) for i in range(len(LABELS))
            },
            "epistemic_uncertainty": {
                LABELS[i]: float(var_probs[i]) for i in range(len(LABELS))
            },
            "features": feature_vector.tolist() if feature_vector is not None else None,
        }