from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torchvision import models

CHEST_XRAY_LABELS = [
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

BONE_XRAY_LABELS = [
    "Normal",
    "Abnormal",
]

CHEST_REPO_ID = "itsomk/chexpert-densenet121"
CHEST_FILENAME = "pytorch_model.safetensors"


class DenseNet121Classifier(nn.Module):
    def __init__(self, num_labels: int, dropout_p: float = 0.2):
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


class BoneDenseNet121Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet = models.densenet121(weights=None)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, 1)

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
        dropped = F.dropout(features, p=0.2, training=True)
        logits = self.densenet.classifier(dropped)
        return logits, features


def load_chest_xray_model() -> DenseNet121Classifier:
    local_path = hf_hub_download(repo_id=CHEST_REPO_ID, filename=CHEST_FILENAME)
    state = load_file(local_path)

    model = DenseNet121Classifier(num_labels=14, dropout_p=0.2)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _remap_bone_state_dict_keys(state_dict: dict) -> dict:
    remapped = {}

    for key, value in state_dict.items():
        new_key = key
        if key.startswith("features.") or key.startswith("classifier."):
            new_key = f"densenet.{key}"
        remapped[new_key] = value

    return remapped


def load_bone_xray_model(model_path: str) -> BoneDenseNet121Binary:
    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        raise ValueError(f"Bone model checkpoint not found at: {model_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    state_dict = _remap_bone_state_dict_keys(state_dict)

    model = BoneDenseNet121Binary()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


class EnsembleModel:
    def __init__(self, model_metadata=None, mc_passes: int = 10):
        self.model_metadata = model_metadata
        self.mc_passes = mc_passes
        self.labels: list[str] = []
        self.task_type: str = ""
        self.model = self._load_model_for_route()

    def _normalized_region(self) -> str:
        return str(getattr(self.model_metadata, "region", "") or "").lower()

    def _normalized_modality(self) -> str:
        return str(getattr(self.model_metadata, "modality", "") or "").lower()

    def _model_path(self) -> str:
        return str(getattr(self.model_metadata, "model_path", "") or "")

    def _architecture(self) -> str:
        return str(getattr(self.model_metadata, "architecture", "") or "").lower()

    def _load_model_for_route(self):
        region = self._normalized_region()
        modality = self._normalized_modality()
        architecture = self._architecture()

        if region == "chest" and modality == "xray":
            self.labels = CHEST_XRAY_LABELS
            self.task_type = "multilabel"
            return load_chest_xray_model()

        if region == "bone" and modality == "xray":
            if architecture != "densenet121":
                raise ValueError(f"Unsupported architecture for bone xray route: {architecture}")
            self.labels = BONE_XRAY_LABELS
            self.task_type = "binary"
            return load_bone_xray_model(self._model_path())

        raise ValueError(
            f"No inference implementation is available yet for region='{region}' and modality='{modality}'."
        )

    def _predict_multilabel(self, tensor: torch.Tensor) -> dict:
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
            self.labels[i] for i in range(len(self.labels)) if mean_probs[i] >= 0.5
        ]

        predictive_entropy = float(
            -np.sum(
                mean_probs * np.log(mean_probs + 1e-12)
                + (1.0 - mean_probs) * np.log((1.0 - mean_probs) + 1e-12)
            )
        )

        reliability_score = float(1.0 - np.mean(var_probs))
        reliability_score = max(0.0, min(1.0, reliability_score))

        return {
            "top_label": self.labels[top_idx],
            "top_probability": float(mean_probs[top_idx]),
            "positive_findings": positive_findings,
            "probabilities": {
                self.labels[i]: float(mean_probs[i]) for i in range(len(self.labels))
            },
            "epistemic_uncertainty": {
                self.labels[i]: float(var_probs[i]) for i in range(len(self.labels))
            },
            "aleatoric_uncertainty": {"predictive_entropy": predictive_entropy},
            "reliability_score": reliability_score,
            "disagreement_score": float(np.mean(var_probs)),
            "secondary_verification_triggered": False,
            "ensemble_member_count": 1,
            "model_id": getattr(self.model_metadata, "model_id", "unknown_model"),
            "model_version": getattr(self.model_metadata, "version", "unknown_version"),
            "features": feature_vector.tolist() if feature_vector is not None else None,
        }

    def _predict_binary(self, tensor: torch.Tensor) -> dict:
        mc_probs = []

        with torch.no_grad():
            feature_vector = None

            for i in range(self.mc_passes):
                logits, features = self.model.forward_with_mc_dropout(tensor)
                abnormal_prob = float(torch.sigmoid(logits).squeeze().cpu().item())
                normal_prob = 1.0 - abnormal_prob
                mc_probs.append([normal_prob, abnormal_prob])

                if i == 0:
                    feature_vector = features.squeeze(0).cpu().numpy()

        mc_probs = np.array(mc_probs, dtype=np.float32)
        mean_probs = mc_probs.mean(axis=0)
        var_probs = mc_probs.var(axis=0)

        top_idx = int(np.argmax(mean_probs))
        predictive_entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-12)))

        reliability_score = float(1.0 - np.mean(var_probs))
        reliability_score = max(0.0, min(1.0, reliability_score))

        return {
            "top_label": self.labels[top_idx],
            "top_probability": float(mean_probs[top_idx]),
            "positive_findings": [self.labels[top_idx]],
            "probabilities": {
                self.labels[i]: float(mean_probs[i]) for i in range(len(self.labels))
            },
            "epistemic_uncertainty": {
                self.labels[i]: float(var_probs[i]) for i in range(len(self.labels))
            },
            "aleatoric_uncertainty": {"predictive_entropy": predictive_entropy},
            "reliability_score": reliability_score,
            "disagreement_score": float(np.mean(var_probs)),
            "secondary_verification_triggered": False,
            "ensemble_member_count": 1,
            "model_id": getattr(self.model_metadata, "model_id", "unknown_model"),
            "model_version": getattr(self.model_metadata, "version", "unknown_version"),
            "features": feature_vector.tolist() if feature_vector is not None else None,
        }

    def predict(self, tensor: torch.Tensor) -> dict:
        if self.task_type == "multilabel":
            return self._predict_multilabel(tensor)

        if self.task_type == "binary":
            return self._predict_binary(tensor)

        raise ValueError("Unsupported inference task type.")