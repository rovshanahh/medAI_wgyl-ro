from pathlib import Path
import json
from threading import Lock

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

BRAIN_MRI_LABELS = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary",
]

RETINA_FUNDUS_LABELS = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]

SKIN_DERMOSCOPY_LABELS = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like lesion",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevus",
    "Vascular lesion",
]

BREAST_MAMMOGRAPHY_LABELS = [
    "Benign",
    "Malignant",
]

ABDOMEN_CT_LABELS = [
    "Cyst",
    "Normal",
    "Stone",
    "Tumor",
]

CHEST_REPO_ID = "itsomk/chexpert-densenet121"
CHEST_FILENAME = "pytorch_model.safetensors"

CALIBRATION_PATH = Path("reference_data/calibration/temperature_scaling.json")


_MODEL_CACHE: dict[str, dict] = {}
_MODEL_CACHE_LOCK = Lock()



def load_temperature_config() -> dict:
    if not CALIBRATION_PATH.exists():
        return {
            "default_temperature": 1.0,
            "models": {},
        }

    try:
        return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "default_temperature": 1.0,
            "models": {},
        }


def safe_temperature(value) -> float:
    try:
        temperature = float(value)
    except Exception:
        return 1.0

    if temperature <= 0:
        return 1.0

    return max(0.05, min(10.0, temperature))


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


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
    def __init__(self, dropout_p: float = 0.2):
        super().__init__()
        self.densenet = models.densenet121(weights=None)
        self.dropout_p = dropout_p
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
        dropped = F.dropout(features, p=self.dropout_p, training=True)
        logits = self.densenet.classifier(dropped)
        return logits, features


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int, dropout_p: float = 0.2):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.dropout_p = dropout_p

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = F.relu(x, inplace=False)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.resnet.fc(features)

    def forward_with_mc_dropout(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        dropped = F.dropout(features, p=self.dropout_p, training=True)
        logits = self.resnet.fc(dropped)
        return logits, features


class BrainMriResNet18(ResNet18Classifier):
    def __init__(self, num_classes: int = 4, dropout_p: float = 0.2):
        super().__init__(num_classes=num_classes, dropout_p=dropout_p)


class RetinaFundusResNet18(ResNet18Classifier):
    def __init__(self, num_classes: int = 5, dropout_p: float = 0.2):
        super().__init__(num_classes=num_classes, dropout_p=dropout_p)


class SkinDermoscopyResNet18(ResNet18Classifier):
    def __init__(self, num_classes: int = 7, dropout_p: float = 0.2):
        super().__init__(num_classes=num_classes, dropout_p=dropout_p)


class BreastMammographyResNet18(ResNet18Classifier):
    def __init__(self, num_classes: int = 2, dropout_p: float = 0.2):
        super().__init__(num_classes=num_classes, dropout_p=dropout_p)


class AbdomenCtResNet18(ResNet18Classifier):
    def __init__(self, num_classes: int = 4, dropout_p: float = 0.2):
        super().__init__(num_classes=num_classes, dropout_p=dropout_p)


def load_chest_xray_model(device: torch.device) -> DenseNet121Classifier:
    local_path = hf_hub_download(repo_id=CHEST_REPO_ID, filename=CHEST_FILENAME)
    state = load_file(local_path)

    model = DenseNet121Classifier(num_labels=len(CHEST_XRAY_LABELS), dropout_p=0.2)
    model.load_state_dict(state, strict=True)
    model.to(device)
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


def _remap_resnet_state_dict_keys(state_dict: dict) -> dict:
    remapped = {}

    for key, value in state_dict.items():
        if not key.startswith("resnet."):
            remapped[f"resnet.{key}"] = value
        else:
            remapped[key] = value

    return remapped


def load_bone_xray_model(model_path: str, device: torch.device) -> BoneDenseNet121Binary:
    checkpoint_path = Path(model_path)

    if not checkpoint_path.exists():
        raise ValueError(f"Bone model checkpoint not found at: {model_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    state_dict = _remap_bone_state_dict_keys(state_dict)

    model = BoneDenseNet121Binary(dropout_p=0.2)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


def load_brain_mri_model(model_path: str, device: torch.device) -> BrainMriResNet18:
    checkpoint_path = Path(model_path)

    if not checkpoint_path.exists():
        raise ValueError(f"Brain MRI model checkpoint not found at: {model_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        num_classes = int(checkpoint.get("num_classes", len(BRAIN_MRI_LABELS)))
    else:
        state_dict = checkpoint
        num_classes = len(BRAIN_MRI_LABELS)

    state_dict = _remap_resnet_state_dict_keys(state_dict)

    model = BrainMriResNet18(num_classes=num_classes, dropout_p=0.2)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


def load_retina_fundus_model(model_path: str, device: torch.device) -> RetinaFundusResNet18:
    checkpoint_path = Path(model_path)

    if not checkpoint_path.exists():
        raise ValueError(f"Retina fundus model checkpoint not found at: {model_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        num_classes = int(checkpoint.get("num_classes", len(RETINA_FUNDUS_LABELS)))
    else:
        state_dict = checkpoint
        num_classes = len(RETINA_FUNDUS_LABELS)

    state_dict = _remap_resnet_state_dict_keys(state_dict)

    model = RetinaFundusResNet18(num_classes=num_classes, dropout_p=0.2)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


def load_skin_dermoscopy_model(model_path: str, device: torch.device) -> SkinDermoscopyResNet18:
    checkpoint_path = Path(model_path)

    if not checkpoint_path.exists():
        raise ValueError(f"Skin dermoscopy model checkpoint not found at: {model_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        num_classes = int(checkpoint.get("num_classes", len(SKIN_DERMOSCOPY_LABELS)))
    else:
        state_dict = checkpoint
        num_classes = len(SKIN_DERMOSCOPY_LABELS)

    state_dict = _remap_resnet_state_dict_keys(state_dict)

    model = SkinDermoscopyResNet18(num_classes=num_classes, dropout_p=0.2)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


def load_breast_mammography_model(
    model_path: str,
    device: torch.device,
) -> BreastMammographyResNet18:
    checkpoint_path = Path(model_path)

    if not checkpoint_path.exists():
        raise ValueError(f"Breast mammography model checkpoint not found at: {model_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        num_classes = int(checkpoint.get("num_classes", len(BREAST_MAMMOGRAPHY_LABELS)))
    else:
        state_dict = checkpoint
        num_classes = len(BREAST_MAMMOGRAPHY_LABELS)

    state_dict = _remap_resnet_state_dict_keys(state_dict)

    model = BreastMammographyResNet18(num_classes=num_classes, dropout_p=0.2)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


def load_abdomen_ct_model(
    model_path: str,
    device: torch.device,
) -> AbdomenCtResNet18:
    checkpoint_path = Path(model_path)

    if not checkpoint_path.exists():
        raise ValueError(f"Abdomen CT model checkpoint not found at: {model_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        num_classes = int(checkpoint.get("num_classes", len(ABDOMEN_CT_LABELS)))
    else:
        state_dict = checkpoint
        num_classes = len(ABDOMEN_CT_LABELS)

    state_dict = _remap_resnet_state_dict_keys(state_dict)

    model = AbdomenCtResNet18(num_classes=num_classes, dropout_p=0.2)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


class EnsembleModel:
    def __init__(self, model_metadata=None, mc_passes: int = 10):
        self.model_metadata = model_metadata
        self.mc_passes = mc_passes
        self.device = get_device()
        self.temperature_config = load_temperature_config()
        self.temperature = self._load_temperature()
        self.calibration_method = self._load_calibration_method()

        cached = self._get_or_load_cached_model()

        self.model = cached["model"]
        self.labels = cached["labels"]
        self.task_type = cached["task_type"]
        self.model_cache_key = cached["cache_key"]

    def _normalized_region(self) -> str:
        return str(getattr(self.model_metadata, "region", "") or "").strip().lower()

    def _normalized_modality(self) -> str:
        return str(getattr(self.model_metadata, "modality", "") or "").strip().lower()

    def _model_path(self) -> str:
        return str(getattr(self.model_metadata, "model_path", "") or "")

    def _architecture(self) -> str:
        return str(getattr(self.model_metadata, "architecture", "") or "").strip().lower()

    def _model_id(self) -> str:
        return str(getattr(self.model_metadata, "model_id", "unknown_model") or "unknown_model")

    def _version(self) -> str:
        return str(getattr(self.model_metadata, "version", "unknown_version") or "unknown_version")

    def _load_temperature(self) -> float:
        model_id = self._model_id()
        default_temperature = self.temperature_config.get("default_temperature", 1.0)
        model_config = self.temperature_config.get("models", {}).get(model_id, {})
        return safe_temperature(model_config.get("temperature", default_temperature))

    def _load_calibration_method(self) -> str:
        model_id = self._model_id()
        model_config = self.temperature_config.get("models", {}).get(model_id, {})
        return str(model_config.get("method", "none"))

    def _calibration_payload(self) -> dict:
        return {
            "enabled": self.temperature != 1.0,
            "method": self.calibration_method,
            "temperature": float(self.temperature),
            "config_path": str(CALIBRATION_PATH),
        }

    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def _cache_key(self) -> str:
        return f"{self._model_id()}:{self._version()}:{self.device.type}"

    def _get_or_load_cached_model(self) -> dict:
        cache_key = self._cache_key()

        with _MODEL_CACHE_LOCK:
            if cache_key in _MODEL_CACHE:
                return _MODEL_CACHE[cache_key]

            loaded = self._load_model_for_route()
            loaded["cache_key"] = cache_key
            _MODEL_CACHE[cache_key] = loaded

            return loaded

    def _load_model_for_route(self) -> dict:
        region = self._normalized_region()
        modality = self._normalized_modality()
        architecture = self._architecture()

        if region == "chest" and modality == "xray":
            return {
                "model": load_chest_xray_model(self.device),
                "labels": CHEST_XRAY_LABELS,
                "task_type": "multilabel",
            }

        if region == "bone" and modality == "xray":
            if architecture != "densenet121":
                raise ValueError(f"Unsupported architecture for bone xray route: {architecture}")

            return {
                "model": load_bone_xray_model(self._model_path(), self.device),
                "labels": BONE_XRAY_LABELS,
                "task_type": "binary",
            }

        if region == "brain" and modality == "mri":
            if architecture != "resnet18":
                raise ValueError(f"Unsupported architecture for brain MRI route: {architecture}")

            return {
                "model": load_brain_mri_model(self._model_path(), self.device),
                "labels": BRAIN_MRI_LABELS,
                "task_type": "multiclass",
            }

        if region == "retina" and modality == "fundus":
            if architecture != "resnet18":
                raise ValueError(f"Unsupported architecture for retina fundus route: {architecture}")

            return {
                "model": load_retina_fundus_model(self._model_path(), self.device),
                "labels": RETINA_FUNDUS_LABELS,
                "task_type": "multiclass",
            }

        if region == "skin" and modality == "dermoscopy":
            if architecture != "resnet18":
                raise ValueError(f"Unsupported architecture for skin dermoscopy route: {architecture}")

            return {
                "model": load_skin_dermoscopy_model(self._model_path(), self.device),
                "labels": SKIN_DERMOSCOPY_LABELS,
                "task_type": "multiclass",
            }

        if region == "breast" and modality == "mammography":
            if architecture != "resnet18":
                raise ValueError(
                    f"Unsupported architecture for breast mammography route: {architecture}"
                )

            return {
                "model": load_breast_mammography_model(self._model_path(), self.device),
                "labels": BREAST_MAMMOGRAPHY_LABELS,
                "task_type": "multiclass",
            }

        if region == "abdomen" and modality == "ct":
            if architecture != "resnet18":
                raise ValueError(f"Unsupported architecture for abdomen CT route: {architecture}")

            return {
                "model": load_abdomen_ct_model(self._model_path(), self.device),
                "labels": ABDOMEN_CT_LABELS,
                "task_type": "multiclass",
            }

        raise ValueError(
            f"No inference implementation is available yet for region='{region}' "
            f"and modality='{modality}'."
        )

    def _prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Inference input must be a torch.Tensor")

        if tensor.ndim != 4:
            raise ValueError(f"Expected tensor shape [B, C, H, W], got {tuple(tensor.shape)}")

        return tensor.detach().float().to(self.device)

    def _base_payload(
        self,
        top_label: str,
        top_probability: float,
        positive_findings: list[str],
        probabilities: dict,
        epistemic_uncertainty: dict,
        predictive_entropy: float,
        reliability_score: float,
        disagreement_score: float,
        feature_vector,
    ) -> dict:
        reliability_score = float(max(0.0, min(1.0, reliability_score)))
        disagreement_score = float(max(0.0, disagreement_score))
        predictive_entropy = float(max(0.0, predictive_entropy))

        epistemic_values = [
            float(value)
            for value in (epistemic_uncertainty or {}).values()
        ]
        mean_epistemic = (
            float(sum(epistemic_values) / len(epistemic_values))
            if epistemic_values
            else 0.0
        )
        max_epistemic = max(epistemic_values) if epistemic_values else 0.0

        uncertainty_level = "LOW"

        if (
            float(top_probability) < 0.60
            or reliability_score < 0.75
            or mean_epistemic > 0.01
            or disagreement_score > 0.01
        ):
            uncertainty_level = "HIGH"
        elif (
            float(top_probability) < 0.70
            or reliability_score < 0.85
            or mean_epistemic > 0.003
            or disagreement_score > 0.003
        ):
            uncertainty_level = "MODERATE"

        return {
            "top_label": top_label,
            "top_probability": float(top_probability),
            "positive_findings": positive_findings,
            "probabilities": probabilities,

            "epistemic_uncertainty": epistemic_uncertainty,
            "aleatoric_uncertainty": {
                "predictive_entropy": predictive_entropy
            },

            "uncertainty_summary": {
                "level": uncertainty_level,
                "method": "single_model_mc_dropout_proxy",
                "mean_epistemic": mean_epistemic,
                "max_epistemic": float(max_epistemic),
                "predictive_entropy": predictive_entropy,
                "disagreement_score": disagreement_score,
                "reliability_score": reliability_score,
                "interpretation": (
                    "Lower uncertainty means the repeated stochastic predictions were more stable. "
                    "This is an MVP uncertainty proxy, not a full clinical reliability guarantee."
                ),
            },

            "reliability_score": reliability_score,
            "disagreement_score": disagreement_score,
            "secondary_verification_triggered": uncertainty_level in {"MODERATE", "HIGH"},

            "ensemble_member_count": 1,
            "mc_passes": self.mc_passes,
            "uncertainty_method": "single_model_mc_dropout_proxy",
            "deep_ensemble_enabled": False,
            "uncertainty_note": (
                "This MVP uses one model with MC-dropout-style stochastic passes as an uncertainty proxy. "
                "A true deep ensemble requires at least three independently trained checkpoints."
            ),

            "model_id": self._model_id(),
            "model_version": self._version(),
            "model_cache_key": self.model_cache_key,
            "device": self.device.type,
            "calibration": self._calibration_payload(),
            "features": feature_vector.tolist() if feature_vector is not None else None,
        }

    def _predict_multilabel(self, tensor: torch.Tensor) -> dict:
        tensor = self._prepare_tensor(tensor)
        mc_probs = []

        with torch.no_grad():
            feature_vector = None

            for i in range(self.mc_passes):
                logits, features = self.model.forward_with_mc_dropout(tensor)
                calibrated_logits = self._apply_temperature(logits)
                probs = torch.sigmoid(calibrated_logits).squeeze(0).detach().cpu().numpy()
                mc_probs.append(probs)

                if i == 0:
                    feature_vector = features.squeeze(0).detach().cpu().numpy()

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

        disagreement_score = float(np.mean(var_probs))
        reliability_score = float(1.0 - disagreement_score)

        return self._base_payload(
            top_label=self.labels[top_idx],
            top_probability=float(mean_probs[top_idx]),
            positive_findings=positive_findings,
            probabilities={
                self.labels[i]: float(mean_probs[i]) for i in range(len(self.labels))
            },
            epistemic_uncertainty={
                self.labels[i]: float(var_probs[i]) for i in range(len(self.labels))
            },
            predictive_entropy=predictive_entropy,
            reliability_score=reliability_score,
            disagreement_score=disagreement_score,
            feature_vector=feature_vector,
        )

    def _predict_binary(self, tensor: torch.Tensor) -> dict:
        tensor = self._prepare_tensor(tensor)
        mc_probs = []

        with torch.no_grad():
            feature_vector = None

            for i in range(self.mc_passes):
                logits, features = self.model.forward_with_mc_dropout(tensor)
                calibrated_logits = self._apply_temperature(logits)
                abnormal_prob = float(torch.sigmoid(calibrated_logits).squeeze().detach().cpu().item())
                normal_prob = 1.0 - abnormal_prob
                mc_probs.append([normal_prob, abnormal_prob])

                if i == 0:
                    feature_vector = features.squeeze(0).detach().cpu().numpy()

        mc_probs = np.array(mc_probs, dtype=np.float32)
        mean_probs = mc_probs.mean(axis=0)
        var_probs = mc_probs.var(axis=0)

        top_idx = int(np.argmax(mean_probs))
        predictive_entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-12)))
        disagreement_score = float(np.mean(var_probs))
        reliability_score = float(1.0 - disagreement_score)

        return self._base_payload(
            top_label=self.labels[top_idx],
            top_probability=float(mean_probs[top_idx]),
            positive_findings=[self.labels[top_idx]],
            probabilities={
                self.labels[i]: float(mean_probs[i]) for i in range(len(self.labels))
            },
            epistemic_uncertainty={
                self.labels[i]: float(var_probs[i]) for i in range(len(self.labels))
            },
            predictive_entropy=predictive_entropy,
            reliability_score=reliability_score,
            disagreement_score=disagreement_score,
            feature_vector=feature_vector,
        )

    def _predict_multiclass(self, tensor: torch.Tensor) -> dict:
        tensor = self._prepare_tensor(tensor)
        mc_probs = []

        with torch.no_grad():
            feature_vector = None

            for i in range(self.mc_passes):
                logits, features = self.model.forward_with_mc_dropout(tensor)
                calibrated_logits = self._apply_temperature(logits)
                probs = torch.softmax(calibrated_logits, dim=1).squeeze(0).detach().cpu().numpy()
                mc_probs.append(probs)

                if i == 0:
                    feature_vector = features.squeeze(0).detach().cpu().numpy()

        mc_probs = np.array(mc_probs, dtype=np.float32)
        mean_probs = mc_probs.mean(axis=0)
        var_probs = mc_probs.var(axis=0)

        top_idx = int(np.argmax(mean_probs))
        predictive_entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-12)))
        disagreement_score = float(np.mean(var_probs))
        reliability_score = float(1.0 - disagreement_score)

        return self._base_payload(
            top_label=self.labels[top_idx],
            top_probability=float(mean_probs[top_idx]),
            positive_findings=[self.labels[top_idx]],
            probabilities={
                self.labels[i]: float(mean_probs[i]) for i in range(len(self.labels))
            },
            epistemic_uncertainty={
                self.labels[i]: float(var_probs[i]) for i in range(len(self.labels))
            },
            predictive_entropy=predictive_entropy,
            reliability_score=reliability_score,
            disagreement_score=disagreement_score,
            feature_vector=feature_vector,
        )

    def predict(self, tensor: torch.Tensor) -> dict:
        if self.task_type == "multilabel":
            return self._predict_multilabel(tensor)

        if self.task_type == "binary":
            return self._predict_binary(tensor)

        if self.task_type == "multiclass":
            return self._predict_multiclass(tensor)

        raise ValueError("Unsupported inference task type.")

    @staticmethod
    def clear_cache() -> None:
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE.clear()

    @staticmethod
    def cache_info() -> dict:
        with _MODEL_CACHE_LOCK:
            return {
                "cached_models": list(_MODEL_CACHE.keys()),
                "count": len(_MODEL_CACHE),
            }