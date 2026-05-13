import json
from pathlib import Path

from .checksum_validator import ChecksumValidator
from .model_metadata import ModelMetadata
from .routing_table import RoutingTable


class ModelRegistry:
    REMOTE_MODEL_IDS = {"chest_xray_mvp"}

    def __init__(self, registry_path: str) -> None:
        self._registry_path = Path(registry_path)
        self._checksum_validator = ChecksumValidator()
        self._registry_version = ""
        self._routing_table = None
        self._models: dict[str, ModelMetadata] = {}
        self._load()

    def _load(self) -> None:
        if not self._registry_path.exists():
            raise FileNotFoundError(f"Model registry not found: {self._registry_path}")

        with self._registry_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self._registry_version = data["registry_version"]
        self._routing_table = RoutingTable(data["routes"])

        self._models = {
            model_id: ModelMetadata(
                model_id=model["model_id"],
                version=model["version"],
                architecture=model["architecture"],
                region=model["region"],
                modality=model["modality"],
                input_shape=tuple(model["input_shape"]),
                status=model["status"],
                checksum=model.get("checksum", ""),
                model_path=model.get("model_path", ""),
                ensemble_members=model.get("ensemble_members", []),
            )
            for model_id, model in data["models"].items()
        }

    @property
    def registry_version(self) -> str:
        return self._registry_version

    def lookup_by_route(self, region: str, modality: str) -> ModelMetadata:
        model_id = self._routing_table.resolve(region, modality)

        if model_id not in self._models:
            raise ValueError(f"Model id {model_id} not found in registry")

        return self._models[model_id]

    def validate_model(self, metadata: ModelMetadata) -> None:
        status = metadata.status.upper()

        if status in {"DEPRECATED", "BLOCKED"}:
            raise ValueError(
                f"Model {metadata.model_id} is not executable; status={metadata.status}"
            )

        if status != "ACTIVE":
            raise ValueError(
                f"Model {metadata.model_id} is registered but not active; status={metadata.status}"
            )

        if metadata.ensemble_members:
            if len(metadata.ensemble_members) < 3:
                raise ValueError(
                    f"Model {metadata.model_id} must define at least 3 ensemble members"
                )

            for member_path in metadata.ensemble_members:
                if not Path(member_path).exists():
                    raise ValueError(
                        f"Ensemble member checkpoint not found for model "
                        f"{metadata.model_id}: {member_path}"
                    )

            return

        if metadata.model_id in self.REMOTE_MODEL_IDS:
            return

        if not metadata.model_path:
            raise ValueError(
                f"Model {metadata.model_id} has no local model_path and is not registered as a remote model"
            )

        model_path = Path(metadata.model_path)

        if not model_path.exists():
            raise ValueError(
                f"Model {metadata.model_id} checkpoint not found at {metadata.model_path}"
            )

        if metadata.checksum:
            if not self._checksum_validator.validate(metadata.model_path, metadata.checksum):
                raise ValueError(f"Checksum mismatch for model {metadata.model_id}")