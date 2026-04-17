from .model_metadata import ModelMetadata
from .model_registry import ModelRegistry


class ModelRouter:
    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def resolve(self, region: str, modality: str, input_shape: tuple[int, ...]) -> ModelMetadata:
        metadata = self._registry.lookup_by_route(region, modality)
        self._registry.validate_model(metadata)

        if metadata.region.lower() != region.lower():
            raise ValueError("Region mismatch between route and model metadata")
        if metadata.modality.lower() != modality.lower():
            raise ValueError("Modality mismatch between route and model metadata")
        if not metadata.supports_shape(input_shape):
            raise ValueError(
                f"Input shape mismatch. Expected {metadata.input_shape}, got {input_shape}"
            )

        return metadata