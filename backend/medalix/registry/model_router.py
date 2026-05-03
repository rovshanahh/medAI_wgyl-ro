from .model_metadata import ModelMetadata
from .model_registry import ModelRegistry


class ModelRouter:
    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def resolve(
        self,
        region: str,
        modality: str,
        input_shape: tuple[int, ...],
    ) -> ModelMetadata:
        normalized_region = str(region or "").strip().lower()
        normalized_modality = str(modality or "").strip().lower()

        if not normalized_region or not normalized_modality:
            raise ValueError(
                f"Cannot resolve model route with region='{region}' and modality='{modality}'"
            )

        metadata = self._registry.lookup_by_route(
            normalized_region,
            normalized_modality,
        )

        self._registry.validate_model(metadata)

        if metadata.normalized_region() != normalized_region:
            raise ValueError(
                f"Region mismatch between route and model metadata. "
                f"Route={normalized_region}, model={metadata.region}"
            )

        if metadata.normalized_modality() != normalized_modality:
            raise ValueError(
                f"Modality mismatch between route and model metadata. "
                f"Route={normalized_modality}, model={metadata.modality}"
            )

        if not metadata.supports_shape(input_shape):
            raise ValueError(
                f"Input shape mismatch for model {metadata.model_id}. "
                f"Expected {metadata.input_shape}, got {input_shape}"
            )

        return metadata