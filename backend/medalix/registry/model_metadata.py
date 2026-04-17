from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    version: str
    architecture: str
    region: str
    modality: str
    input_shape: tuple[int, ...]
    status: str
    checksum: str
    model_path: str
    ensemble_members: list[dict[str, Any]] = field(default_factory=list)

    def is_active(self) -> bool:
        return self.status.upper() == "ACTIVE"

    def supports_shape(self, shape: tuple[int, ...]) -> bool:
        return tuple(shape) == tuple(self.input_shape)

    def has_ensemble(self) -> bool:
        return len(self.ensemble_members) >= 3