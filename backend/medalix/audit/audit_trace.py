from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AuditTrace:
    filename: str
    input_gate: dict
    detection: dict
    routing: dict
    selected_model: dict
    ood: dict
    quality: dict
    inference_summary: dict
    policy: dict
    explainability: dict
    pipeline_stages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "input_gate": dict(self.input_gate or {}),
            "detection": dict(self.detection or {}),
            "routing": dict(self.routing or {}),
            "selected_model": dict(self.selected_model or {}),
            "ood": dict(self.ood or {}),
            "quality": dict(self.quality or {}),
            "inference_summary": dict(self.inference_summary or {}),
            "policy": dict(self.policy or {}),
            "explainability": dict(self.explainability or {}),
            "pipeline_stages": list(self.pipeline_stages or []),
        }