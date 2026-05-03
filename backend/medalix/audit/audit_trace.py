from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class AuditTrace:
    analysis_id: str
    filename: str
    content_type: str | None
    size_bytes: int | None
    input_gate: dict
    detection: dict
    routing: dict
    selected_model: dict
    ood: dict
    quality: dict
    inference_summary: dict
    policy: dict
    explainability: dict
    conversion: dict
    pipeline_stages: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "created_at": self.created_at,
            "filename": self.filename,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "input_gate": dict(self.input_gate or {}),
            "detection": dict(self.detection or {}),
            "routing": dict(self.routing or {}),
            "selected_model": dict(self.selected_model or {}),
            "ood": dict(self.ood or {}),
            "quality": dict(self.quality or {}),
            "inference_summary": dict(self.inference_summary or {}),
            "policy": dict(self.policy or {}),
            "explainability": dict(self.explainability or {}),
            "conversion": dict(self.conversion or {}),
            "pipeline_stages": list(self.pipeline_stages or []),
        }