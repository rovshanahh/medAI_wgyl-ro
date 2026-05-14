from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class PipelineState:
    analysis_id: str
    filename: str
    content_type: str
    size_bytes: int
    temp_path: str | None = None
    current_stage: str = "created"

    input_gate_result: dict | None = None
    detection_result: dict | None = None
    preprocessing_result: Any = None
    tensor: Any = None
    original_image: Any = None
    inference_result: dict | None = None
    explainability_result: dict | None = None
    quality_result: dict | None = None
    ood_result: dict | None = None
    pre_inference_safety_result: dict | None = None
    routing_result: dict | None = None
    policy_result: dict | None = None
    selected_model: Any = None
    trace: Any = None

    warnings: list[str] = field(default_factory=list)

    stage_history: list[str] = field(default_factory=list)
    stage_events: list[dict] = field(default_factory=list)

    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def set_stage(self, stage: str) -> None:
        now = datetime.now(UTC).isoformat()

        self.current_stage = stage
        self.updated_at = now
        self.stage_history.append(stage)

        self.stage_events.append(
            {
                "stage": stage,
                "timestamp": now,
            }
        )

    def timing_summary(self) -> list[dict]:
        if not self.stage_events:
            return []

        summary = []

        for index, event in enumerate(self.stage_events):
            started_at = event["timestamp"]
            ended_at = (
                self.stage_events[index + 1]["timestamp"]
                if index + 1 < len(self.stage_events)
                else self.updated_at
            )

            summary.append(
                {
                    "stage": event["stage"],
                    "started_at": started_at,
                    "ended_at": ended_at,
                }
            )

        return summary

    def to_debug_dict(self) -> dict:
        return {
            "analysis_id": self.analysis_id,
            "filename": self.filename,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "temp_path": self.temp_path,
            "current_stage": self.current_stage,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stage_history": list(self.stage_history),
            "stage_events": list(self.stage_events),
            "timing_summary": self.timing_summary(),
            "warnings": list(self.warnings),
        }