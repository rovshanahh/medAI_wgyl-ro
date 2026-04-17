from dataclasses import dataclass, field
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
    routing_result: dict | None = None
    policy_result: dict | None = None
    selected_model: Any = None
    trace: Any = None

    warnings: list[str] = field(default_factory=list)
    stage_history: list[str] = field(default_factory=list)

    def set_stage(self, stage: str) -> None:
        self.current_stage = stage
        self.stage_history.append(stage)