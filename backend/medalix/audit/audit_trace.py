from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuditTrace:
    filename: str
    input_gate: dict[str, Any]
    quality: dict[str, Any]
    ood: dict[str, Any]
    routing: dict[str, Any]
    policy: dict[str, Any]
    inference_summary: dict[str, Any] = field(default_factory=dict)