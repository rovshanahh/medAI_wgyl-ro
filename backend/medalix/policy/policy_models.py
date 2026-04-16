from dataclasses import dataclass, field
from typing import Any


@dataclass
class PolicyInput:
    ood_result: dict[str, Any]
    routing_result: dict[str, Any]
    inference_result: dict[str, Any]
    quality_result: dict[str, Any]


@dataclass
class PolicyOutput:
    action: str
    reason: str
    risk_category: str
    warnings: list[str] = field(default_factory=list)