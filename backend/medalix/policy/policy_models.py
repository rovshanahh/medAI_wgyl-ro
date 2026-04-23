from dataclasses import dataclass, field


VALID_ACTIONS = {
    "ANSWER",
    "REFUSE",
    "ESCALATE",
    "REQUEST_EVIDENCE",
    "STOP",
}

VALID_RISK_CATEGORIES = {
    "LOW",
    "MODERATE",
    "HIGH",
}


@dataclass(frozen=True)
class PolicyInput:
    ood_result: dict
    routing_result: dict
    inference_result: dict
    quality_result: dict

    def is_complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.ood_result,
                self.routing_result,
                self.inference_result,
                self.quality_result,
            )
        )

    def to_dict(self) -> dict:
        return {
            "ood_result": dict(self.ood_result or {}),
            "routing_result": dict(self.routing_result or {}),
            "inference_result": dict(self.inference_result or {}),
            "quality_result": dict(self.quality_result or {}),
        }


@dataclass(frozen=True)
class PolicyOutput:
    action: str
    reason: str
    risk_category: str
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid policy action: {self.action}. Expected one of: {sorted(VALID_ACTIONS)}"
            )

        if self.risk_category not in VALID_RISK_CATEGORIES:
            raise ValueError(
                f"Invalid risk category: {self.risk_category}. Expected one of: {sorted(VALID_RISK_CATEGORIES)}"
            )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("PolicyOutput.reason must be a non-empty string")

        if self.warnings is None:
            object.__setattr__(self, "warnings", [])
        else:
            object.__setattr__(
                self,
                "warnings",
                [str(w).strip() for w in self.warnings if str(w).strip()],
            )

    def is_terminal(self) -> bool:
        return self.action in {"ANSWER", "REFUSE", "ESCALATE", "STOP"}

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "risk_category": self.risk_category,
            "warnings": list(self.warnings),
        }