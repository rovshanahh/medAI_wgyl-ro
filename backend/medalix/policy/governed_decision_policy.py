import json
from pathlib import Path

from medalix.policy.policy_models import PolicyInput, PolicyOutput


class GovernedDecisionPolicy:
    def __init__(self, config_path: str = "reference_data/policy_thresholds.json") -> None:
        self._config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> dict:
        with self._config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _max_risk(self, left: str, right: str) -> str:
        order = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
        return left if order.get(left, 0) >= order.get(right, 0) else right

    def _ensure_complete_input(self, policy_input: PolicyInput) -> None:
        if not policy_input.is_complete():
            raise ValueError("PolicyInput is incomplete")

    def _average_epistemic(self, inference_result: dict) -> float:
        epi_dict = inference_result.get("epistemic_uncertainty", {})

        if not epi_dict:
            return 0.0

        return sum(float(v) for v in epi_dict.values()) / len(epi_dict)

    def classify_risk(
        self,
        ood_result: dict,
        routing_result: dict,
        inference_result: dict,
        quality_result: dict,
    ) -> str:
        if ood_result.get("tier") == "HARD_OOD":
            return "HIGH"

        if ood_result.get("tier") == "NEAR_OOD":
            return "MODERATE"

        if quality_result.get("blocking", False):
            return "MODERATE"

        set_size = int(routing_result.get("set_size", 0) or 0)
        requires_confirmation = bool(routing_result.get("requires_confirmation", False))

        avg_epi = self._average_epistemic(inference_result)
        disagreement_score = float(inference_result.get("disagreement_score", 0.0) or 0.0)
        reliability_score = float(inference_result.get("reliability_score", 1.0) or 1.0)

        moderate_set_size = int(self._config.get("moderate_set_size", 2))
        high_set_size = int(self._config.get("high_set_size", 3))

        moderate_epistemic = float(self._config.get("moderate_epistemic_threshold", 0.003))
        high_epistemic = float(self._config.get("high_epistemic_threshold", 0.01))

        moderate_disagreement = float(
            self._config.get("moderate_disagreement_threshold", 0.003)
        )
        high_disagreement = float(self._config.get("high_disagreement_threshold", 0.01))

        moderate_reliability_floor = float(
            self._config.get("moderate_reliability_floor", 0.85)
        )
        high_reliability_floor = float(self._config.get("high_reliability_floor", 0.70))

        if (
            set_size >= high_set_size
            or avg_epi > high_epistemic
            or disagreement_score > high_disagreement
            or reliability_score < high_reliability_floor
        ):
            return "HIGH"

        if (
            requires_confirmation
            or set_size >= moderate_set_size
            or avg_epi > moderate_epistemic
            or disagreement_score > moderate_disagreement
            or reliability_score < moderate_reliability_floor
        ):
            return "MODERATE"

        return "LOW"

    def build_warnings(self, action: str, risk_category: str) -> list[str]:
        warnings = []

        if risk_category == "HIGH":
            warnings.append("High-risk output")
            warnings.append("Manual review recommended")
        elif risk_category == "MODERATE":
            warnings.append("Moderate-risk output")

        if action == "ESCALATE":
            warnings.append("Human review recommended")
        elif action == "REQUEST_EVIDENCE":
            warnings.append("Additional evidence required")
        elif action == "REFUSE":
            warnings.append("Prediction withheld due to low reliability")
        elif action == "STOP":
            warnings.append("Inference blocked for safety reasons")

        return warnings

    def _output(self, action: str, reason: str, risk_category: str) -> PolicyOutput:
        return PolicyOutput(
            action=action,
            reason=reason,
            risk_category=risk_category,
            warnings=self.build_warnings(action, risk_category),
        )

    def evaluate(self, policy_input: PolicyInput) -> PolicyOutput:
        self._ensure_complete_input(policy_input)

        ood_result = policy_input.ood_result or {}
        routing_result = policy_input.routing_result or {}
        inference_result = policy_input.inference_result or {}
        quality_result = policy_input.quality_result or {}

        if not inference_result:
            return self._output(
                action="STOP",
                reason="Inference result is missing, so no answer can be produced safely.",
                risk_category="HIGH",
            )

        risk_category = self.classify_risk(
            ood_result=ood_result,
            routing_result=routing_result,
            inference_result=inference_result,
            quality_result=quality_result,
        )

        answer_threshold = float(self._config.get("answer_threshold", 0.70))
        refuse_probability_threshold = float(
            self._config.get("refuse_probability_threshold", 0.45)
        )
        escalate_probability_threshold = float(
            self._config.get("escalate_probability_threshold", 0.60)
        )
        refuse_reliability_threshold = float(
            self._config.get("refuse_reliability_threshold", 0.60)
        )
        escalate_reliability_threshold = float(
            self._config.get("escalate_reliability_threshold", 0.75)
        )
        refuse_epistemic_threshold = float(
            self._config.get("refuse_epistemic_threshold", 0.02)
        )
        escalate_epistemic_threshold = float(
            self._config.get("escalate_epistemic_threshold", 0.01)
        )
        escalate_disagreement_threshold = float(
            self._config.get("escalate_disagreement_threshold", 0.01)
        )

        top_prob = float(inference_result.get("top_probability", 0.0) or 0.0)
        reliability_score = float(inference_result.get("reliability_score", 0.0) or 0.0)
        disagreement_score = float(inference_result.get("disagreement_score", 0.0) or 0.0)
        avg_epi = self._average_epistemic(inference_result)

        if ood_result.get("is_hard_ood", False) or ood_result.get("tier") == "HARD_OOD":
            return self._output(
                action="STOP",
                reason="Hard out-of-distribution input detected.",
                risk_category="HIGH",
            )

        if ood_result.get("tier") == "NEAR_OOD":
            return self._output(
                action="ESCALATE",
                reason="Input is near the distribution boundary and should be reviewed.",
                risk_category=self._max_risk(risk_category, "MODERATE"),
            )

        if quality_result.get("blocking", False):
            return self._output(
                action="REQUEST_EVIDENCE",
                reason=quality_result.get(
                    "reason",
                    "Image quality is not sufficient for a reliable answer.",
                ),
                risk_category=self._max_risk(risk_category, "MODERATE"),
            )

        if routing_result.get("requires_confirmation", False):
            return self._output(
                action="ESCALATE",
                reason="Routing confidence is not strong enough for automatic answering.",
                risk_category=self._max_risk(risk_category, "MODERATE"),
            )

        if top_prob < refuse_probability_threshold:
            return self._output(
                action="ESCALATE",
                reason="Prediction confidence is very low, so human review is required.",
                risk_category=self._max_risk(risk_category, "HIGH"),
            )

        if reliability_score < refuse_reliability_threshold:
            return self._output(
                action="ESCALATE",
                reason="Prediction reliability is low, so human review is required.",
                risk_category=self._max_risk(risk_category, "HIGH"),
            )

        if avg_epi > refuse_epistemic_threshold:
            return self._output(
                action="ESCALATE",
                reason="Epistemic uncertainty is high, so human review is required.",
                risk_category=self._max_risk(risk_category, "HIGH"),
            )

        if (
            top_prob < escalate_probability_threshold
            or reliability_score < escalate_reliability_threshold
            or avg_epi > escalate_epistemic_threshold
            or disagreement_score > escalate_disagreement_threshold
        ):
            return self._output(
                action="ESCALATE",
                reason="Prediction is possible, but uncertainty suggests human review.",
                risk_category=self._max_risk(risk_category, "MODERATE"),
            )

        if top_prob < answer_threshold:
            return self._output(
                action="ESCALATE",
                reason="Prediction confidence is below the automatic answer threshold.",
                risk_category=self._max_risk(risk_category, "MODERATE"),
            )

        return self._output(
            action="ANSWER",
            reason="Routing, quality, distribution, and uncertainty checks passed.",
            risk_category=risk_category,
        )