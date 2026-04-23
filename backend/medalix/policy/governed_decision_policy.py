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

    def classify_risk(
        self,
        ood_result: dict,
        routing_result: dict,
        inference_result: dict,
    ) -> str:
        if ood_result.get("tier") in {"HARD_OOD", "NEAR_OOD"}:
            return "HIGH"

        set_size = int(routing_result.get("set_size", 0) or 0)

        epi_dict = inference_result.get("epistemic_uncertainty", {})
        avg_epi = 0.0
        if epi_dict:
            avg_epi = sum(float(v) for v in epi_dict.values()) / len(epi_dict)

        moderate_set_size = self._config.get("moderate_set_size", 2)
        high_set_size = self._config.get("high_set_size", 3)
        moderate_epistemic = self._config.get("moderate_epistemic_threshold", 0.003)
        high_epistemic = self._config.get("high_epistemic_threshold", 0.01)

        if set_size >= high_set_size or avg_epi > high_epistemic:
            return "HIGH"

        if set_size >= moderate_set_size or avg_epi > moderate_epistemic:
            return "MODERATE"

        return "LOW"

    def build_warnings(self, action: str, risk_category: str) -> list[str]:
        warnings = []

        if risk_category == "HIGH":
            warnings.append("High-risk output")
            warnings.append("Review recommended")
        elif risk_category == "MODERATE":
            warnings.append("Moderate-risk output")

        if action == "ESCALATE":
            warnings.append("Manual confirmation recommended")
        if action == "REQUEST_EVIDENCE":
            warnings.append("Additional evidence required")
        if action == "REFUSE":
            warnings.append("Prediction withheld due to low reliability")
        if action == "STOP":
            warnings.append("Inference blocked for safety reasons")

        return warnings

    def evaluate(self, policy_input: PolicyInput) -> PolicyOutput:
        self._ensure_complete_input(policy_input)

        ood_result = policy_input.ood_result or {}
        routing_result = policy_input.routing_result or {}
        inference_result = policy_input.inference_result or {}
        quality_result = policy_input.quality_result or {}

        risk_category = self.classify_risk(
            ood_result=ood_result,
            routing_result=routing_result,
            inference_result=inference_result,
        )

        answer_threshold = float(self._config.get("answer_threshold", 0.70))
        refusal_threshold = float(self._config.get("refusal_threshold", 0.80))
        escalation_threshold = float(self._config.get("escalation_threshold", 0.90))
        refuse_epistemic_threshold = float(
            self._config.get("refuse_epistemic_threshold", 0.02)
        )

        top_prob = float(inference_result.get("top_probability", 0.0) or 0.0)
        reliability_score = float(inference_result.get("reliability_score", 0.0) or 0.0)
        disagreement_score = float(inference_result.get("disagreement_score", 0.0) or 0.0)

        epi_dict = inference_result.get("epistemic_uncertainty", {})
        avg_epi = 0.0
        if epi_dict:
            avg_epi = sum(float(v) for v in epi_dict.values()) / len(epi_dict)

        if ood_result.get("is_hard_ood", False):
            action = "STOP"
            reason = "Hard OOD detected"
            return PolicyOutput(
                action=action,
                reason=reason,
                risk_category="HIGH",
                warnings=self.build_warnings(action, "HIGH"),
            )

        if ood_result.get("tier") == "NEAR_OOD":
            action = "STOP"
            reason = "Near-OOD input detected"
            return PolicyOutput(
                action=action,
                reason=reason,
                risk_category="HIGH",
                warnings=self.build_warnings(action, "HIGH"),
            )

        if quality_result.get("blocking", False):
            action = "REQUEST_EVIDENCE"
            reason = quality_result.get("reason", "Quality check failed")
            return PolicyOutput(
                action=action,
                reason=reason,
                risk_category=self._max_risk(risk_category, "MODERATE"),
                warnings=self.build_warnings(action, self._max_risk(risk_category, "MODERATE")),
            )

        if routing_result.get("requires_confirmation", False):
            action = "ESCALATE"
            reason = "Routing confidence too low"
            return PolicyOutput(
                action=action,
                reason=reason,
                risk_category=self._max_risk(risk_category, "MODERATE"),
                warnings=self.build_warnings(action, self._max_risk(risk_category, "MODERATE")),
            )

        if disagreement_score >= escalation_threshold:
            action = "ESCALATE"
            reason = "Ensemble disagreement exceeds escalation threshold"
            return PolicyOutput(
                action=action,
                reason=reason,
                risk_category="HIGH",
                warnings=self.build_warnings(action, "HIGH"),
            )

        if reliability_score < refusal_threshold:
            action = "REFUSE"
            reason = "Prediction reliability is below refusal threshold"
            elevated_risk = self._max_risk(risk_category, "MODERATE")
            return PolicyOutput(
                action=action,
                reason=reason,
                risk_category=elevated_risk,
                warnings=self.build_warnings(action, elevated_risk),
            )

        if top_prob < answer_threshold and avg_epi > refuse_epistemic_threshold:
            action = "REFUSE"
            reason = "Prediction not reliable enough to answer safely"
            return PolicyOutput(
                action=action,
                reason=reason,
                risk_category=self._max_risk(risk_category, "MODERATE"),
                warnings=self.build_warnings(action, self._max_risk(risk_category, "MODERATE")),
            )

        action = "ANSWER"
        reason = "All gates passed"
        return PolicyOutput(
            action=action,
            reason=reason,
            risk_category=risk_category,
            warnings=self.build_warnings(action, risk_category),
        )