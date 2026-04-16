class GovernedDecisionPolicy:
    def classify_risk(
        self,
        ood_result: dict,
        routing_result: dict,
        inference_result: dict,
    ) -> str:
        if ood_result.get("tier") == "HARD_OOD":
            return "HIGH"

        if ood_result.get("tier") == "NEAR_OOD":
            return "HIGH"

        set_size = routing_result.get("set_size", 0)

        epi_dict = inference_result.get("epistemic_uncertainty", {})
        avg_epi = 0.0
        if epi_dict:
            avg_epi = sum(epi_dict.values()) / len(epi_dict)

        if set_size >= 5 or avg_epi > 0.01:
            return "HIGH"

        if set_size >= 3 or avg_epi > 0.003:
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

    def evaluate(
        self,
        ood_result: dict,
        routing_result: dict,
        inference_result: dict,
        quality_result: dict,
    ) -> dict:
        risk_category = self.classify_risk(
            ood_result=ood_result,
            routing_result=routing_result,
            inference_result=inference_result,
        )

        if ood_result.get("is_hard_ood", False):
            action = "STOP"
            reason = "Hard OOD detected"
            return {
                "action": action,
                "reason": reason,
                "risk_category": "HIGH",
                "warnings": self.build_warnings(action, "HIGH"),
            }

        if ood_result.get("tier") == "NEAR_OOD":
            action = "STOP"
            reason = "Unrelated or out-of-distribution image detected"
            return {
                "action": action,
                "reason": reason,
                "risk_category": "HIGH",
                "warnings": self.build_warnings(action, "HIGH"),
            }

        if quality_result.get("blocking", False):
            action = "REQUEST_EVIDENCE"
            reason = quality_result.get("reason", "Quality check failed")
            return {
                "action": action,
                "reason": reason,
                "risk_category": risk_category,
                "warnings": self.build_warnings(action, risk_category),
            }

        if routing_result.get("requires_confirmation", False):
            action = "ESCALATE"
            reason = "Routing confidence too low"
            return {
                "action": action,
                "reason": reason,
                "risk_category": risk_category,
                "warnings": self.build_warnings(action, risk_category),
            }

        top_prob = inference_result.get("top_probability", 0.0)
        epi_dict = inference_result.get("epistemic_uncertainty", {})
        avg_epi = 0.0
        if epi_dict:
            avg_epi = sum(epi_dict.values()) / len(epi_dict)

        if top_prob < 0.75 and avg_epi > 0.02:
            action = "REFUSE"
            reason = "Prediction not reliable enough to answer safely"
            return {
                "action": action,
                "reason": reason,
                "risk_category": risk_category,
                "warnings": self.build_warnings(action, risk_category),
            }

        action = "ANSWER"
        reason = "All gates passed"
        return {
            "action": action,
            "reason": reason,
            "risk_category": risk_category,
            "warnings": self.build_warnings(action, risk_category),
        }