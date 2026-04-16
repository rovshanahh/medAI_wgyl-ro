class ConformalRouter:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def decide(self, inference_result: dict) -> dict:
        probs = inference_result.get("probabilities", {})
        if not probs:
            return {
                "accepted_findings_set": [],
                "set_size": 0,
                "requires_confirmation": True,
                "alpha": self.alpha,
                "margin": 0.0,
                "top_probability": 0.0,
            }

        sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        top_label, top_prob = sorted_items[0]
        second_prob = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
        margin = top_prob - second_prob

        positive_findings = inference_result.get("positive_findings", [])

        if top_prob >= 0.85 and len(positive_findings) <= 5:
            accepted_findings_set = positive_findings if positive_findings else [top_label]
            requires_confirmation = False
        else:
            accepted_findings_set = positive_findings[:5] if positive_findings else [label for label, _ in sorted_items[:2]]
            requires_confirmation = True

        return {
            "accepted_findings_set": accepted_findings_set,
            "set_size": len(accepted_findings_set),
            "requires_confirmation": requires_confirmation,
            "alpha": self.alpha,
            "margin": margin,
            "top_probability": top_prob,
        }