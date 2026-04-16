import json
from pathlib import Path


class ConformalRouter:
    def __init__(self, config_path: str = "reference_data/conformal_scores.json"):
        config = json.loads(Path(config_path).read_text())
        self.alpha = config["alpha"]
        self.threshold = config["threshold"]

    def decide(self, inference_result: dict) -> dict:
        probs = inference_result.get("probabilities", {})
        if not probs:
            return {
                "accepted_findings_set": [],
                "set_size": 0,
                "requires_confirmation": True,
                "alpha": self.alpha,
                "threshold": self.threshold,
                "top_probability": 0.0,
                "nonconformity": 1.0,
            }

        sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_label, top_prob = sorted_items[0]
        nonconformity = 1.0 - float(top_prob)

        positive_findings = inference_result.get("positive_findings", [])

        if nonconformity <= self.threshold:
            accepted_findings_set = positive_findings if positive_findings else [top_label]
            requires_confirmation = False
        else:
            accepted_findings_set = positive_findings[:5] if positive_findings else [top_label]
            requires_confirmation = True

        return {
            "accepted_findings_set": accepted_findings_set,
            "set_size": len(accepted_findings_set),
            "requires_confirmation": requires_confirmation,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "top_probability": top_prob,
            "nonconformity": nonconformity,
        }