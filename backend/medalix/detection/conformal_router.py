import json
from pathlib import Path


class ConformalRouter:
    def __init__(self, config_path: str = "reference_data/conformal_scores.json"):
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.alpha = config["alpha"]
        self.threshold = config["threshold"]

    def decide(self, detection_result: dict) -> dict:
        detection_result = detection_result or {}

        region = detection_result.get("region")
        modality = detection_result.get("modality")
        confidence = float(detection_result.get("confidence", 0.0) or 0.0)

        if not region or not modality:
            return {
                "accepted_findings_set": [],
                "set_size": 0,
                "requires_confirmation": True,
                "alpha": self.alpha,
                "threshold": self.threshold,
                "top_probability": 0.0,
                "nonconformity": 1.0,
                "routing_candidates": [],
            }

        candidate = f"{region}:{modality}"
        nonconformity = 1.0 - confidence

        if nonconformity <= self.threshold:
            accepted_findings_set = [candidate]
            requires_confirmation = False
        else:
            accepted_findings_set = [candidate]
            requires_confirmation = True

        return {
            "accepted_findings_set": accepted_findings_set,
            "set_size": len(accepted_findings_set),
            "requires_confirmation": requires_confirmation,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "top_probability": confidence,
            "nonconformity": nonconformity,
            "routing_candidates": accepted_findings_set,
        }