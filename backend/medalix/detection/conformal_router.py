import json
from pathlib import Path


class ConformalRouter:
    def __init__(self, config_path: str = "reference_data/conformal_scores.json"):
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.alpha = float(config["alpha"])
        self.threshold = float(config["threshold"])

    def _make_candidate(self, region: str, modality: str) -> str:
        return f"{region.strip().lower()}:{modality.strip().lower()}"

    def decide(self, detection_result: dict) -> dict:
        detection_result = detection_result or {}

        region = detection_result.get("region")
        modality = detection_result.get("modality")
        confidence = float(detection_result.get("confidence", 0.0) or 0.0)
        supported = bool(detection_result.get("supported", False))

        if not region or not modality or not supported:
            return {
                "accepted_findings_set": [],
                "set_size": 0,
                "requires_confirmation": True,
                "alpha": self.alpha,
                "threshold": self.threshold,
                "top_probability": confidence,
                "nonconformity": 1.0,
                "routing_candidates": [],
                "selected_model": None,
            }

        candidate = self._make_candidate(region, modality)
        nonconformity = 1.0 - confidence

        if confidence >= 0.80 and nonconformity <= self.threshold:
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
            "selected_model": None,
        }