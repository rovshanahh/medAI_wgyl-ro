import json
from pathlib import Path


class ConformalRouter:
    def __init__(self, config_path: str = "reference_data/conformal_scores.json"):
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.alpha = float(config.get("alpha", 0.1))
        self.threshold = float(config.get("threshold", 0.30))

    def _make_candidate(self, region: str, modality: str) -> str:
        return f"{region.strip().lower()}:{modality.strip().lower()}"

    def decide(self, detection_result: dict) -> dict:
        detection_result = detection_result or {}

        region = detection_result.get("region")
        modality = detection_result.get("modality")
        confidence = float(detection_result.get("confidence", 0.0) or 0.0)
        supported = bool(detection_result.get("supported", False))
        requires_detection_confirmation = bool(
            detection_result.get("requires_confirmation", False)
        )

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
                "reason": "Routing blocked because detection is incomplete or unsupported.",
            }

        candidate = self._make_candidate(region, modality)
        nonconformity = 1.0 - confidence

        requires_confirmation = (
            requires_detection_confirmation
            or confidence < 0.80
            or nonconformity > self.threshold
        )

        return {
            "accepted_findings_set": [candidate],
            "set_size": 1,
            "requires_confirmation": requires_confirmation,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "top_probability": confidence,
            "nonconformity": nonconformity,
            "routing_candidates": [candidate],
            "selected_model": None,
            "reason": (
                "Single confident routing candidate selected."
                if not requires_confirmation
                else "Routing candidate selected but confirmation is required."
            ),
        }