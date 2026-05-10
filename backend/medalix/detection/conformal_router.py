import json
from pathlib import Path


class ConformalRouter:
    def __init__(
        self,
        calibration_path: str = "reference_data/conformal/calibration_scores.json",
    ):
        self.calibration_path = Path(calibration_path)
        self.config = self._load_config()

        self.alpha = float(self.config.get("alpha", 0.1))
        self.threshold = float(self.config.get("threshold", 0.25))
        self.method = self.config.get("method", "lightweight_margin_conformal")
        self.route_labels = set(self.config.get("route_labels", []))

    def _load_config(self) -> dict:
        if not self.calibration_path.exists():
            return {
                "alpha": 0.1,
                "threshold": 0.25,
                "method": "lightweight_margin_conformal",
                "route_labels": [],
            }

        with self.calibration_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def route(self, route_result: dict) -> dict:
        probabilities = route_result.get("probabilities", {}) or {}
        selected_route = route_result.get("route_label")
        confidence = float(route_result.get("confidence") or 0.0)

        if not route_result.get("supported", False):
            return {
                "accepted_findings_set": [],
                "set_size": 0,
                "requires_confirmation": True,
                "alpha": self.alpha,
                "threshold": self.threshold,
                "top_probability": confidence,
                "nonconformity": None,
                "routing_candidates": [],
                "routing_candidate_details": [],
                "selected_model": None,
                "method": self.method,
                "reason": "Input was not supported by route detector.",
            }

        candidates = []

        for label, probability in probabilities.items():
            if label == "unknown":
                continue

            if self.route_labels and label not in self.route_labels:
                continue

            probability = float(probability)
            nonconformity = 1.0 - probability

            if nonconformity <= self.threshold:
                candidates.append(
                    {
                        "route": label,
                        "probability": probability,
                        "nonconformity": nonconformity,
                    }
                )

        candidates.sort(key=lambda item: item["probability"], reverse=True)

        if not candidates and selected_route:
            candidates = [
                {
                    "route": selected_route,
                    "probability": confidence,
                    "nonconformity": 1.0 - confidence,
                }
            ]

        accepted_routes = [item["route"] for item in candidates]
        requires_confirmation = len(accepted_routes) != 1

        return {
            "accepted_findings_set": accepted_routes,
            "set_size": len(accepted_routes),
            "requires_confirmation": requires_confirmation,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "top_probability": confidence,
            "nonconformity": 1.0 - confidence,
            "routing_candidates": accepted_routes,
            "routing_candidate_details": candidates,
            "selected_model": None,
            "method": self.method,
            "reason": (
                "Conformal routing produced an ambiguous or empty route set."
                if requires_confirmation
                else "Conformal routing produced a single accepted route."
            ),
        }