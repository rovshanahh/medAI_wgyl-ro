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
        raw_route = route_result.get("raw_route_label")
        confidence = float(route_result.get("confidence") or 0.0)

        if (
            not route_result.get("supported", False)
            or not selected_route
            or selected_route == "unknown"
        ):
            return {
                "accepted_findings_set": [],
                "set_size": 0,
                "requires_confirmation": True,
                "route_decision": "NO_TRUSTED_ROUTE",
                "alpha": self.alpha,
                "threshold": self.threshold,
                "top_probability": confidence,
                "nonconformity": None,
                "routing_candidates": [],
                "routing_candidate_details": [],
                "selected_model": None,
                "method": self.method,
                "reason": (
                    "Conformal routing could not continue because the route detector "
                    "did not provide a trusted supported route."
                ),
                "audit": {
                    "selected_route": selected_route,
                    "raw_route": raw_route,
                    "route_supported": route_result.get("supported", False),
                    "route_requires_confirmation": route_result.get(
                        "requires_confirmation",
                        True,
                    ),
                    "route_reason": route_result.get("reason"),
                },
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

        accepted_routes = [item["route"] for item in candidates]

        # Conservative fallback:
        # If the conformal set is empty but the detector route was accepted,
        # keep the selected route as a candidate but require confirmation.
        if not accepted_routes:
            candidates = [
                {
                    "route": selected_route,
                    "probability": confidence,
                    "nonconformity": 1.0 - confidence,
                }
            ]
            accepted_routes = [selected_route]
            requires_confirmation = True
            route_decision = "EMPTY_SET_FALLBACK"
            reason = (
                "Conformal routing produced an empty route set. The detector route is "
                "kept for audit, but confirmation is required."
            )

        elif len(accepted_routes) > 1:
            requires_confirmation = True
            route_decision = "AMBIGUOUS_SET"
            reason = (
                "Conformal routing produced multiple possible routes, so confirmation "
                "is required before automatic answering."
            )

        else:
            requires_confirmation = accepted_routes[0] != selected_route
            route_decision = "ACCEPTED" if not requires_confirmation else "MISMATCH"
            reason = (
                "Conformal routing produced a single accepted route."
                if not requires_confirmation
                else "Conformal routing selected a route that does not match the detector route."
            )

        return {
            "accepted_findings_set": accepted_routes,
            "set_size": len(accepted_routes),
            "requires_confirmation": requires_confirmation,
            "route_decision": route_decision,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "top_probability": confidence,
            "nonconformity": 1.0 - confidence,
            "routing_candidates": accepted_routes,
            "routing_candidate_details": candidates,
            "selected_model": None,
            "method": self.method,
            "reason": reason,
            "audit": {
                "selected_route": selected_route,
                "raw_route": raw_route,
                "route_detector_confidence": confidence,
                "route_detector_margin": route_result.get("margin"),
                "route_detector_decision": route_result.get("route_decision"),
                "route_detector_reasons": route_result.get("decision_reasons", []),
            },
        }
