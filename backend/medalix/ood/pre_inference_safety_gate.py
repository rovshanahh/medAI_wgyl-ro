class PreInferenceSafetyGate:
    """
    Pre-inference route/OOD safety gate.

    This gate should only stop clearly unsafe inputs before inference:
    - unsupported route
    - strongly non-medical input
    - extremely low medical confidence

    Borderline medical/routing cases continue as PASS_WITH_REVIEW.
    """

    MIN_MEDICAL_CONFIDENCE_STOP = 0.25
    MAX_NON_MEDICAL_CONFIDENCE_STOP = 0.90

    MIN_MEDICAL_CONFIDENCE_REVIEW = 0.45
    MIN_ROUTE_CONFIDENCE_REVIEW = 0.55
    MIN_ROUTE_MARGIN_REVIEW = 0.05

    def evaluate(
        self,
        medical_gate_result: dict | None,
        route_result: dict | None,
        manual_override: bool = False,
    ) -> dict:
        medical_gate_result = medical_gate_result or {}
        route_result = route_result or {}

        stop_reasons = []
        review_warnings = []

        medical_confidence = float(
            medical_gate_result.get("medical_probability")
            or medical_gate_result.get("confidence")
            or 0.0
        )

        non_medical_confidence = float(
            medical_gate_result.get("non_medical_probability") or 0.0
        )

        route_confidence = float(route_result.get("confidence") or 0.0)
        route_margin = float(route_result.get("margin") or 0.0)
        route_supported = bool(route_result.get("supported", False))
        route_label = route_result.get("route_label") or "unknown"
        medical_gate_rescued = bool(route_result.get("medical_gate_rescued", False))

        # If a human manually confirms a known route, do not block just because
        # automatic route metadata is imperfect. Continue with review instead.
        known_supported_routes = {
            "chest_xray",
            "bone_xray",
            "brain_mri",
            "abdomen_ct",
            "breast_mammography",
            "retina_fundus",
            "skin_dermoscopy",
        }

        if manual_override and route_label in known_supported_routes:
            route_supported = True
            route_confidence = max(route_confidence, 1.0)
            route_margin = max(route_margin, 1.0)

        # Hard stops only
        if not route_supported or route_label in {"unknown", "unsupported"}:
            stop_reasons.append("Route is not supported for inference.")

        if (
            route_supported
            and route_label in known_supported_routes
            and (manual_override or medical_gate_rescued)
        ):
            # Human-confirmed OR route-detector-rescued supported medical route.
            # Do not hard-stop only because the broad medical gate is weak.
            # This is important for skin/dermoscopy and visually unusual X-rays.
            if (
                route_label == "skin_dermoscopy"
                and medical_gate_rescued
                and route_confidence >= 0.95
            ):
                review_warnings.append(
                    "Skin/dermoscopy route was strongly detected despite weak broad medical-gate confidence; continuing with review."
                )
            elif non_medical_confidence >= 0.97:
                stop_reasons.append(
                    f"Non-medical confidence {non_medical_confidence:.3f} is extremely high even after route confirmation."
                )
            elif medical_confidence < self.MIN_MEDICAL_CONFIDENCE_REVIEW:
                review_warnings.append(
                    f"Medical-image gate confidence {medical_confidence:.3f} is low, but supported route confirmation allows reviewed inference."
                )
        else:
            if medical_confidence < self.MIN_MEDICAL_CONFIDENCE_STOP:
                stop_reasons.append(
                    f"Medical-image confidence {medical_confidence:.3f} is too low for safe inference."
                )

            if non_medical_confidence >= self.MAX_NON_MEDICAL_CONFIDENCE_STOP:
                stop_reasons.append(
                    f"Non-medical confidence {non_medical_confidence:.3f} is too high for safe inference."
                )

        # Review warnings only — do not block
        if medical_confidence < self.MIN_MEDICAL_CONFIDENCE_REVIEW:
            review_warnings.append(
                f"Medical-image confidence {medical_confidence:.3f} is borderline."
            )

        if route_confidence < self.MIN_ROUTE_CONFIDENCE_REVIEW:
            review_warnings.append(
                f"Route confidence {route_confidence:.3f} is borderline."
            )

        if route_margin < self.MIN_ROUTE_MARGIN_REVIEW:
            review_warnings.append(
                f"Route margin {route_margin:.3f} is low."
            )

        if manual_override:
            review_warnings.append(
                "Manual route confirmation was used."
            )

        passed = len(stop_reasons) == 0

        if not passed:
            decision = "STOP"
            risk = "HIGH"
            reason = "Pre-inference safety gate blocked analysis before model inference."
        elif review_warnings:
            decision = "PASS_WITH_REVIEW"
            risk = "MODERATE"
            reason = "Pre-inference safety checks passed with review warnings."
        else:
            decision = "PASS"
            risk = "LOW"
            reason = "Pre-inference safety checks passed."

        return {
            "passed": passed,
            "decision": decision,
            "risk": risk,
            "route_label": route_label,
            "manual_override": manual_override,
            "reasons": stop_reasons,
            "warnings": review_warnings,
            "checks": {
                "medical_confidence": medical_confidence,
                "non_medical_confidence": non_medical_confidence,
                "route_confidence": route_confidence,
                "route_margin": route_margin,
                "route_supported": route_supported,
                "medical_gate_rescued": medical_gate_rescued,
                "min_medical_confidence_stop": self.MIN_MEDICAL_CONFIDENCE_STOP,
                "max_non_medical_confidence_stop": self.MAX_NON_MEDICAL_CONFIDENCE_STOP,
                "min_medical_confidence_review": self.MIN_MEDICAL_CONFIDENCE_REVIEW,
                "min_route_confidence_review": self.MIN_ROUTE_CONFIDENCE_REVIEW,
                "min_route_margin_review": self.MIN_ROUTE_MARGIN_REVIEW,
            },
            "reason": reason,
        }
