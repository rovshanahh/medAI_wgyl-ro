NON_DIAGNOSTIC_DISCLAIMER = (
    "Research-use only. This output is non-diagnostic and must not be used "
    "for clinical decision-making."
)


class ErrorResponseBuilder:
    @staticmethod
    def _base_error(
        analysis_id: str,
        stage: str,
        message: str,
        error_type: str,
    ) -> dict:
        return {
            "analysis_id": analysis_id,
            "status": "failed",
            "error_type": error_type,
            "stage": stage,
            "message": message,
            "input_gate": {
                "accepted_for_analysis": False,
                "confidence": None,
                "message": message,
                "selected_route": "unknown",
                "route_scores": {},
                "top_level_gate": {},
            },
            "detection": {
                "region": None,
                "modality": None,
                "confidence": None,
                "requires_confirmation": True,
                "supported": False,
                "reason": message,
            },
            "routing": {
                "accepted_findings_set": [],
                "selected_model": None,
                "set_size": 0,
                "requires_confirmation": True,
                "alpha": None,
                "threshold": None,
                "top_probability": None,
                "nonconformity": None,
                "routing_candidates": [],
            },
            "quality": {
                "status": "failed",
                "warnings": [message],
                "requires_reupload": True,
                "blocking": True,
                "reason": message,
                "metrics": {},
            },
            "ood": {
                "tier": None,
                "score": None,
                "is_hard_ood": False,
                "reason": None,
                "method": None,
                "metrics": {},
            },
            "explainability": {
                "method": None,
                "heatmap_path": None,
                "warning": "Explainability was not generated because processing failed.",
            },
            "selected_model": {},
            "policy": {
                "action": "STOP",
                "reason": message,
                "risk_category": "HIGH",
                "warnings": ["Processing stopped due to an error"],
            },
            "warnings": [message],
            "disclaimer": NON_DIAGNOSTIC_DISCLAIMER,
        }

    @staticmethod
    def build_validation_error(analysis_id: str, stage: str, message: str) -> dict:
        return ErrorResponseBuilder._base_error(
            analysis_id=analysis_id,
            stage=stage,
            message=message,
            error_type="validation_error",
        )

    @staticmethod
    def build_processing_error(analysis_id: str, stage: str, message: str) -> dict:
        return ErrorResponseBuilder._base_error(
            analysis_id=analysis_id,
            stage=stage,
            message=message,
            error_type="processing_error",
        )