class ErrorResponseBuilder:
    @staticmethod
    def build_validation_error(analysis_id: str, stage: str, message: str) -> dict:
        return {
            "analysis_id": analysis_id,
            "status": "failed",
            "error_type": "validation_error",
            "stage": stage,
            "message": message,
        }

    @staticmethod
    def build_processing_error(analysis_id: str, stage: str, message: str) -> dict:
        return {
            "analysis_id": analysis_id,
            "status": "failed",
            "error_type": "processing_error",
            "stage": stage,
            "message": message,
        }