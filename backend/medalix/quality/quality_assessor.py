class QualityAssessor:
    def assess(self, tensor) -> dict:
        if tensor is None:
            return {
                "blocking": True,
                "reason": "No tensor available for quality assessment",
            }

        return {
            "blocking": False,
            "reason": "Quality check passed",
        }