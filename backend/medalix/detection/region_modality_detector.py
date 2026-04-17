class RegionModalityDetector:
    def predict(self, input_gate_result: dict | None = None) -> dict:
        input_gate_result = input_gate_result or {}
        confidence = input_gate_result.get("confidence", 0.95)

        return {
            "region": "chest",
            "modality": "xray",
            "confidence": confidence,
            "requires_confirmation": confidence < 0.80,
            "supported": True,
        }