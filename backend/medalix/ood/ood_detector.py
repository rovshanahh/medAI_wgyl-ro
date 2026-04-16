import json
import math
from pathlib import Path


class OODDetector:
    def __init__(self, stats_path: str = "reference_data/feature_stats.json"):
        stats = json.loads(Path(stats_path).read_text())
        self.mean_vector = stats["mean_vector"]
        self.std_vector = stats["std_vector"]

    def evaluate(self, tensor, inference_result: dict) -> dict:
        features = inference_result.get("features", [])
        if not features:
            return {
                "score": 9999.0,
                "tier": "HARD_OOD",
                "is_hard_ood": True,
            }

        dist = math.sqrt(
            sum(
                ((f - m) / max(s, 1e-6)) ** 2
                for f, m, s in zip(features, self.mean_vector, self.std_vector)
            )
        )

        if dist > 120:
            return {"score": dist, "tier": "HARD_OOD", "is_hard_ood": True}
        if dist > 80:
            return {"score": dist, "tier": "NEAR_OOD", "is_hard_ood": False}
        return {"score": dist, "tier": "IN_DISTRIBUTION", "is_hard_ood": False}
    