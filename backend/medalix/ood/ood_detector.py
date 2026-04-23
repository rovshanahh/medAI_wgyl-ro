import json
import math
from pathlib import Path


class OODDetector:
    def __init__(self, stats_path: str = "reference_data/feature_stats.json"):
        stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        self.mean_vector = stats["mean_vector"]
        self.std_vector = stats["std_vector"]
        self.hard_ood_threshold = float(stats.get("hard_ood_threshold", 120.0))
        self.near_ood_threshold = float(stats.get("near_ood_threshold", 80.0))

    def evaluate(self, tensor, inference_result: dict | None = None) -> dict:
        inference_result = inference_result or {}
        features = inference_result.get("features", [])

        if not features:
            return {
                "score": 9999.0,
                "tier": "HARD_OOD",
                "is_hard_ood": True,
                "reason": "Feature vector missing for OOD evaluation",
            }

        if len(features) != len(self.mean_vector) or len(features) != len(self.std_vector):
            return {
                "score": 9999.0,
                "tier": "HARD_OOD",
                "is_hard_ood": True,
                "reason": "Feature vector shape does not match reference statistics",
            }

        normalized_squared_distance = sum(
            ((float(f) - float(m)) / max(float(s), 1e-6)) ** 2
            for f, m, s in zip(features, self.mean_vector, self.std_vector)
        )
        dist = math.sqrt(normalized_squared_distance)

        if dist > self.hard_ood_threshold:
            return {
                "score": float(dist),
                "tier": "HARD_OOD",
                "is_hard_ood": True,
                "reason": "Input is far from the reference feature distribution",
            }

        if dist > self.near_ood_threshold:
            return {
                "score": float(dist),
                "tier": "NEAR_OOD",
                "is_hard_ood": False,
                "reason": "Input is near the boundary of the reference feature distribution",
            }

        return {
            "score": float(dist),
            "tier": "IN_DISTRIBUTION",
            "is_hard_ood": False,
            "reason": "Input is within the reference feature distribution",
        }