import json
import math
from pathlib import Path

import torch


class OODDetector:
    def __init__(self, stats_path: str = "reference_data/feature_stats.json"):
        self.stats_path = Path(stats_path)

        if self.stats_path.exists():
            stats = json.loads(self.stats_path.read_text(encoding="utf-8"))
        else:
            stats = {}

        self.mean_vector = stats.get("mean_vector", [])
        self.std_vector = stats.get("std_vector", [])

        # Important:
        # If this field is missing, we do NOT enforce feature-based OOD.
        # Your old feature_stats.json is likely chest-only, so applying it to
        # brain/bone is wrong.
        self.stats_route_label = stats.get("route_label")

        self.hard_ood_threshold = float(stats.get("hard_ood_threshold", 120.0))
        self.near_ood_threshold = float(stats.get("near_ood_threshold", 80.0))

    def _feature_distance_ood(self, features: list[float], route_label: str | None) -> dict:
        if not self.stats_route_label:
            return {
                "score": None,
                "tier": "UNKNOWN",
                "is_hard_ood": False,
                "reason": (
                    "Feature statistics are not route-specific. "
                    "Feature-based OOD was not enforced."
                ),
                "method": "feature_distance_skipped",
            }

        if route_label != self.stats_route_label:
            return {
                "score": None,
                "tier": "UNKNOWN",
                "is_hard_ood": False,
                "reason": (
                    f"Feature statistics belong to route '{self.stats_route_label}', "
                    f"but current route is '{route_label}'. "
                    "Feature-based OOD was not enforced."
                ),
                "method": "feature_distance_skipped",
            }

        if not self.mean_vector or not self.std_vector:
            return {
                "score": None,
                "tier": "UNKNOWN",
                "is_hard_ood": False,
                "reason": "Feature statistics are missing. OOD check was not enforced.",
                "method": "feature_distance",
            }

        if len(features) != len(self.mean_vector) or len(features) != len(self.std_vector):
            return {
                "score": None,
                "tier": "UNKNOWN",
                "is_hard_ood": False,
                "reason": (
                    "Feature vector shape does not match reference statistics. "
                    "OOD check was not enforced."
                ),
                "method": "feature_distance",
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
                "reason": "Input is far from the route-specific reference feature distribution.",
                "method": "feature_distance",
            }

        if dist > self.near_ood_threshold:
            return {
                "score": float(dist),
                "tier": "NEAR_OOD",
                "is_hard_ood": False,
                "reason": "Input is near the boundary of the route-specific reference feature distribution.",
                "method": "feature_distance",
            }

        return {
            "score": float(dist),
            "tier": "IN_DISTRIBUTION",
            "is_hard_ood": False,
            "reason": "Input is within the route-specific reference feature distribution.",
            "method": "feature_distance",
        }

    def _basic_tensor_check(self, tensor) -> dict:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return {
                "score": None,
                "tier": "UNKNOWN",
                "is_hard_ood": False,
                "reason": "Tensor is missing. OOD check was not enforced.",
                "method": "basic_tensor_check",
                "metrics": {},
            }

        working = tensor.detach().float().cpu()

        has_nan = bool(torch.isnan(working).any())
        has_inf = bool(torch.isinf(working).any())
        tensor_std = float(working.std())

        if has_nan or has_inf:
            return {
                "score": None,
                "tier": "HARD_OOD",
                "is_hard_ood": True,
                "reason": "Input tensor contains NaN or infinite values.",
                "method": "basic_tensor_check",
                "metrics": {
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                    "std": tensor_std,
                },
            }

        if tensor_std < 1e-6:
            return {
                "score": tensor_std,
                "tier": "HARD_OOD",
                "is_hard_ood": True,
                "reason": "Input tensor is nearly constant and contains no useful visual information.",
                "method": "basic_tensor_check",
                "metrics": {
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                    "std": tensor_std,
                },
            }

        return {
            "score": tensor_std,
            "tier": "IN_DISTRIBUTION",
            "is_hard_ood": False,
            "reason": (
                "Basic tensor check passed. Feature-based OOD was skipped unless "
                "route-specific feature statistics are available."
            ),
            "method": "basic_tensor_check",
            "metrics": {
                "has_nan": has_nan,
                "has_inf": has_inf,
                "std": tensor_std,
            },
        }

    def evaluate(
        self,
        tensor,
        inference_result: dict | None = None,
        route_label: str | None = None,
    ) -> dict:
        inference_result = inference_result or {}
        features = inference_result.get("features", [])

        if features:
            feature_result = self._feature_distance_ood(
                features=features,
                route_label=route_label,
            )

            if feature_result["tier"] != "UNKNOWN":
                return feature_result

        return self._basic_tensor_check(tensor)