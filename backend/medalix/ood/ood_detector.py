from pathlib import Path
import json

import torch
import torch.nn.functional as F


class OODDetector:
    """
    SBDDM-inspired MVP OOD detector.

    This is not a full trained diffusion model. It is a lightweight score-based
    diffusion-style stability detector using 5 noise/denoising steps, combined
    with optional route-specific feature reference statistics.

    Output tiers:
    - IN_DISTRIBUTION
    - NEAR_OOD
    - HARD_OOD
    """

    def __init__(
        self,
        stats_path: str = "reference_data/ood/feature_stats.json",
        diffusion_steps: int = 5,
    ):
        self.stats_path = Path(stats_path)
        self.diffusion_steps = diffusion_steps
        self.feature_stats = self._load_feature_stats()

    def _load_feature_stats(self) -> dict:
        if not self.stats_path.exists():
            return {}

        try:
            return json.loads(self.stats_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _basic_tensor_check(self, tensor: torch.Tensor) -> dict:
        if not isinstance(tensor, torch.Tensor):
            return {
                "failed": True,
                "reason": "Input is not a tensor.",
                "metrics": {},
            }

        has_nan = bool(torch.isnan(tensor).any().item())
        has_inf = bool(torch.isinf(tensor).any().item())

        tensor_std = float(tensor.detach().float().std().cpu().item())
        tensor_mean = float(tensor.detach().float().mean().cpu().item())

        failed = has_nan or has_inf or tensor_std < 1e-6

        reason = "Basic tensor check passed."
        if has_nan:
            reason = "Tensor contains NaN values."
        elif has_inf:
            reason = "Tensor contains infinite values."
        elif tensor_std < 1e-6:
            reason = "Tensor has near-zero variance."

        return {
            "failed": failed,
            "reason": reason,
            "metrics": {
                "has_nan": has_nan,
                "has_inf": has_inf,
                "mean": tensor_mean,
                "std": tensor_std,
            },
        }

    def _diffusion_stability_score(self, tensor: torch.Tensor) -> dict:
        """
        Lightweight 5-step diffusion-style perturbation stability scoring.

        The idea:
        - Add increasing Gaussian noise.
        - Apply simple denoising with average pooling.
        - Measure reconstruction instability.
        - Higher score means more unstable / more suspicious.
        """

        x = tensor.detach().float().cpu()

        if x.ndim != 4:
            return {
                "score": 1.0,
                "step_errors": [],
                "reason": "Invalid tensor shape for diffusion-style OOD scoring.",
            }

        step_errors = []

        for step in range(1, self.diffusion_steps + 1):
            noise_scale = 0.03 * step

            noise = torch.randn_like(x) * noise_scale
            noisy = x + noise

            denoised = F.avg_pool2d(
                noisy,
                kernel_size=3,
                stride=1,
                padding=1,
            )

            error = torch.mean(torch.abs(x - denoised)).item()
            step_errors.append(float(error))

        score = float(sum(step_errors) / len(step_errors))

        return {
            "score": score,
            "step_errors": step_errors,
            "reason": "Five-step diffusion-style stability score computed.",
        }

    def _feature_distance_score(
        self,
        inference_result: dict,
        route_label: str,
    ) -> dict:
        route_stats = self.feature_stats.get(route_label)

        if not route_stats:
            return {
                "available": False,
                "score": None,
                "reason": "Feature-based OOD was skipped because no route-specific feature statistics are available.",
                "metrics": {},
            }

        features = inference_result.get("features")

        if not features:
            return {
                "available": False,
                "score": None,
                "reason": "Feature-based OOD was skipped because inference features are unavailable.",
                "metrics": {},
            }

        feature_tensor = torch.tensor(features, dtype=torch.float32)
        mean_tensor = torch.tensor(route_stats["mean"], dtype=torch.float32)
        std_tensor = torch.tensor(route_stats["std"], dtype=torch.float32)

        std_tensor = torch.clamp(std_tensor, min=1e-6)

        if feature_tensor.shape != mean_tensor.shape:
            return {
                "available": False,
                "score": None,
                "reason": "Feature-based OOD was skipped because feature shape does not match stored statistics.",
                "metrics": {
                    "feature_shape": list(feature_tensor.shape),
                    "mean_shape": list(mean_tensor.shape),
                },
            }

        z_scores = torch.abs((feature_tensor - mean_tensor) / std_tensor)
        mean_z = float(z_scores.mean().item())
        max_z = float(z_scores.max().item())

        return {
            "available": True,
            "score": mean_z,
            "reason": "Route-specific feature distance score computed.",
            "metrics": {
                "mean_z": mean_z,
                "max_z": max_z,
                "threshold_near": route_stats.get("threshold_near"),
                "threshold_hard": route_stats.get("threshold_hard"),
            },
        }

    def _decide_tier(
        self,
        basic_check: dict,
        diffusion_result: dict,
        feature_result: dict,
        route_label: str,
    ) -> dict:
        if basic_check["failed"]:
            return {
                "tier": "HARD_OOD",
                "is_hard_ood": True,
                "reason": basic_check["reason"],
            }

        diffusion_score = diffusion_result["score"]

        diffusion_near_threshold = 0.65
        diffusion_hard_threshold = 0.90

        feature_available = feature_result.get("available", False)
        feature_score = feature_result.get("score")

        if feature_available:
            route_stats = self.feature_stats.get(route_label, {})
            feature_near_threshold = float(route_stats.get("threshold_near", 2.5))
            feature_hard_threshold = float(route_stats.get("threshold_hard", 4.0))

            if feature_score >= feature_hard_threshold or diffusion_score >= diffusion_hard_threshold:
                return {
                    "tier": "HARD_OOD",
                    "is_hard_ood": True,
                    "reason": "Input is far from route-specific feature statistics or unstable under diffusion-style scoring.",
                }

            if feature_score >= feature_near_threshold or diffusion_score >= diffusion_near_threshold:
                return {
                    "tier": "NEAR_OOD",
                    "is_hard_ood": False,
                    "reason": "Input is close to the OOD boundary and should be reviewed carefully.",
                }

            return {
                "tier": "IN_DISTRIBUTION",
                "is_hard_ood": False,
                "reason": "Input passed tensor, diffusion-style, and feature-distance OOD checks.",
            }

        if diffusion_score >= diffusion_hard_threshold:
            return {
                "tier": "HARD_OOD",
                "is_hard_ood": True,
                "reason": "Input is unstable under five-step diffusion-style OOD scoring.",
            }

        if diffusion_score >= diffusion_near_threshold:
            return {
                "tier": "NEAR_OOD",
                "is_hard_ood": False,
                "reason": "Input is near the OOD boundary under diffusion-style OOD scoring.",
            }

        return {
            "tier": "IN_DISTRIBUTION",
            "is_hard_ood": False,
            "reason": (
                "Basic tensor and five-step diffusion-style OOD checks passed. "
                "Feature-based OOD was skipped unless route-specific feature statistics are available."
            ),
        }

    def evaluate(
        self,
        tensor: torch.Tensor,
        inference_result: dict,
        route_label: str,
    ) -> dict:
        basic_check = self._basic_tensor_check(tensor)
        diffusion_result = self._diffusion_stability_score(tensor)
        feature_result = self._feature_distance_score(
            inference_result=inference_result or {},
            route_label=route_label,
        )

        decision = self._decide_tier(
            basic_check=basic_check,
            diffusion_result=diffusion_result,
            feature_result=feature_result,
            route_label=route_label,
        )

        score = diffusion_result["score"]

        if feature_result.get("available") and feature_result.get("score") is not None:
            score = max(float(score), float(feature_result["score"]) / 5.0)

        return {
            "tier": decision["tier"],
            "score": float(score),
            "is_hard_ood": bool(decision["is_hard_ood"]),
            "reason": decision["reason"],
            "method": "sbdm_inspired_5_step_stability_plus_feature_distance",
            "metrics": {
                "diffusion_steps": self.diffusion_steps,
                "diffusion_score": diffusion_result["score"],
                "diffusion_step_errors": diffusion_result["step_errors"],
                "basic_tensor_check": basic_check["metrics"],
                "feature_distance": feature_result,
            },
        }