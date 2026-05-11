import torch
import torch.nn.functional as F


class QualityAssessor:
    def __init__(
        self,
        blur_threshold: float = 0.0015,
        severe_blur_threshold: float = 0.0004,
        noise_threshold: float = 0.05,
        severe_noise_threshold: float = 0.12,
        min_contrast_std: float = 0.08,
        severe_min_contrast_std: float = 0.04,
        min_dynamic_range: float = 0.20,
        min_foreground_fraction: float = 0.02,
        max_artifact_score: float = 0.25,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.severe_blur_threshold = severe_blur_threshold
        self.noise_threshold = noise_threshold
        self.severe_noise_threshold = severe_noise_threshold
        self.min_contrast_std = min_contrast_std
        self.severe_min_contrast_std = severe_min_contrast_std
        self.min_dynamic_range = min_dynamic_range
        self.min_foreground_fraction = min_foreground_fraction
        self.max_artifact_score = max_artifact_score

    def assess(self, tensor) -> dict:
        if tensor is None:
            return self._result(
                status="invalid",
                blocking=True,
                reason="No tensor available for quality assessment",
                warnings=["Missing preprocessed tensor"],
                requires_reupload=True,
                metrics={},
            )

        if not isinstance(tensor, torch.Tensor):
            return self._result(
                status="invalid",
                blocking=True,
                reason="Quality assessor received a non-tensor input",
                warnings=["Invalid preprocessing output"],
                requires_reupload=True,
                metrics={},
            )

        if tensor.ndim != 4:
            return self._result(
                status="invalid",
                blocking=True,
                reason=f"Expected tensor with shape [B, C, H, W], got {tuple(tensor.shape)}",
                warnings=["Malformed input tensor"],
                requires_reupload=True,
                metrics={},
            )

        working = tensor.detach().float().cpu()

        if working.numel() == 0:
            return self._result(
                status="invalid",
                blocking=True,
                reason="Tensor is empty",
                warnings=["Empty image tensor"],
                requires_reupload=True,
                metrics={},
            )

        if torch.isnan(working).any() or torch.isinf(working).any():
            return self._result(
                status="invalid",
                blocking=True,
                reason="Tensor contains NaN or infinite values",
                warnings=["Corrupted tensor values"],
                requires_reupload=True,
                metrics={},
            )

        normalized_gray = self._normalized_gray(working)

        noise_score = self._estimate_noise(normalized_gray)
        blur_score = self._estimate_blur(normalized_gray)
        dynamic_range = self._estimate_dynamic_range(normalized_gray)
        contrast_std = self._estimate_contrast_std(normalized_gray)
        foreground_fraction = self._estimate_foreground_fraction(normalized_gray)
        artifact_score = self._estimate_artifact_score(normalized_gray)
        orientation_ok = self._check_orientation(working)

        warnings = []
        blocking = False
        requires_reupload = False
        reasons = []

        if blur_score < self.blur_threshold:
            warnings.append("Image appears blurry")
            reasons.append("Blur score below threshold")

        if noise_score > self.noise_threshold:
            warnings.append("Image appears noisy")
            reasons.append("Noise score above threshold")

        if contrast_std < self.min_contrast_std or dynamic_range < self.min_dynamic_range:
            warnings.append("Low contrast or narrow intensity range detected")
            reasons.append("Contrast score below threshold")

        if foreground_fraction < self.min_foreground_fraction:
            warnings.append("Image may be incomplete or mostly empty")
            reasons.append("Foreground fraction below threshold")

        if artifact_score > self.max_artifact_score:
            warnings.append("Potential artifact severity detected")
            reasons.append("Artifact score above threshold")

        if not orientation_ok:
            warnings.append("Image orientation may be inconsistent")
            reasons.append("Orientation check failed")

        severe_quality_failure = (
            blur_score < self.severe_blur_threshold
            or noise_score > self.severe_noise_threshold
            or contrast_std < self.severe_min_contrast_std
            or dynamic_range < self.min_dynamic_range * 0.5
            or foreground_fraction < self.min_foreground_fraction
        )

        if severe_quality_failure:
            blocking = True
            requires_reupload = True

        status = "passed"
        reason = "Quality check passed"

        if warnings and not blocking:
            status = "warning"
            reason = "; ".join(reasons)

        if blocking:
            status = "failed"
            reason = "; ".join(reasons) if reasons else "Quality check failed"

        return self._result(
            status=status,
            blocking=blocking,
            reason=reason,
            warnings=warnings,
            requires_reupload=requires_reupload,
            metrics={
                "noise_score": noise_score,
                "blur_score": blur_score,
                "dynamic_range": dynamic_range,
                "contrast_std": contrast_std,
                "foreground_fraction": foreground_fraction,
                "artifact_score": artifact_score,
                "orientation_ok": orientation_ok,
            },
        )

    def _result(
        self,
        status: str,
        blocking: bool,
        reason: str,
        warnings: list[str],
        requires_reupload: bool,
        metrics: dict,
    ) -> dict:
        return {
            "status": status,
            "blocking": blocking,
            "reason": reason,
            "warnings": warnings,
            "requires_reupload": requires_reupload,
            "metrics": metrics,
        }

    def _normalized_gray(self, tensor: torch.Tensor) -> torch.Tensor:
        gray = tensor.mean(dim=1, keepdim=True)

        min_val = gray.amin(dim=(2, 3), keepdim=True)
        max_val = gray.amax(dim=(2, 3), keepdim=True)

        normalized = (gray - min_val) / (max_val - min_val + 1e-6)
        return normalized.clamp(0.0, 1.0)

    def _estimate_noise(self, gray: torch.Tensor) -> float:
        smooth = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        residual = torch.abs(gray - smooth)
        return float(residual.mean().item())

    def _estimate_blur(self, gray: torch.Tensor) -> float:
        lap = (
            -4 * gray[:, :, 1:-1, 1:-1]
            + gray[:, :, :-2, 1:-1]
            + gray[:, :, 2:, 1:-1]
            + gray[:, :, 1:-1, :-2]
            + gray[:, :, 1:-1, 2:]
        )
        return float(lap.var().item())

    def _estimate_dynamic_range(self, gray: torch.Tensor) -> float:
        min_val = float(gray.min().item())
        max_val = float(gray.max().item())
        return max_val - min_val

    def _estimate_contrast_std(self, gray: torch.Tensor) -> float:
        return float(gray.std(unbiased=False).item())

    def _estimate_foreground_fraction(self, gray: torch.Tensor) -> float:
        threshold = gray.mean()
        foreground = (gray > threshold).float().mean()
        return float(foreground.item())

    def _estimate_artifact_score(self, gray: torch.Tensor) -> float:
        row_diff = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :]).mean()
        col_diff = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]).mean()
        return float(torch.abs(row_diff - col_diff).item())

    def _check_orientation(self, tensor: torch.Tensor) -> bool:
        _, _, height, width = tensor.shape
        return height > 0 and width > 0 and height <= width * 4 and width <= height * 4