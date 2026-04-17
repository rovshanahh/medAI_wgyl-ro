import torch


class QualityAssessor:
    def __init__(
        self,
        blur_threshold: float = 0.0015,
        noise_threshold: float = 0.35,
        min_dynamic_range: float = 0.05,
        min_foreground_fraction: float = 0.02,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.noise_threshold = noise_threshold
        self.min_dynamic_range = min_dynamic_range
        self.min_foreground_fraction = min_foreground_fraction

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

        noise_score = self._estimate_noise(working)
        blur_score = self._estimate_blur(working)
        dynamic_range = self._estimate_dynamic_range(working)
        foreground_fraction = self._estimate_foreground_fraction(working)
        artifact_score = self._estimate_artifact_score(working)
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

        if dynamic_range < self.min_dynamic_range:
            warnings.append("Low contrast or narrow intensity range detected")
            reasons.append("Dynamic range below threshold")

        if foreground_fraction < self.min_foreground_fraction:
            warnings.append("Image may be incomplete or mostly empty")
            reasons.append("Foreground fraction below threshold")

        if artifact_score > 0.25:
            warnings.append("Potential artifact severity detected")
            reasons.append("Artifact score above threshold")

        if not orientation_ok:
            warnings.append("Image orientation may be inconsistent")
            reasons.append("Orientation check failed")

        if (
            blur_score < self.blur_threshold
            or dynamic_range < self.min_dynamic_range
            or foreground_fraction < self.min_foreground_fraction
        ):
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

    def _estimate_noise(self, tensor: torch.Tensor) -> float:
        gray = tensor.mean(dim=1, keepdim=True)
        diff_h = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :]).mean()
        diff_w = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]).mean()
        return float((diff_h + diff_w) / 2.0)

    def _estimate_blur(self, tensor: torch.Tensor) -> float:
        gray = tensor.mean(dim=1, keepdim=True)
        lap = (
            -4 * gray[:, :, 1:-1, 1:-1]
            + gray[:, :, :-2, 1:-1]
            + gray[:, :, 2:, 1:-1]
            + gray[:, :, 1:-1, :-2]
            + gray[:, :, 1:-1, 2:]
        )
        return float(lap.var())

    def _estimate_dynamic_range(self, tensor: torch.Tensor) -> float:
        min_val = float(tensor.min())
        max_val = float(tensor.max())
        return max_val - min_val

    def _estimate_foreground_fraction(self, tensor: torch.Tensor) -> float:
        gray = tensor.mean(dim=1)
        threshold = gray.mean()
        foreground = (gray > threshold).float().mean()
        return float(foreground)

    def _estimate_artifact_score(self, tensor: torch.Tensor) -> float:
        gray = tensor.mean(dim=1, keepdim=True)
        row_diff = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :]).mean()
        col_diff = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]).mean()
        return float(torch.abs(row_diff - col_diff))

    def _check_orientation(self, tensor: torch.Tensor) -> bool:
        _, _, height, width = tensor.shape
        return height > 0 and width > 0