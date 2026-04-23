import torch


class RegionModalityDetector:
    SUPPORTED_REGIONS = {
        "brain",
        "chest",
        "abdomen",
        "breast",
        "skin",
        "retina",
        "bone",
    }

    SUPPORTED_MODALITIES = {
        "mri",
        "ct",
        "xray",
        "mammography",
        "dermoscopy",
        "fundus",
    }

    REGION_KEYWORDS = {
        "brain": {"brain", "head"},
        "chest": {"chest", "thorax", "lung", "cxr"},
        "abdomen": {"abdomen", "abdominal", "liver", "kidney", "pelvis"},
        "breast": {"breast", "mammo", "mammogram", "mammography"},
        "skin": {"skin", "derm", "dermoscopy", "lesion", "melanoma"},
        "retina": {"retina", "retinal", "fundus", "eye", "ocular"},
        "bone": {"bone", "wrist", "hand", "elbow", "shoulder", "finger", "humerus", "forearm", "mura"},
    }

    MODALITY_KEYWORDS = {
        "mri": {"mri", "mr"},
        "ct": {"ct"},
        "xray": {"xray", "x-ray", "radiograph", "cxr"},
        "mammography": {"mammography", "mammogram", "mammo"},
        "dermoscopy": {"dermoscopy", "derm"},
        "fundus": {"fundus", "retinal", "ocular"},
    }

    def _collect_text(self, filename: str | None, content_type: str | None) -> str:
        parts = []
        if filename:
            parts.append(filename.lower())
        if content_type:
            parts.append(content_type.lower())
        return " ".join(parts)

    def _match_label(self, text: str, keyword_map: dict[str, set[str]]) -> tuple[str | None, float]:
        matches = []

        for label, keywords in keyword_map.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                matches.append((label, score))

        if not matches:
            return None, 0.0

        matches.sort(key=lambda item: item[1], reverse=True)

        if len(matches) > 1 and matches[0][1] == matches[1][1]:
            return None, 0.5

        best_label, best_score = matches[0]
        return best_label, 0.95 if best_score >= 2 else 0.85

    def _tensor_is_xray_like(self, tensor: torch.Tensor | None) -> bool:
        if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            return False

        working = tensor.detach().float().cpu()

        if working.shape[1] != 3:
            return False

        ch0 = working[:, 0]
        ch1 = working[:, 1]
        ch2 = working[:, 2]

        channel_gap = (
            (ch0 - ch1).abs().mean()
            + (ch1 - ch2).abs().mean()
            + (ch0 - ch2).abs().mean()
        ) / 3.0

        return float(channel_gap) < 1e-4

    def predict(
        self,
        filename: str | None = None,
        content_type: str | None = None,
        tensor: torch.Tensor | None = None,
    ) -> dict:
        text = self._collect_text(filename, content_type)

        region, region_conf = self._match_label(text, self.REGION_KEYWORDS)
        modality, modality_conf = self._match_label(text, self.MODALITY_KEYWORDS)

        if region is not None and modality is not None:
            confidence = min(region_conf, modality_conf)
            return {
                "region": region,
                "modality": modality,
                "confidence": confidence,
                "requires_confirmation": confidence < 0.80,
                "supported": (
                    region in self.SUPPORTED_REGIONS and modality in self.SUPPORTED_MODALITIES
                ),
                "reason": None,
            }

        if self._tensor_is_xray_like(tensor):
            fallback_region = region or "bone"
            fallback_modality = modality or "xray"

            return {
                "region": fallback_region,
                "modality": fallback_modality,
                "confidence": 0.81,
                "requires_confirmation": False,
                "supported": (
                    fallback_region in self.SUPPORTED_REGIONS
                    and fallback_modality in self.SUPPORTED_MODALITIES
                ),
                "reason": "Used temporary grayscale X-ray fallback routing.",
            }

        if region is not None or modality is not None:
            return {
                "region": region,
                "modality": modality,
                "confidence": min(
                    region_conf if region is not None else 0.60,
                    modality_conf if modality is not None else 0.60,
                ),
                "requires_confirmation": True,
                "supported": False,
                "reason": "Only partial routing information could be inferred.",
            }

        return {
            "region": None,
            "modality": None,
            "confidence": 0.0,
            "requires_confirmation": True,
            "supported": False,
            "reason": "Could not infer region or modality from available input hints.",
        }