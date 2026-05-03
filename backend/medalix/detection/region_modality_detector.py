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

    ROUTES = {
        "brain_mri": {
            "region": "brain",
            "modality": "mri",
            "keywords": {
                "brain",
                "head",
                "mri",
                "mr",
                "glioma",
                "meningioma",
                "pituitary",
                "tumor",
                "tumour",
                "no_tumor",
                "notumor",
            },
        },
        "chest_xray": {
            "region": "chest",
            "modality": "xray",
            "keywords": {
                "chest",
                "thorax",
                "lung",
                "cxr",
                "xray",
                "x-ray",
                "radiograph",
                "pneumonia",
            },
        },
        "bone_xray": {
            "region": "bone",
            "modality": "xray",
            "keywords": {
                "bone",
                "wrist",
                "hand",
                "elbow",
                "shoulder",
                "finger",
                "humerus",
                "forearm",
                "mura",
                "xray",
                "x-ray",
                "radiograph",
            },
        },
    }

    def _collect_text(self, filename: str | None, content_type: str | None) -> str:
        parts = []

        if filename:
            parts.append(filename.lower())

        if content_type:
            parts.append(content_type.lower())

        return " ".join(parts)

    def _score_routes_from_text(self, text: str) -> dict[str, float]:
        scores = {}

        for route_name, route_info in self.ROUTES.items():
            keywords = route_info["keywords"]
            hit_count = sum(1 for keyword in keywords if keyword in text)

            if hit_count == 0:
                scores[route_name] = 0.0
            elif hit_count == 1:
                scores[route_name] = 0.70
            elif hit_count == 2:
                scores[route_name] = 0.85
            else:
                scores[route_name] = 0.95

        return scores

    def _is_grayscale_tensor(self, tensor: torch.Tensor | None) -> bool:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return False

        if tensor.ndim != 4:
            return False

        if tensor.shape[1] != 3:
            return False

        working = tensor.detach().float().cpu()

        ch0 = working[:, 0]
        ch1 = working[:, 1]
        ch2 = working[:, 2]

        channel_gap = (
            (ch0 - ch1).abs().mean()
            + (ch1 - ch2).abs().mean()
            + (ch0 - ch2).abs().mean()
        ) / 3.0

        return float(channel_gap) < 1e-4

    def _select_best_route(self, route_scores: dict[str, float]) -> tuple[str | None, float, bool]:
        ranked = sorted(route_scores.items(), key=lambda item: item[1], reverse=True)

        best_route, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        if best_score <= 0.0:
            return None, 0.0, True

        if best_score < 0.80:
            return best_route, best_score, True

        if best_score - second_score < 0.10:
            return best_route, best_score, True

        return best_route, best_score, False

    def predict(
        self,
        filename: str | None = None,
        content_type: str | None = None,
        tensor: torch.Tensor | None = None,
    ) -> dict:
        text = self._collect_text(filename, content_type)
        route_scores = self._score_routes_from_text(text)

        selected_route, confidence, requires_confirmation = self._select_best_route(route_scores)

        if selected_route is not None:
            route_info = self.ROUTES[selected_route]

            return {
                "route_label": selected_route,
                "region": route_info["region"],
                "modality": route_info["modality"],
                "confidence": confidence,
                "requires_confirmation": requires_confirmation,
                "supported": True,
                "route_scores": route_scores,
                "reason": (
                    "Route inferred from filename/content hints."
                    if not requires_confirmation
                    else "Route is possible but needs confirmation."
                ),
            }

        if self._is_grayscale_tensor(tensor):
            return {
                "route_label": None,
                "region": None,
                "modality": None,
                "confidence": 0.40,
                "requires_confirmation": True,
                "supported": False,
                "route_scores": route_scores,
                "reason": (
                    "Image is grayscale, but grayscale alone is not enough to decide "
                    "between MRI and X-ray."
                ),
            }

        return {
            "route_label": None,
            "region": None,
            "modality": None,
            "confidence": 0.0,
            "requires_confirmation": True,
            "supported": False,
            "route_scores": route_scores,
            "reason": "Could not infer region/modality from available input.",
        }