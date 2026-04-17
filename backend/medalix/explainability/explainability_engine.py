from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image


class ExplainabilityEngine:
    def __init__(self, model, output_dir: str = "logs/heatmaps"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        original_image: Image.Image,
        tensor: torch.Tensor,
        inference_result: dict,
        filename: str,
    ) -> dict:
        if self.model is None:
            raise ValueError("Explainability model is missing")

        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Explainability input tensor must be a torch.Tensor")

        if tensor.ndim != 4:
            raise ValueError(
                f"Expected explainability tensor shape [B, C, H, W], got {tuple(tensor.shape)}"
            )

        target_layers = self._resolve_target_layers()
        if not target_layers:
            raise ValueError("No valid target layer found for explainability generation")

        rgb_image = original_image.convert("RGB").resize((224, 224))
        rgb_array = np.array(rgb_image).astype(np.float32) / 255.0

        cam = GradCAMPlusPlus(model=self.model, target_layers=target_layers)

        grayscale_cam = cam(input_tensor=tensor)[0]
        visualization = show_cam_on_image(
            rgb_array,
            grayscale_cam,
            use_rgb=True,
            image_weight=0.65,
        )

        safe_name = Path(filename).stem.replace(" ", "_")
        heatmap_path = self.output_dir / f"{safe_name}_gradcampp.png"

        plt.figure(figsize=(6, 6))
        plt.imshow(visualization)
        plt.axis("off")
        plt.savefig(heatmap_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        return {
            "method": "Grad-CAM++",
            "heatmap_path": str(heatmap_path),
            "warning": "",
        }

    def _resolve_target_layers(self):
        if hasattr(self.model, "densenet") and hasattr(self.model.densenet, "features"):
            features = self.model.densenet.features
            if len(features) > 0:
                return [features[-1]]

        raise ValueError("Unsupported model structure for Grad-CAM++ target layer resolution")