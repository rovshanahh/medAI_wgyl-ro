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
        target_layers = [self.model.densenet.features[-1]]

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