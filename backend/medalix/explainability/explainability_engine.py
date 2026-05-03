from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class BinaryLogitTarget:
    def __init__(self, positive: bool):
        self.positive = positive

    def __call__(self, model_output):
        if model_output.ndim == 2:
            logit = model_output[:, 0]
        else:
            logit = model_output[0]

        return logit if self.positive else -logit


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

        self.model.eval()

        target_layers = self._resolve_target_layers()
        if not target_layers:
            raise ValueError("No valid target layer found for explainability generation")

        tensor = tensor.detach().float()
        input_height = int(tensor.shape[-2])
        input_width = int(tensor.shape[-1])

        rgb_image = original_image.convert("RGB").resize((input_width, input_height))
        rgb_array = np.array(rgb_image).astype(np.float32) / 255.0

        targets = self._resolve_cam_targets(tensor, inference_result)

        with GradCAMPlusPlus(model=self.model, target_layers=target_layers) as cam:
            grayscale_cam = cam(
                input_tensor=tensor,
                targets=targets,
            )[0]

        visualization = show_cam_on_image(
            rgb_array,
            grayscale_cam,
            use_rgb=True,
            image_weight=0.65,
        )

        safe_name = Path(filename).stem.replace(" ", "_")
        heatmap_filename = f"{safe_name}_gradcampp.png"
        heatmap_path = self.output_dir / heatmap_filename

        plt.figure(figsize=(6, 6))
        plt.imshow(visualization)
        plt.axis("off")
        plt.savefig(heatmap_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        return {
            "method": "Grad-CAM++",
            "heatmap_path": f"/heatmaps/{heatmap_filename}",
            "warning": "",
            "target_label": inference_result.get("top_label"),
        }

    def _resolve_target_layers(self):
        # DenseNet121 — chest and bone models
        if hasattr(self.model, "densenet") and hasattr(self.model.densenet, "features"):
            features = self.model.densenet.features
            if len(features) > 0:
                return [features[-1]]

        # ResNet18 — brain MRI model
        if hasattr(self.model, "resnet") and hasattr(self.model.resnet, "layer4"):
            return [self.model.resnet.layer4[-1]]

        raise ValueError("Unsupported model structure for Grad-CAM++ target layer resolution")

    def _resolve_cam_targets(self, tensor: torch.Tensor, inference_result: dict):
        top_label = inference_result.get("top_label")
        probabilities = inference_result.get("probabilities", {}) or {}

        with torch.no_grad():
            output = self.model(tensor)

        if output.ndim == 1:
            output = output.unsqueeze(0)

        output_size = int(output.shape[1])

        # Bone binary model has one logit:
        # positive logit → Abnormal
        # negative logit → Normal
        if output_size == 1:
            is_abnormal = str(top_label).lower() == "abnormal"
            return [BinaryLogitTarget(positive=is_abnormal)]

        # Multiclass / multilabel models:
        # choose the index of top_label from probability keys if available.
        labels = list(probabilities.keys())

        if top_label in labels:
            return [ClassifierOutputTarget(labels.index(top_label))]

        # Safe fallback: use highest raw model output.
        target_index = int(torch.argmax(output, dim=1).item())
        return [ClassifierOutputTarget(target_index)]