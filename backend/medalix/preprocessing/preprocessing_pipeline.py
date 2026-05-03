from io import BytesIO

from PIL import Image, ImageOps
from torchvision import transforms


class PreprocessingPipeline:
    XRAY_MODALITIES = {"xray", "mammography"}
    RGB_MODALITIES = {"dermoscopy", "fundus"}

    def __init__(self):
        self._grayscale_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self._rgb_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _decode_image(self, raw_bytes: bytes) -> Image.Image:
        try:
            image = Image.open(BytesIO(raw_bytes))
            image.load()
            return ImageOps.exif_transpose(image)
        except Exception as exc:
            raise ValueError(f"Failed to decode uploaded image: {str(exc)}") from exc

    def _select_profile(self, modality: str | None) -> str:
        modality = (modality or "").strip().lower()

        if modality in self.XRAY_MODALITIES:
            return "grayscale_2d"

        if modality in {"ct", "mri"}:
            return "grayscale_2d"

        if modality in self.RGB_MODALITIES:
            return "rgb_2d"

        return "rgb_2d"

    def run(self, raw_bytes: bytes, modality: str | None = None):
        image = self._decode_image(raw_bytes)
        profile = self._select_profile(modality)

        if profile == "grayscale_2d":
            grayscale_image = image.convert("L")
            display_image = grayscale_image.convert("RGB")
            tensor = self._grayscale_transform(grayscale_image).unsqueeze(0)
            return display_image, tensor

        display_image = image.convert("RGB")
        tensor = self._rgb_transform(display_image).unsqueeze(0)
        return display_image, tensor