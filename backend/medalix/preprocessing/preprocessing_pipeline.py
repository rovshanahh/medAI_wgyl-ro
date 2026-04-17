from io import BytesIO

from PIL import Image, ImageOps
from torchvision import transforms


class PreprocessingPipeline:
    def __init__(self):
        self.transform = transforms.Compose(
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

    def run(self, raw_bytes: bytes):
        try:
            image = Image.open(BytesIO(raw_bytes))
            image.load()
        except Exception as exc:
            raise ValueError(f"Failed to decode uploaded image: {str(exc)}") from exc

        image = ImageOps.exif_transpose(image).convert("L")
        tensor = self.transform(image).unsqueeze(0)
        return image, tensor