from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps, ImageEnhance, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


RADIOLOGY_SAMPLES = [
    "test_samples/chest_xray.jpg",
    "test_samples/bone_xray.png",
    "test_samples/abdomen_ct.jpg",
    "test_samples/breast_mammography.jpg",
    "test_samples/brain_mri.jpg",
]

COLOR_MEDICAL_SAMPLES = [
    "test_samples/retina_fundus.jpg",
    "test_samples/skin_dermoscopy.jpg",
]

OUTPUT_PATH = Path("reference_data/input_gate/medical_image_gate_model.pth")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8
LR = 8e-5
SEED = 42
DATASET_LENGTH = 4500

CLASS_NAMES = ["non_medical", "radiology_medical", "color_medical"]

random.seed(SEED)
torch.manual_seed(SEED)


def load_image(path: Path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def augment_medical(image: Image.Image, color: bool) -> Image.Image:
    image = image.convert("RGB")

    if random.random() < 0.5:
        image = ImageOps.mirror(image)

    image = image.rotate(
        random.uniform(-8, 8),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )

    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.35))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))

    if color:
        image = ImageEnhance.Color(image).enhance(random.uniform(0.75, 1.35))
    else:
        if random.random() < 0.8:
            gray = image.convert("L")
            image = Image.merge("RGB", (gray, gray, gray))

    if random.random() < 0.35:
        arr = np.asarray(image).astype(np.float32)
        noise = np.random.normal(0, random.uniform(2, 8), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr)

    return image


def make_grayscale_photo(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    return Image.merge("RGB", (gray, gray, gray))


def make_document_like_image() -> Image.Image:
    image = Image.new("RGB", (224, 224), color=(245, 245, 240))
    draw = ImageDraw.Draw(image)

    for _ in range(random.randint(8, 18)):
        x1 = random.randint(10, 40)
        y = random.randint(10, 210)
        x2 = random.randint(120, 215)
        draw.line((x1, y, x2, y), fill=(random.randint(30, 120),) * 3, width=random.randint(1, 3))

    return image


def make_noise_image() -> Image.Image:
    mode = random.choice(["gray", "color", "dark", "bright"])

    if mode == "gray":
        arr = np.random.normal(120, 45, (224, 224)).clip(0, 255).astype(np.uint8)
        return Image.merge("RGB", (Image.fromarray(arr), Image.fromarray(arr), Image.fromarray(arr)))

    if mode == "dark":
        arr = np.random.normal(25, 15, (224, 224, 3)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    if mode == "bright":
        arr = np.random.normal(230, 15, (224, 224, 3)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class ThreeClassMedicalGateDataset(Dataset):
    def __init__(self, transform):
        self.transform = transform

        self.radiology_images = [
            img for p in RADIOLOGY_SAMPLES if (img := load_image(Path(p))) is not None
        ]
        self.color_medical_images = [
            img for p in COLOR_MEDICAL_SAMPLES if (img := load_image(Path(p))) is not None
        ]

        if not self.radiology_images:
            raise RuntimeError("No radiology medical samples found.")

        if not self.color_medical_images:
            raise RuntimeError("No color medical samples found.")

        self.cifar = datasets.CIFAR10(
            root="reference_data/non_medical_cifar10",
            train=True,
            download=True,
        )

        self.fashion = datasets.FashionMNIST(
            root="reference_data/non_medical_fashionmnist",
            train=True,
            download=True,
        )

        self.length = DATASET_LENGTH

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        bucket = index % 3

        if bucket == 0:
            negative_type = random.choice(["cifar_color", "cifar_gray", "fashion_gray", "document", "noise"])

            if negative_type == "cifar_color":
                image, _ = self.cifar[random.randint(0, len(self.cifar) - 1)]
                image = image.convert("RGB")

            elif negative_type == "cifar_gray":
                image, _ = self.cifar[random.randint(0, len(self.cifar) - 1)]
                image = make_grayscale_photo(image.convert("RGB"))

            elif negative_type == "fashion_gray":
                image, _ = self.fashion[random.randint(0, len(self.fashion) - 1)]
                image = image.convert("RGB")

            elif negative_type == "document":
                image = make_document_like_image()

            else:
                image = make_noise_image()

            label = 0

        elif bucket == 1:
            image = random.choice(self.radiology_images)
            image = augment_medical(image, color=False)
            label = 1

        else:
            image = random.choice(self.color_medical_images)
            image = augment_medical(image, color=True)
            label = 2

        return self.transform(image), torch.tensor(label, dtype=torch.long)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = ThreeClassMedicalGateDataset(transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch}: train_acc={correct / max(total, 1):.3f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "medical_threshold": 0.70,
            "non_medical_threshold": 0.65,
            "training_note": "Three-class first gate: non_medical, radiology_medical, color_medical.",
        },
        OUTPUT_PATH,
    )

    print(f"Saved model to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
