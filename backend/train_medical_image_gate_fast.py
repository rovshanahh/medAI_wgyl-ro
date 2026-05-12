from pathlib import Path
import random

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


MEDICAL_SAMPLES = [
    "test_samples/chest_xray.jpg",
    "test_samples/bone_xray.png",
    "test_samples/abdomen_ct.jpg",
    "test_samples/breast_mammography.jpg",
    "test_samples/brain_mri.jpg",
    "test_samples/retina_fundus.jpg",
    "test_samples/skin_dermoscopy.jpg",
]

OUTPUT_PATH = Path("reference_data/input_gate/medical_image_gate_model.pth")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


class FastMedicalGateDataset(Dataset):
    def __init__(self, transform):
        self.transform = transform

        self.medical_paths = [
            Path(path) for path in MEDICAL_SAMPLES if Path(path).exists()
        ]

        if not self.medical_paths:
            raise RuntimeError("No medical samples found in backend/test_samples.")

        self.cifar = datasets.CIFAR10(
            root="reference_data/non_medical_cifar10",
            train=True,
            download=True,
        )

        self.length = 1200

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index % 2 == 0:
            path = random.choice(self.medical_paths)
            image = Image.open(path).convert("RGB")

            # augment tiny medical sample set by flips / rotation / contrast
            if random.random() < 0.5:
                image = ImageOps.mirror(image)

            label = 1
        else:
            image, _ = self.cifar[random.randint(0, len(self.cifar) - 1)]
            image = image.convert("RGB")
            label = 0

        return self.transform(image), torch.tensor(label, dtype=torch.long)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = FastMedicalGateDataset(train_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

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

        acc = correct / max(total, 1)
        print(f"Epoch {epoch}: train_acc={acc:.3f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": ["non_medical", "medical"],
            "image_size": IMAGE_SIZE,
            "threshold": 0.75,
        },
        OUTPUT_PATH,
    )

    print(f"Saved model to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
