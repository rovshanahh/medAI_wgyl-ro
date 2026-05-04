from pathlib import Path
import copy
import csv
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


RAW_ROOT = Path("raw_datasets/ham10000")
METADATA_CSV = RAW_ROOT / "HAM10000_metadata.csv"

IMAGE_DIRS = [
    RAW_ROOT / "HAM10000_images_part_1",
    RAW_ROOT / "HAM10000_images_part_2",
]

MODEL_OUT = Path("reference_data/models/skin_dermoscopy_resnet18.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-4
SEED = 42

CLASS_ORDER = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]

CLASS_NAMES = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def find_image_path(image_id: str) -> Path:
    for image_dir in IMAGE_DIRS:
        candidate = image_dir / f"{image_id}.jpg"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Image not found for image_id={image_id}")


def read_metadata() -> list[tuple[Path, int, str]]:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_CSV}")

    samples = []

    with METADATA_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"image_id", "dx"}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(f"Metadata CSV is missing columns: {sorted(missing)}")

        for row in reader:
            image_id = row["image_id"].strip()
            dx = row["dx"].strip().lower()

            if dx not in CLASS_ORDER:
                continue

            image_path = find_image_path(image_id)
            label = CLASS_ORDER.index(dx)
            samples.append((image_path, label, dx))

    if not samples:
        raise RuntimeError("No valid HAM10000 samples were found.")

    return samples


def stratified_split(samples: list[tuple[Path, int, str]]) -> tuple[list, list, list]:
    by_label = {label: [] for label in range(len(CLASS_ORDER))}

    for sample in samples:
        by_label[sample[1]].append(sample)

    train_samples = []
    val_samples = []
    test_samples = []

    for label, label_samples in by_label.items():
        random.shuffle(label_samples)

        total = len(label_samples)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        train_samples.extend(label_samples[:train_end])
        val_samples.extend(label_samples[train_end:val_end])
        test_samples.extend(label_samples[val_end:])

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    return train_samples, val_samples, test_samples


class Ham10000Dataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int, str]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label, _ = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def build_transforms():
    train_tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.92, 1.08),
            ),
            transforms.ColorJitter(
                brightness=0.12,
                contrast=0.12,
                saturation=0.08,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_tf, eval_tf


def build_loaders():
    all_samples = read_metadata()
    train_samples, val_samples, test_samples = stratified_split(all_samples)

    train_tf, eval_tf = build_transforms()

    train_ds = Ham10000Dataset(train_samples, transform=train_tf)
    val_ds = Ham10000Dataset(val_samples, transform=eval_tf)
    test_ds = Ham10000Dataset(test_samples, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def class_counts(samples: list[tuple[Path, int, str]]) -> dict:
    counts = {CLASS_NAMES[class_key]: 0 for class_key in CLASS_ORDER}

    for _, _, dx in samples:
        counts[CLASS_NAMES[dx]] += 1

    return counts


def build_model(device: torch.device) -> nn.Module:
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_ORDER))
    return model.to(device)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += images.size(0)

    return total_loss / total_count, total_correct / total_count


def evaluate_details(model, loader, device):
    model.eval()

    readable_labels = [CLASS_NAMES[class_key] for class_key in CLASS_ORDER]

    confusion = {
        true_label: {pred_label: 0 for pred_label in readable_labels}
        for true_label in readable_labels
    }

    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            for true_idx, pred_idx in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                true_label = readable_labels[true_idx]
                pred_label = readable_labels[pred_idx]

                confusion[true_label][pred_label] += 1
                total += 1

                if true_idx == pred_idx:
                    correct += 1

    accuracy = correct / total if total else 0.0
    return accuracy, confusion


def print_confusion(confusion: dict) -> None:
    labels = list(confusion.keys())

    print("\nConfusion matrix:")
    print("true \\ pred".ljust(28) + "".join(label[:18].ljust(20) for label in labels))

    for true_label in labels:
        row = true_label[:26].ljust(28)

        for pred_label in labels:
            row += str(confusion[true_label][pred_label]).ljust(20)

        print(row)


def main() -> None:
    set_seed(SEED)

    device = get_device()
    print("Device:", device)

    (
        train_ds,
        val_ds,
        test_ds,
        train_loader,
        val_loader,
        test_loader,
    ) = build_loaders()

    print("Classes:", [CLASS_NAMES[class_key] for class_key in CLASS_ORDER])
    print(f"Train size: {len(train_ds)}")
    print(f"Val size:   {len(val_ds)}")
    print(f"Test size:  {len(test_ds)}")

    print("\nTrain class counts:")
    print(class_counts(train_ds.samples))

    print("\nVal class counts:")
    print(class_counts(val_ds.samples))

    print("\nTest class counts:")
    print(class_counts(test_ds.samples))

    model = build_model(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            print(f"  ✓ New best val_acc: {best_val_acc:.4f}")

    if best_state is None:
        raise RuntimeError("Training failed. No best model state was saved.")

    model.load_state_dict(best_state)

    test_acc, confusion = evaluate_details(
        model=model,
        loader=test_loader,
        device=device,
    )

    print(f"\nTest accuracy: {test_acc:.4f}")
    print_confusion(confusion)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state,
            "class_order": CLASS_ORDER,
            "class_names": [CLASS_NAMES[class_key] for class_key in CLASS_ORDER],
            "num_classes": len(CLASS_ORDER),
            "image_size": IMAGE_SIZE,
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "confusion": confusion,
            "architecture": "resnet18",
            "task": "skin_dermoscopy_lesion_classification",
            "dataset": "HAM10000",
        },
        MODEL_OUT,
    )

    print(f"\nSaved to: {MODEL_OUT}")
    print(f"Best val_acc: {best_val_acc:.4f}")
    print(f"Test acc:     {test_acc:.4f}")


if __name__ == "__main__":
    main()