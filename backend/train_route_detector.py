from pathlib import Path
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


DATA_ROOT = Path("datasets/routing")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

MODEL_OUT = Path("reference_data/route_detector/route_detector_model.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-4
SEED = 42

EXPECTED_CLASSES = [
    "bone_xray",
    "brain_mri",
    "chest_xray",
    "retina_fundus",
    "unknown",
]


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


def validate_dataset_structure() -> None:
    required_dirs = []

    for split in ["train", "val", "test"]:
        for class_name in EXPECTED_CLASSES:
            required_dirs.append(DATA_ROOT / split / class_name)

    missing = [str(path) for path in required_dirs if not path.exists()]

    if missing:
        raise FileNotFoundError("Missing folders:\n" + "\n".join(missing))


def build_loaders():
    train_tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
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
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_tf)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_tf)

    if train_ds.classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Unexpected class order: {train_ds.classes}. "
            f"Expected: {EXPECTED_CLASSES}"
        )

    if val_ds.classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Unexpected val class order: {val_ds.classes}. "
            f"Expected: {EXPECTED_CLASSES}"
        )

    if test_ds.classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Unexpected test class order: {test_ds.classes}. "
            f"Expected: {EXPECTED_CLASSES}"
        )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def build_model(device: torch.device, num_classes: int) -> nn.Module:
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
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


def evaluate_details(model, loader, device, class_names):
    model.eval()

    confusion = {
        true_label: {pred_label: 0 for pred_label in class_names}
        for true_label in class_names
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
                true_label = class_names[true_idx]
                pred_label = class_names[pred_idx]

                confusion[true_label][pred_label] += 1
                total += 1

                if true_label == pred_label:
                    correct += 1

    accuracy = correct / total if total else 0.0
    return accuracy, confusion


def print_confusion(confusion: dict) -> None:
    labels = list(confusion.keys())

    print("\nConfusion matrix:")
    print("true \\ pred".ljust(16) + "".join(label[:12].ljust(14) for label in labels))

    for true_label in labels:
        row = true_label.ljust(16)

        for pred_label in labels:
            row += str(confusion[true_label][pred_label]).ljust(14)

        print(row)


def main() -> None:
    set_seed(SEED)
    validate_dataset_structure()

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

    print("Classes:", train_ds.classes)
    print(f"Train size: {len(train_ds)}")
    print(f"Val size:   {len(val_ds)}")
    print(f"Test size:  {len(test_ds)}")

    model = build_model(device, num_classes=len(EXPECTED_CLASSES))

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
        class_names=train_ds.classes,
    )

    print(f"\nTest accuracy: {test_acc:.4f}")
    print_confusion(confusion)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state,
            "class_names": train_ds.classes,
            "image_size": IMAGE_SIZE,
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "confusion": confusion,
            "purpose": (
                "Route detector: brain_mri vs bone_xray vs chest_xray "
                "vs retina_fundus vs unknown"
            ),
        },
        MODEL_OUT,
    )

    print(f"\nSaved to: {MODEL_OUT}")
    print(f"Best val_acc: {best_val_acc:.4f}")
    print(f"Test acc:     {test_acc:.4f}")


if __name__ == "__main__":
    main()