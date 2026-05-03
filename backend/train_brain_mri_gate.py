"""
Train the Brain MRI input gate.

Binary classifier:
  brain_mri      -> accepted
  not_brain_mri  -> rejected

Dataset path:
  datasets/input_gates/brain_mri/
      train/brain_mri
      train/not_brain_mri
      val/brain_mri
      val/not_brain_mri
      test/brain_mri
      test/not_brain_mri
"""

from pathlib import Path
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


DATA_ROOT = Path("datasets/input_gates/brain_mri")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

MODEL_OUT = Path("reference_data/input_gate/brain_mri_input_gate_model.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-4
SEED = 42

EXPECTED_CLASSES = ["brain_mri", "not_brain_mri"]


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
    required_dirs = [
        TRAIN_DIR / "brain_mri",
        TRAIN_DIR / "not_brain_mri",
        VAL_DIR / "brain_mri",
        VAL_DIR / "not_brain_mri",
        TEST_DIR / "brain_mri",
        TEST_DIR / "not_brain_mri",
    ]

    missing = [str(path) for path in required_dirs if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing dataset folders:\n" + "\n".join(missing)
        )


def build_transforms():
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

    return train_tf, eval_tf


def build_loaders():
    train_tf, eval_tf = build_transforms()

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_tf)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_tf)

    if train_ds.classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Unexpected class order: {train_ds.classes}. "
            f"Expected: {EXPECTED_CLASSES}. "
            "Folder names must be exactly brain_mri and not_brain_mri."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def build_model(device: torch.device) -> nn.Module:
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
):
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


def evaluate_binary_details(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
):
    model.eval()

    brain_idx = class_names.index("brain_mri")
    not_brain_idx = class_names.index("not_brain_mri")

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            tp += ((preds == brain_idx) & (labels == brain_idx)).sum().item()
            tn += ((preds == not_brain_idx) & (labels == not_brain_idx)).sum().item()
            fp += ((preds == brain_idx) & (labels == not_brain_idx)).sum().item()
            fn += ((preds == not_brain_idx) & (labels == brain_idx)).sum().item()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    false_accept_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_reject_rate = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "accuracy": accuracy,
        "true_accepts": tp,
        "true_rejects": tn,
        "false_accepts": fp,
        "false_rejects": fn,
        "false_accept_rate": false_accept_rate,
        "false_reject_rate": false_reject_rate,
    }


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

    test_metrics = evaluate_binary_details(
        model=model,
        loader=test_loader,
        device=device,
        class_names=train_ds.classes,
    )

    print("\nTest results")
    print(f"Accuracy:           {test_metrics['accuracy']:.4f}")
    print(f"True accepts:       {test_metrics['true_accepts']}")
    print(f"True rejects:       {test_metrics['true_rejects']}")
    print(f"False accepts:      {test_metrics['false_accepts']}")
    print(f"False rejects:      {test_metrics['false_rejects']}")
    print(f"False accept rate:  {test_metrics['false_accept_rate']:.4f}")
    print(f"False reject rate:  {test_metrics['false_reject_rate']:.4f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state,
            "class_names": train_ds.classes,
            "image_size": IMAGE_SIZE,
            "best_val_acc": best_val_acc,
            "test_metrics": test_metrics,
            "acceptance_threshold": 0.90,
            "purpose": "Brain MRI input gate: brain_mri vs not_brain_mri",
        },
        MODEL_OUT,
    )

    print(f"\nSaved to: {MODEL_OUT}")
    print(f"Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()