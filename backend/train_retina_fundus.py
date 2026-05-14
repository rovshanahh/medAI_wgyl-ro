from pathlib import Path
import argparse
import copy
import csv
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.metrics import classification_report, f1_score


RAW_ROOT = Path("raw_datasets/aptos2019")

TRAIN_CSV = RAW_ROOT / "train_1.csv"
VAL_CSV = RAW_ROOT / "valid.csv"
TEST_CSV = RAW_ROOT / "test.csv"

TRAIN_DIR = RAW_ROOT / "train_images"
VAL_DIR = RAW_ROOT / "val_images"
TEST_DIR = RAW_ROOT / "test_images"

MODEL_OUT = Path("reference_data/models/retina_fundus_resnet18.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-4
SEED = 42

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]


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


def find_image_path(image_dir: Path, image_id: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Image not found for id_code={image_id} in {image_dir}")


def read_label_csv(csv_path: Path, image_dir: Path) -> list[tuple[Path, int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    samples = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if "id_code" not in reader.fieldnames or "diagnosis" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} must contain columns: id_code, diagnosis. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            image_id = row["id_code"].strip()
            diagnosis = int(row["diagnosis"])

            if diagnosis < 0 or diagnosis > 4:
                raise ValueError(f"Invalid diagnosis label {diagnosis} for image {image_id}")

            image_path = find_image_path(image_dir, image_id)
            samples.append((image_path, diagnosis))

    return samples


class AptosRetinaDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def build_transforms():
    train_tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
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

    train_samples = read_label_csv(TRAIN_CSV, TRAIN_DIR)
    val_samples = read_label_csv(VAL_CSV, VAL_DIR)
    test_samples = read_label_csv(TEST_CSV, TEST_DIR)

    train_ds = AptosRetinaDataset(train_samples, transform=train_tf)
    val_ds = AptosRetinaDataset(val_samples, transform=eval_tf)
    test_ds = AptosRetinaDataset(test_samples, transform=eval_tf)

    train_sampler = build_weighted_sampler(train_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def build_model(device: torch.device) -> nn.Module:
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model.to(device)


def class_counts(samples: list[tuple[Path, int]]) -> dict:
    counts = {name: 0 for name in CLASS_NAMES}

    for _, label in samples:
        counts[CLASS_NAMES[label]] += 1

    return counts

def raw_label_counts(samples: list[tuple[Path, int]]) -> dict[int, int]:
    counts = {label: 0 for label in range(len(CLASS_NAMES))}

    for _, label in samples:
        counts[label] += 1

    return counts


def build_weighted_sampler(samples: list[tuple[Path, int]]) -> WeightedRandomSampler:
    counts = raw_label_counts(samples)

    sample_weights = []
    for _, label in samples:
        sample_weights.append(1.0 / max(counts[label], 1))

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_class_weights(samples: list[tuple[Path, int]], device: torch.device) -> torch.Tensor:
    counts = raw_label_counts(samples)
    total = sum(counts.values())
    num_classes = len(CLASS_NAMES)

    weights = []
    for label in range(num_classes):
        weights.append(total / (num_classes * max(counts[label], 1)))

    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    weights = weights / weights.mean()
    return weights


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

    confusion = {
        true_label: {pred_label: 0 for pred_label in CLASS_NAMES}
        for true_label in CLASS_NAMES
    }

    total = 0
    correct = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            for true_idx, pred_idx in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                true_label = CLASS_NAMES[true_idx]
                pred_label = CLASS_NAMES[pred_idx]

                confusion[true_label][pred_label] += 1
                total += 1
                y_true.append(true_idx)
                y_pred.append(pred_idx)

                if true_idx == pred_idx:
                    correct += 1

    accuracy = correct / total if total else 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    return accuracy, macro_f1, weighted_f1, confusion, report


def print_confusion(confusion: dict) -> None:
    labels = list(confusion.keys())

    print("\nConfusion matrix:")
    print("true \\ pred".ljust(20) + "".join(label[:15].ljust(17) for label in labels))

    for true_label in labels:
        row = true_label.ljust(20)

        for pred_label in labels:
            row += str(confusion[true_label][pred_label]).ljust(17)

        print(row)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default=str(MODEL_OUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    output_path = Path(args.output)

    device = get_device()
    print("Device:", device)
    print("Seed:", args.seed)
    print("Output:", output_path)

    (
        train_ds,
        val_ds,
        test_ds,
        train_loader,
        val_loader,
        test_loader,
    ) = build_loaders()

    print("Classes:", CLASS_NAMES)
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

    class_weights = build_class_weights(train_ds.samples, device)

    print("\nClass weights:")
    for class_name, weight in zip(CLASS_NAMES, class_weights.detach().cpu().tolist()):
        print(f"  {class_name}: {weight:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = -1.0
    best_val_macro_f1 = -1.0
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

        val_acc_eval, val_macro_f1, val_weighted_f1, _, _ = evaluate_details(
            model=model,
            loader=val_loader,
            device=device,
        )

        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_acc = val_acc_eval
            best_state = copy.deepcopy(model.state_dict())
            print(
                f"  ✓ New best val_macro_f1: {best_val_macro_f1:.4f} "
                f"(val_acc={best_val_acc:.4f})"
            )

    if best_state is None:
        raise RuntimeError("Training failed. No best model state was saved.")

    model.load_state_dict(best_state)

    test_acc, test_macro_f1, test_weighted_f1, confusion, classification_metrics = evaluate_details(
        model=model,
        loader=test_loader,
        device=device,
    )

    print(f"\nTest accuracy:    {test_acc:.4f}")
    print(f"Test macro-F1:    {test_macro_f1:.4f}")
    print(f"Test weighted-F1: {test_weighted_f1:.4f}")
    print_confusion(confusion)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state,
            "class_names": CLASS_NAMES,
            "num_classes": len(CLASS_NAMES),
            "image_size": IMAGE_SIZE,
            "best_val_acc": best_val_acc,
            "best_val_macro_f1": best_val_macro_f1,
            "test_acc": test_acc,
            "test_macro_f1": test_macro_f1,
            "test_weighted_f1": test_weighted_f1,
            "confusion": confusion,
            "classification_report": classification_metrics,
            "architecture": "resnet18",
            "task": "retina_fundus_dr_severity",
            "dataset": "APTOS 2019 Blindness Detection",
            "seed": args.seed,
            "ensemble_member": True,
        },
        output_path,
    )

    print(f"\nSaved to: {output_path}")
    print(f"Best val_acc:      {best_val_acc:.4f}")
    print(f"Best val_macro-F1: {best_val_macro_f1:.4f}")
    print(f"Test acc:          {test_acc:.4f}")
    print(f"Test macro-F1:     {test_macro_f1:.4f}")


if __name__ == "__main__":
    main()