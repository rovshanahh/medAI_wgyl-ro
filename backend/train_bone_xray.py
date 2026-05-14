from pathlib import Path
from collections import Counter
import argparse
import copy
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import classification_report, f1_score


DATA_ROOT = Path("raw_datasets/MURA-v1.1")
TRAIN_ROOT = DATA_ROOT / "train"
VAL_ROOT = DATA_ROOT / "valid"

CLASS_NAMES = ["Normal", "Abnormal"]

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4
SEED = 42

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


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


def infer_label(path: Path) -> int:
    text = str(path).lower()

    if "positive" in text:
        return 1

    if "negative" in text:
        return 0

    raise ValueError(f"Could not infer MURA label from path: {path}")


def collect_samples(root: Path) -> list[tuple[Path, int]]:
    if not root.exists():
        raise FileNotFoundError(f"Missing MURA folder: {root}")

    samples = []

    for path in root.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            samples.append((path, infer_label(path)))

    if not samples:
        raise RuntimeError(f"No images found in {root}")

    random.shuffle(samples)
    return samples


class MuraBoneDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_tf, eval_tf


def class_counts(samples):
    counts = Counter(label for _, label in samples)
    return {CLASS_NAMES[label]: counts[label] for label in range(len(CLASS_NAMES))}


def build_weighted_sampler(samples):
    counts = Counter(label for _, label in samples)

    weights = []
    for _, label in samples:
        weights.append(1.0 / max(counts[label], 1))

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_class_weights(samples, device):
    counts = Counter(label for _, label in samples)
    total = sum(counts.values())
    num_classes = len(CLASS_NAMES)

    weights = []
    for label in range(num_classes):
        weights.append(total / (num_classes * max(counts[label], 1)))

    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    weights = weights / weights.mean()
    return weights


def build_loaders():
    train_tf, eval_tf = build_transforms()

    train_samples = collect_samples(TRAIN_ROOT)
    val_samples = collect_samples(VAL_ROOT)

    train_ds = MuraBoneDataset(train_samples, transform=train_tf)
    val_ds = MuraBoneDataset(val_samples, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=build_weighted_sampler(train_samples),
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_ds, val_ds, train_loader, val_loader


def build_model(device):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model.to(device)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        total_loss += loss.item() * images.size(0)
        total_correct += (preds.cpu() == labels.cpu().long()).sum().item()
        total_count += images.size(0)

    return total_loss / total_count, total_correct / total_count


def evaluate_details(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs >= 0.5).long().tolist()

            y_true.extend(labels.tolist())
            y_pred.extend(preds)

    accuracy = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    confusion = {
        true_name: {pred_name: 0 for pred_name in CLASS_NAMES}
        for true_name in CLASS_NAMES
    }

    for true_label, pred_label in zip(y_true, y_pred):
        confusion[CLASS_NAMES[true_label]][CLASS_NAMES[pred_label]] += 1

    return accuracy, macro_f1, weighted_f1, confusion, report


def print_confusion(confusion):
    print("\nConfusion matrix:")
    print("true \\ pred".ljust(14) + "".join(label.ljust(12) for label in CLASS_NAMES))

    for true_label in CLASS_NAMES:
        row = true_label.ljust(14)
        for pred_label in CLASS_NAMES:
            row += str(confusion[true_label][pred_label]).ljust(12)
        print(row)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default="reference_data/models/bone/bone_seed_42.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_path = Path(args.output)
    device = get_device()

    print("Device:", device)
    print("Seed:", args.seed)
    print("Output:", output_path)

    train_ds, val_ds, train_loader, val_loader = build_loaders()

    print("Classes:", CLASS_NAMES)
    print("Train size:", len(train_ds))
    print("Val size:", len(val_ds))

    print("\nTrain class counts:")
    print(class_counts(train_ds.samples))

    print("\nVal class counts:")
    print(class_counts(val_ds.samples))

    model = build_model(device)

    pos_weight = build_class_weights(train_ds.samples, device)[1]
    print("\nPositive class weight:", float(pos_weight.detach().cpu().item()))

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_macro_f1 = -1.0
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        val_acc_eval, val_macro_f1, val_weighted_f1, _, _ = evaluate_details(
            model,
            val_loader,
            device,
        )

        scheduler.step()

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_acc = val_acc_eval
            best_state = copy.deepcopy(model.state_dict())
            print(f"  ✓ New best val_macro_f1: {best_val_macro_f1:.4f}")

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    model.load_state_dict(best_state)

    val_acc, val_macro_f1, val_weighted_f1, confusion, report = evaluate_details(
        model,
        val_loader,
        device,
    )

    print(f"\nFinal val accuracy:    {val_acc:.4f}")
    print(f"Final val macro-F1:    {val_macro_f1:.4f}")
    print(f"Final val weighted-F1: {val_weighted_f1:.4f}")
    print_confusion(confusion)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state,
            "class_names": CLASS_NAMES,
            "num_classes": 2,
            "image_size": IMAGE_SIZE,
            "best_val_acc": best_val_acc,
            "best_val_macro_f1": best_val_macro_f1,
            "final_val_acc": val_acc,
            "final_val_macro_f1": val_macro_f1,
            "final_val_weighted_f1": val_weighted_f1,
            "confusion": confusion,
            "classification_report": report,
            "architecture": "densenet121",
            "task": "mura_bone_xray_binary_classification",
            "dataset": "MURA-v1.1",
            "seed": args.seed,
            "ensemble_member": True,
        },
        output_path,
    )

    print(f"\nSaved to: {output_path}")
    print(f"Best val_macro-F1: {best_val_macro_f1:.4f}")


if __name__ == "__main__":
    main()
