from pathlib import Path
import argparse
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, f1_score


PROJECT_DATA_ROOT = Path("raw_datasets/brain-tumor-mri-dataset")
DOWNLOADS_DATA_ROOT = Path.home() / "Downloads" / "brain-tumor-mri-dataset"

DATA_ROOT = PROJECT_DATA_ROOT if PROJECT_DATA_ROOT.exists() else DOWNLOADS_DATA_ROOT
MODEL_OUT = Path("models/brain/brain_mri_resnet18.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-4
SEED = 42

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


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



def class_counts(dataset) -> dict:
    counts = {class_name: 0 for class_name in dataset.classes}

    for _, label in dataset.samples:
        counts[dataset.classes[label]] += 1

    return counts


def build_weighted_sampler(dataset) -> WeightedRandomSampler:
    counts = class_counts(dataset)

    sample_weights = []
    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        sample_weights.append(1.0 / max(counts[class_name], 1))

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_class_weights(dataset, device: torch.device) -> torch.Tensor:
    counts = class_counts(dataset)
    total = sum(counts.values())
    num_classes = len(dataset.classes)

    weights = []
    for class_name in dataset.classes:
        weights.append(total / (num_classes * max(counts[class_name], 1)))

    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    weights = weights / weights.mean()
    return weights


def build_loaders():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = datasets.ImageFolder(DATA_ROOT / "Training", transform=train_transform)
    val_dataset   = datasets.ImageFolder(DATA_ROOT / "Testing",  transform=val_transform)

    print("Train classes:", train_dataset.classes)
    print("Val   classes:", val_dataset.classes)

    train_sampler = build_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def build_model(num_classes: int, device: torch.device):
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def evaluate_macro_f1(model, loader, device, class_names):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

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

        total_loss    += loss.item() * images.size(0)
        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_count   += images.size(0)

    return total_loss / total_count, total_correct / total_count



def evaluate_details(model, loader, device, class_names):
    model.eval()

    confusion = {
        true_label: {pred_label: 0 for pred_label in class_names}
        for true_label in class_names
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
                true_label = class_names[true_idx]
                pred_label = class_names[pred_idx]

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
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    return accuracy, macro_f1, weighted_f1, confusion, report


def print_confusion(confusion: dict) -> None:
    labels = list(confusion.keys())

    print("\nConfusion matrix:")
    print("true \\ pred".ljust(18) + "".join(label[:14].ljust(16) for label in labels))

    for true_label in labels:
        row = true_label[:16].ljust(18)

        for pred_label in labels:
            row += str(confusion[true_label][pred_label]).ljust(16)

        print(row)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default=str(MODEL_OUT))
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    output_path = Path(args.output)

    device = get_device()
    print("Device:", device)
    print("Seed:", args.seed)
    print("Output:", output_path)
    print("Data root:", DATA_ROOT)

    train_dataset, val_dataset, train_loader, val_loader = build_loaders()
    print(f"Train size: {len(train_dataset)}  |  Val size: {len(val_dataset)}")

    print("\nTrain class counts:")
    print(class_counts(train_dataset))

    print("\nVal class counts:")
    print(class_counts(val_dataset))

    num_classes = len(train_dataset.classes)
    model     = build_model(num_classes, device)
    class_weights = build_class_weights(train_dataset, device)

    print("\nClass weights:")
    for class_name, weight in zip(train_dataset.classes, class_weights.detach().cpu().tolist()):
        print(f"  {class_name}: {weight:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    best_val_acc = -1.0
    best_val_macro_f1 = -1.0
    best_state = None

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        val_macro_f1 = evaluate_macro_f1(
            model=model,
            loader=val_loader,
            device=device,
            class_names=train_dataset.classes,
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
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            print(
                f"  ✓ New best val_macro_f1: {best_val_macro_f1:.4f} "
                f"(val_acc={best_val_acc:.4f})"
            )

    model.load_state_dict(best_state)

    val_acc_final, val_macro_f1, val_weighted_f1, confusion, classification_metrics = evaluate_details(
        model=model,
        loader=val_loader,
        device=device,
        class_names=train_dataset.classes,
    )

    print(f"\nFinal val accuracy:    {val_acc_final:.4f}")
    print(f"Final val macro-F1:    {val_macro_f1:.4f}")
    print(f"Final val weighted-F1: {val_weighted_f1:.4f}")
    print_confusion(confusion)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "class_names": train_dataset.classes,
            "image_size": IMAGE_SIZE,
            "best_val_acc": best_val_acc,
            "best_val_macro_f1": best_val_macro_f1,
            "final_val_acc": val_acc_final,
            "final_val_macro_f1": val_macro_f1,
            "final_val_weighted_f1": val_weighted_f1,
            "confusion": confusion,
            "classification_report": classification_metrics,
            "architecture": "resnet18",
            "num_classes": num_classes,
            "seed": args.seed,
            "ensemble_member": True,
        },
        output_path,
    )

    print(f"\nSaved to: {output_path}")
    print(f"Best val_acc:          {best_val_acc:.4f}")
    print(f"Best val_macro-F1:     {best_val_macro_f1:.4f}")
    print(f"Final val macro-F1:    {val_macro_f1:.4f}")
    print(f"Final val weighted-F1: {val_weighted_f1:.4f}")


if __name__ == "__main__":
    main()