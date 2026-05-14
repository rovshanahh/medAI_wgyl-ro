from pathlib import Path
from collections import Counter
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


DATA_ROOT = Path("datasets/abdomen_ct")
OUTPUT_PATH = Path("reference_data/models/abdomen_ct_resnet18.pth")

CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-4
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, eval_transform


def validate_dataset_structure():
    for split in ["train", "val", "test"]:
        split_dir = DATA_ROOT / split

        if not split_dir.exists():
            raise RuntimeError(f"Missing split folder: {split_dir}")

        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name

            if not class_dir.exists():
                raise RuntimeError(f"Missing class folder: {class_dir}")


def count_classes(dataset):
    reverse_map = {index: name for name, index in dataset.class_to_idx.items()}
    counts = Counter()

    for _, label in dataset.samples:
        counts[reverse_map[label]] += 1

    return dict(counts)


def build_loaders():
    validate_dataset_structure()

    train_transform, eval_transform = build_transforms()

    train_dataset = datasets.ImageFolder(DATA_ROOT / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_ROOT / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(DATA_ROOT / "test", transform=eval_transform)

    if train_dataset.classes != CLASS_NAMES:
        raise RuntimeError(
            f"Unexpected classes: {train_dataset.classes}. Expected: {CLASS_NAMES}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def build_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model = model.to(device)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    return running_loss / total, correct / total, all_labels, all_predictions


def print_confusion_matrix(labels, predictions):
    matrix = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]

    for true_label, pred_label in zip(labels, predictions):
        matrix[true_label][pred_label] += 1

    print("\nConfusion matrix:")
    print("true \\ pred".ljust(18), end="")

    for class_name in CLASS_NAMES:
        print(class_name[:16].ljust(18), end="")

    print()

    for index, class_name in enumerate(CLASS_NAMES):
        print(class_name[:16].ljust(18), end="")

        for value in matrix[index]:
            print(str(value).ljust(18), end="")

        print()


def save_checkpoint(model, best_val_acc, test_acc, output_path: Path, seed: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "architecture": "resnet18",
        "image_size": IMAGE_SIZE,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "modality": "ct",
        "region": "abdomen",
        "task": "kidney_ct_classification",
        "seed": seed,
        "ensemble_member": True,
    }

    torch.save(checkpoint, output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)

    set_seed(args.seed)

    device = get_device()
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_path}")

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_loaders()

    print(f"Classes: {train_dataset.classes}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size:   {len(val_dataset)}")
    print(f"Test size:  {len(test_dataset)}")

    print("\nTrain class counts:")
    print(count_classes(train_dataset))

    print("\nVal class counts:")
    print(count_classes(val_dataset))

    print("\nTest class counts:")
    print(count_classes(test_dataset))

    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            print(f"  ✓ New best val_acc: {best_val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, labels, predictions = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"\nTest accuracy: {test_acc:.4f}")
    print_confusion_matrix(labels, predictions)

    save_checkpoint(
        model=model,
        best_val_acc=best_val_acc,
        test_acc=test_acc,
        output_path=output_path,
        seed=args.seed,
    )

    print(f"\nSaved to: {output_path}")
    print(f"Best val_acc: {best_val_acc:.4f}")
    print(f"Test acc:     {test_acc:.4f}")


if __name__ == "__main__":
    main()