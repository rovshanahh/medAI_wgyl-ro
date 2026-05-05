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


RAW_ROOT = Path("raw_datasets/cbis_ddsm")
CSV_ROOT = RAW_ROOT / "csv"
JPEG_ROOT = RAW_ROOT / "jpeg"

DICOM_INFO_CSV = CSV_ROOT / "dicom_info.csv"

CASE_CSVS = [
    CSV_ROOT / "mass_case_description_train_set.csv",
    CSV_ROOT / "mass_case_description_test_set.csv",
    CSV_ROOT / "calc_case_description_train_set.csv",
    CSV_ROOT / "calc_case_description_test_set.csv",
]

MODEL_OUT = Path("reference_data/models/breast_mammography_resnet18.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-4
SEED = 42

CLASS_NAMES = [
    "Benign",
    "Malignant",
]

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


def normalize_path(value: str) -> str:
    return (value or "").strip().replace("\\", "/").replace("\n", "")


def normalize_pathology(value: str) -> int | None:
    pathology = normalize_path(value).upper()

    if pathology in {"BENIGN", "BENIGN_WITHOUT_CALLBACK"}:
        return 0

    if pathology == "MALIGNANT":
        return 1

    return None


def build_patient_to_cropped_jpeg_map() -> dict[str, list[Path]]:
    if not DICOM_INFO_CSV.exists():
        raise FileNotFoundError(f"Missing DICOM info CSV: {DICOM_INFO_CSV}")

    mapping: dict[str, list[Path]] = {}

    with DICOM_INFO_CSV.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)

        for row in reader:
            patient_id = normalize_path(row.get("PatientID", ""))
            image_path = normalize_path(row.get("image_path", ""))
            series_description = normalize_path(row.get("SeriesDescription", "")).lower()
            body_part = normalize_path(row.get("BodyPartExamined", "")).upper()
            modality = normalize_path(row.get("Modality", "")).upper()

            if not patient_id or not image_path:
                continue

            if body_part != "BREAST":
                continue

            if modality != "MG":
                continue

            if "cropped images" not in series_description:
                continue

            jpeg_relative = image_path.replace("CBIS-DDSM/jpeg/", "")
            jpeg_path = JPEG_ROOT / jpeg_relative

            if jpeg_path.exists():
                mapping.setdefault(patient_id, []).append(jpeg_path)

    if not mapping:
        raise RuntimeError("No PatientID-to-cropped-JPEG mappings found.")

    return mapping


def read_case_samples() -> list[tuple[Path, int]]:
    patient_to_jpegs = build_patient_to_cropped_jpeg_map()
    samples = []

    for csv_path in CASE_CSVS:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing case CSV: {csv_path}")

        with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)

            required_columns = {"cropped image file path", "pathology"}
            missing = required_columns - set(reader.fieldnames or [])

            if missing:
                raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

            for row in reader:
                label = normalize_pathology(row.get("pathology", ""))

                if label is None:
                    continue

                cropped_path = normalize_path(row.get("cropped image file path", ""))

                if not cropped_path:
                    continue

                patient_key = cropped_path.split("/")[0]
                jpeg_paths = patient_to_jpegs.get(patient_key, [])

                for jpeg_path in jpeg_paths:
                    if jpeg_path.exists():
                        samples.append((jpeg_path, label))

    samples = list(dict.fromkeys(samples))

    if not samples:
        raise RuntimeError("No matched cropped mammography samples found.")

    print(f"Matched cropped mammography samples: {len(samples)}")

    return samples


def stratified_split(samples: list[tuple[Path, int]]) -> tuple[list, list, list]:
    by_label = {label: [] for label in range(len(CLASS_NAMES))}

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


class MammographyDataset(Dataset):
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


def class_counts(samples: list[tuple[Path, int]]) -> dict:
    counts = {name: 0 for name in CLASS_NAMES}

    for _, label in samples:
        counts[CLASS_NAMES[label]] += 1

    return counts


def build_loaders():
    all_samples = read_case_samples()
    train_samples, val_samples, test_samples = stratified_split(all_samples)

    train_tf, eval_tf = build_transforms()

    train_ds = MammographyDataset(train_samples, transform=train_tf)
    val_ds = MammographyDataset(val_samples, transform=eval_tf)
    test_ds = MammographyDataset(test_samples, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def build_model(device: torch.device) -> nn.Module:
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
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

    confusion = {
        true_label: {pred_label: 0 for pred_label in CLASS_NAMES}
        for true_label in CLASS_NAMES
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
                true_label = CLASS_NAMES[true_idx]
                pred_label = CLASS_NAMES[pred_idx]

                confusion[true_label][pred_label] += 1
                total += 1

                if true_idx == pred_idx:
                    correct += 1

    accuracy = correct / total if total else 0.0
    return accuracy, confusion


def print_confusion(confusion: dict) -> None:
    labels = list(confusion.keys())

    print("\nConfusion matrix:")
    print("true \\ pred".ljust(16) + "".join(label.ljust(14) for label in labels))

    for true_label in labels:
        row = true_label.ljust(16)

        for pred_label in labels:
            row += str(confusion[true_label][pred_label]).ljust(14)

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
            "class_names": CLASS_NAMES,
            "num_classes": len(CLASS_NAMES),
            "image_size": IMAGE_SIZE,
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "confusion": confusion,
            "architecture": "resnet18",
            "task": "breast_mammography_benign_malignant_classification",
            "dataset": "CBIS-DDSM",
        },
        MODEL_OUT,
    )

    print(f"\nSaved to: {MODEL_OUT}")
    print(f"Best val_acc: {best_val_acc:.4f}")
    print(f"Test acc:     {test_acc:.4f}")


if __name__ == "__main__":
    main()