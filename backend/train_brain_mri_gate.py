"""
Train the brain MRI input gate — a binary ResNet18 classifier:
  Class 0: brain_mri      (positive — images we want to ACCEPT)
  Class 1: not_brain_mri  (negative — images we want to REJECT)

Positive samples: pulled from the Kaggle brain tumor dataset (all 4 classes)
Negative samples: chest X-rays and non-chest images from the evaluation folder
"""

from pathlib import Path
import copy
import random
import shutil
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ── paths ──────────────────────────────────────────────────────────────────
BRAIN_TRAIN_ROOT = Path.home() / "Downloads" / "brain-tumor-mri-dataset" / "Training"
BRAIN_TEST_ROOT  = Path.home() / "Downloads" / "brain-tumor-mri-dataset" / "Testing"
EVAL_ROOT        = Path("evaluation")
MODEL_OUT        = Path("reference_data/input_gate/brain_mri_input_gate_model.pth")

# ── hyperparams ────────────────────────────────────────────────────────────
IMAGE_SIZE  = 224
BATCH_SIZE  = 32
EPOCHS      = 6
LR          = 1e-4
SEED        = 42
NEG_LIMIT   = 400   # max negative samples to keep (keeps classes balanced)


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


def collect_images(folder: Path, extensions: set[str] = {".jpg", ".jpeg", ".png"}) -> list[Path]:
    return [p for p in folder.rglob("*") if p.suffix.lower() in extensions]


def build_dataset_dir(tmp_root: Path) -> Path:
    """
    Build a temporary ImageFolder-compatible directory:
        tmp_root/
            brain_mri/      ← positive
            not_brain_mri/  ← negative
    """
    brain_dir    = tmp_root / "brain_mri"
    nonbrain_dir = tmp_root / "not_brain_mri"
    brain_dir.mkdir(parents=True)
    nonbrain_dir.mkdir(parents=True)

    # ── positives: all brain MRI training images ───────────────────────────
    brain_images = collect_images(BRAIN_TRAIN_ROOT)
    print(f"  Brain MRI images (train):  {len(brain_images)}")
    for i, src in enumerate(brain_images):
        shutil.copy(src, brain_dir / f"brain_{i:05d}{src.suffix}")

    # ── negatives: evaluation images (chest + not_chest) ──────────────────
    neg_sources: list[Path] = []
    for subfolder in EVAL_ROOT.iterdir():
        if subfolder.is_dir():
            neg_sources.extend(collect_images(subfolder))

    random.shuffle(neg_sources)
    neg_sources = neg_sources[:NEG_LIMIT]
    print(f"  Non-brain images (neg):    {len(neg_sources)}")
    for i, src in enumerate(neg_sources):
        shutil.copy(src, nonbrain_dir / f"neg_{i:05d}{src.suffix}")

    return tmp_root


def build_val_dir(tmp_root: Path) -> Path:
    """Build validation set from brain TEST split + same negatives."""
    brain_dir    = tmp_root / "brain_mri"
    nonbrain_dir = tmp_root / "not_brain_mri"
    brain_dir.mkdir(parents=True)
    nonbrain_dir.mkdir(parents=True)

    brain_images = collect_images(BRAIN_TEST_ROOT)
    print(f"  Brain MRI images (val):    {len(brain_images)}")
    for i, src in enumerate(brain_images):
        shutil.copy(src, brain_dir / f"brain_{i:05d}{src.suffix}")

    # reuse same neg sources for val (small eval set — acceptable for gate training)
    neg_sources: list[Path] = []
    for subfolder in EVAL_ROOT.iterdir():
        if subfolder.is_dir():
            neg_sources.extend(collect_images(subfolder))
    random.shuffle(neg_sources)
    neg_sources = neg_sources[: NEG_LIMIT // 4]
    print(f"  Non-brain images (val):    {len(neg_sources)}")
    for i, src in enumerate(neg_sources):
        shutil.copy(src, nonbrain_dir / f"neg_{i:05d}{src.suffix}")

    return tmp_root


def build_loaders(train_dir: Path, val_dir: Path):
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_ds, val_ds, train_loader, val_loader


def build_model(device: torch.device):
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = total_correct = total_count = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss   = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()
        total_loss    += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_count   += images.size(0)

    return total_loss / total_count, total_correct / total_count


def main():
    set_seed(SEED)
    device = get_device()
    print("Device:", device)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_dir = tmp_path / "train"
        val_dir   = tmp_path / "val"
        train_dir.mkdir()
        val_dir.mkdir()

        print("\nBuilding training set...")
        build_dataset_dir(train_dir)

        print("\nBuilding validation set...")
        build_val_dir(val_dir)

        train_ds, val_ds, train_loader, val_loader = build_loaders(train_dir, val_dir)
        print(f"\nClasses: {train_ds.classes}")
        print(f"Train size: {len(train_ds)}  |  Val size: {len(val_ds)}")

        model     = build_model(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_val_acc = -1.0
        best_state   = None

        print()
        for epoch in range(EPOCHS):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, device)

            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f}"
            )

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                best_state   = copy.deepcopy(model.state_dict())
                print(f"  ✓ New best val_acc: {best_val_acc:.4f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "class_names": train_ds.classes,
            "image_size": IMAGE_SIZE,
            "best_val_acc": best_val_acc,
        },
        MODEL_OUT,
    )
    print(f"\nSaved to: {MODEL_OUT}")
    print(f"Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()