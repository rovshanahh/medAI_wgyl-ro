from pathlib import Path
import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


MEDICAL_ROOTS = [
    Path("raw_datasets/chest_xray/train"),
    Path("raw_datasets/brain-tumor-mri-dataset/Training"),
    Path("raw_datasets/ham10000/HAM10000_images_part_1"),
    Path("raw_datasets/ham10000/HAM10000_images_part_2"),
    Path("raw_datasets/aptos2019/train_images"),
    Path("raw_datasets/abdomen_ct"),
    Path("raw_datasets/MURA-v1.1/train"),
    Path("raw_datasets/cbis-ddsm/jpeg"),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

IMAGE_SIZE = 128
BATCH_SIZE = 32
DEFAULT_EPOCHS = 3
DEFAULT_MAX_IMAGES = 4000
LR = 1e-4
SEED = 42

MODEL_OUT = Path("reference_data/ood/diffusion_ood_model.pth")
THRESHOLDS_OUT = Path("reference_data/ood/diffusion_ood_thresholds.json")


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


def collect_images(max_images: int) -> list[Path]:
    paths = []

    for root in MEDICAL_ROOTS:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(path)

    random.shuffle(paths)
    return paths[:max_images]


class MedicalImageDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]

        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))

        return self.transform(image)


class SmallDenoisingUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        t_map = noise_level.view(batch_size, 1, 1, 1).expand(
            batch_size,
            1,
            height,
            width,
        )

        x = torch.cat([x, t_map], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d2 = self.dec2(e3)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)

        return torch.sigmoid(self.out(d1))


def add_noise(images: torch.Tensor, steps: torch.Tensor):
    noise_levels = 0.05 + (steps.float() / 4.0) * 0.20
    noise = torch.randn_like(images) * noise_levels.view(-1, 1, 1, 1)
    noisy = torch.clamp(images + noise, 0.0, 1.0)
    return noisy, noise_levels


@torch.no_grad()
def collect_errors(model, loader, device, max_batches: int = 40):
    model.eval()
    errors = []

    for batch_index, images in enumerate(loader):
        if batch_index >= max_batches:
            break

        images = images.to(device)
        batch_size = images.size(0)

        for step in range(5):
            steps = torch.full(
                (batch_size,),
                step,
                dtype=torch.long,
                device=device,
            )
            noisy, noise_levels = add_noise(images, steps)
            reconstructed = model(noisy, noise_levels)

            batch_errors = torch.mean(
                torch.abs(images - reconstructed),
                dim=(1, 2, 3),
            )

            errors.extend(batch_errors.detach().cpu().tolist())

    return errors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES)
    parser.add_argument("--output", type=str, default=str(MODEL_OUT))
    parser.add_argument("--thresholds", type=str, default=str(THRESHOLDS_OUT))
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)

    device = get_device()
    paths = collect_images(args.max_images)

    if len(paths) < 500:
        raise RuntimeError(f"Not enough medical images found. Found only {len(paths)}.")

    split_index = int(len(paths) * 0.85)
    train_paths = paths[:split_index]
    val_paths = paths[split_index:]

    train_loader = DataLoader(
        MedicalImageDataset(train_paths),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        MedicalImageDataset(val_paths),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = SmallDenoisingUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    print("Device:", device)
    print("Train images:", len(train_paths))
    print("Val images:", len(val_paths))
    print("Epochs:", args.epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_count = 0

        for images in train_loader:
            images = images.to(device)
            batch_size = images.size(0)

            steps = torch.randint(0, 5, (batch_size,), device=device)
            noisy, noise_levels = add_noise(images, steps)

            optimizer.zero_grad()
            reconstructed = model(noisy, noise_levels)
            loss = F.l1_loss(reconstructed, images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_count += batch_size

        print(f"Epoch {epoch}/{args.epochs} | train_l1={total_loss / total_count:.5f}")

    val_errors = collect_errors(model, val_loader, device)

    near_threshold = float(np.percentile(val_errors, 90))
    hard_threshold = float(np.percentile(val_errors, 97))

    output_path = Path(args.output)
    threshold_path = Path(args.thresholds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "small_denoising_unet",
            "image_size": IMAGE_SIZE,
            "diffusion_steps": 5,
            "training_images": len(train_paths),
            "validation_images": len(val_paths),
            "near_threshold": near_threshold,
            "hard_threshold": hard_threshold,
        },
        output_path,
    )

    threshold_path.write_text(
        json.dumps(
            {
                "method": "trained_5_step_denoising_diffusion_ood",
                "model_path": str(output_path),
                "near_threshold": near_threshold,
                "hard_threshold": hard_threshold,
                "validation_error_mean": float(np.mean(val_errors)),
                "validation_error_std": float(np.std(val_errors)),
                "validation_error_p90": near_threshold,
                "validation_error_p97": hard_threshold,
                "diffusion_steps": 5,
                "image_size": IMAGE_SIZE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved model:", output_path)
    print("Saved thresholds:", threshold_path)
    print("Near threshold:", near_threshold)
    print("Hard threshold:", hard_threshold)


if __name__ == "__main__":
    main()
