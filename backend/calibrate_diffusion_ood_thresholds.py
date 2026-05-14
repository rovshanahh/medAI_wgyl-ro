from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


MODEL_PATH = Path("reference_data/ood/diffusion_ood_model.pth")
OUT_PATH = Path("reference_data/ood/route_diffusion_thresholds.json")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = 128
MAX_PER_ROUTE = 120
SEED = 42

ROUTE_ROOTS = {
    "retina_fundus": [
        Path("raw_datasets/aptos2019/test_images"),
        Path("raw_datasets/aptos2019/val_images"),
    ],
    "skin_dermoscopy": [
        Path("raw_datasets/ham10000/HAM10000_images_part_1"),
        Path("raw_datasets/ham10000/HAM10000_images_part_2"),
    ],
    "brain_mri": [
        Path("raw_datasets/brain-tumor-mri-dataset/Testing"),
    ],
    "bone_xray": [
        Path("raw_datasets/MURA-v1.1/valid"),
    ],
    "chest_xray": [
        Path("raw_datasets/chest_xray/test"),
        Path("raw_datasets/chest_xray/val"),
    ],
    "abdomen_ct": [
        Path("datasets/abdomen_ct/test"),
        Path("raw_datasets/abdomen_ct"),
    ],
    "breast_mammography": [
        Path("raw_datasets/cbis_ddsm/jpeg"),
    ],
}


class SmallDenoisingUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x, noise_level):
        b, _, h, w = x.shape
        t_map = noise_level.view(b, 1, 1, 1).expand(b, 1, h, w)
        x = torch.cat([x, t_map], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d2 = self.dec2(e3)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)

        return torch.sigmoid(self.out(d1))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_paths(roots):
    paths = []

    for root in roots:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(path)

    random.shuffle(paths)
    return paths[:MAX_PER_ROUTE]


def load_image(path):
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


def add_noise(images, step, device):
    noise_level = torch.full(
        (images.size(0),),
        0.05 + (float(step) / 4.0) * 0.20,
        dtype=torch.float32,
        device=device,
    )

    noise = torch.randn_like(images) * noise_level.view(-1, 1, 1, 1)
    noisy = torch.clamp(images + noise, 0.0, 1.0)
    return noisy, noise_level


@torch.no_grad()
def score_image(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    errors = []

    for step in range(5):
        noisy, noise_level = add_noise(image_tensor, step, device)
        reconstructed = model(noisy, noise_level)
        error = torch.mean(torch.abs(image_tensor - reconstructed)).item()
        errors.append(float(error))

    return float(sum(errors) / len(errors))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = SmallDenoisingUNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    output = {
        "method": "route_calibrated_trained_5_step_denoising_diffusion_ood",
        "model_path": str(MODEL_PATH),
        "max_per_route": MAX_PER_ROUTE,
        "routes": {},
    }

    for route, roots in ROUTE_ROOTS.items():
        paths = collect_paths(roots)

        scores = []
        failed = 0

        for path in paths:
            try:
                tensor = load_image(path)
                scores.append(score_image(model, tensor, device))
            except Exception:
                failed += 1

        if len(scores) < 10:
            print(f"{route}: skipped, only {len(scores)} valid images")
            continue

        near = float(np.percentile(scores, 95))
        hard = float(np.percentile(scores, 99))

        output["routes"][route] = {
            "near_threshold": near,
            "hard_threshold": hard,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "p90": float(np.percentile(scores, 90)),
            "p95": near,
            "p99": hard,
            "valid_samples": len(scores),
            "failed_samples": failed,
        }

        print(
            f"{route}: n={len(scores)} "
            f"mean={np.mean(scores):.4f} "
            f"p95={near:.4f} "
            f"p99={hard:.4f} "
            f"max={np.max(scores):.4f}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nSaved:", OUT_PATH)


if __name__ == "__main__":
    main()
