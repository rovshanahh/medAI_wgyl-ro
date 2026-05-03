from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw


OUT_ROOT = Path("datasets/routing")

TRAIN_LIMIT = 1000
VAL_LIMIT = 250
TEST_LIMIT = 250

IMAGE_SIZE = 224
SEED = 42


def make_dirs() -> None:
    for split in ["train", "val", "test"]:
        (OUT_ROOT / split / "unknown").mkdir(parents=True, exist_ok=True)


def random_noise_image() -> Image.Image:
    arr = np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def random_color_blocks() -> Image.Image:
    img = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        color=(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        ),
    )

    draw = ImageDraw.Draw(img)

    for _ in range(random.randint(5, 20)):
        x1 = random.randint(0, IMAGE_SIZE - 1)
        y1 = random.randint(0, IMAGE_SIZE - 1)
        x2 = random.randint(x1, IMAGE_SIZE)
        y2 = random.randint(y1, IMAGE_SIZE)

        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )

        draw.rectangle([x1, y1, x2, y2], fill=color)

    return img


def random_lines_image() -> Image.Image:
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="white")
    draw = ImageDraw.Draw(img)

    for _ in range(random.randint(20, 80)):
        x1 = random.randint(0, IMAGE_SIZE - 1)
        y1 = random.randint(0, IMAGE_SIZE - 1)
        x2 = random.randint(0, IMAGE_SIZE - 1)
        y2 = random.randint(0, IMAGE_SIZE - 1)

        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )

        draw.line([x1, y1, x2, y2], fill=color, width=random.randint(1, 5))

    return img


def random_fake_screenshot() -> Image.Image:
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)

    for i in range(8):
        y = 15 + i * 24
        draw.rectangle(
            [15, y, random.randint(120, 210), y + 10],
            fill=(
                random.randint(80, 220),
                random.randint(80, 220),
                random.randint(80, 220),
            ),
        )

    for _ in range(8):
        x1 = random.randint(10, 180)
        y1 = random.randint(10, 180)
        x2 = min(x1 + random.randint(20, 60), IMAGE_SIZE - 10)
        y2 = min(y1 + random.randint(15, 40), IMAGE_SIZE - 10)
        draw.rectangle([x1, y1, x2, y2], outline=(80, 80, 80), width=2)

    return img


def generate_unknown_image() -> Image.Image:
    image_type = random.choice(
        [
            "noise",
            "blocks",
            "lines",
            "screenshot",
        ]
    )

    if image_type == "noise":
        return random_noise_image()

    if image_type == "blocks":
        return random_color_blocks()

    if image_type == "lines":
        return random_lines_image()

    return random_fake_screenshot()


def save_split(split: str, count: int) -> None:
    out_dir = OUT_ROOT / split / "unknown"

    for i in range(count):
        img = generate_unknown_image()
        img.save(out_dir / f"unknown_{i:05d}.png")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    make_dirs()

    save_split("train", TRAIN_LIMIT)
    save_split("val", VAL_LIMIT)
    save_split("test", TEST_LIMIT)

    print("Unknown routing images prepared:")
    print(f"train/unknown: {TRAIN_LIMIT}")
    print(f"val/unknown:   {VAL_LIMIT}")
    print(f"test/unknown:  {TEST_LIMIT}")


if __name__ == "__main__":
    main()