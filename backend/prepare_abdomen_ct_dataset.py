import random
import shutil
from pathlib import Path
from PIL import Image


RAW_ROOT = Path(
    "raw_datasets/abdomen_ct/"
    "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/"
    "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
)

OUTPUT_ROOT = Path("datasets/abdomen_ct")

CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MAX_IMAGES_PER_CLASS = 1200
SEED = 42


def reset_output_dir() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    for split in ["train", "val", "test"]:
        for class_name in CLASS_NAMES:
            (OUTPUT_ROOT / split / class_name).mkdir(parents=True, exist_ok=True)


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def collect_images(class_name: str) -> list[Path]:
    class_dir = RAW_ROOT / class_name

    if not class_dir.exists():
        raise RuntimeError(f"Missing class folder: {class_dir}")

    images = [
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    images = [path for path in images if is_valid_image(path)]

    random.shuffle(images)

    if MAX_IMAGES_PER_CLASS:
        images = images[:MAX_IMAGES_PER_CLASS]

    return images


def split_images(images: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train = images[:train_end]
    val = images[train_end:val_end]
    test = images[val_end:]

    return train, val, test


def copy_images(images: list[Path], split: str, class_name: str) -> None:
    output_dir = OUTPUT_ROOT / split / class_name

    for index, source_path in enumerate(images, start=1):
        safe_class_name = class_name.lower().replace(" ", "_")
        target_path = output_dir / f"{safe_class_name}_{index:05d}{source_path.suffix.lower()}"
        shutil.copy2(source_path, target_path)


def main() -> None:
    random.seed(SEED)

    if not RAW_ROOT.exists():
        raise RuntimeError(f"Raw abdomen CT dataset not found: {RAW_ROOT}")

    reset_output_dir()

    print("Preparing abdomen CT dataset")
    print(f"Raw root: {RAW_ROOT}")
    print(f"Output:   {OUTPUT_ROOT}")
    print()

    total_counts = {}

    for class_name in CLASS_NAMES:
        images = collect_images(class_name)
        train, val, test = split_images(images)

        copy_images(train, "train", class_name)
        copy_images(val, "val", class_name)
        copy_images(test, "test", class_name)

        total_counts[class_name] = {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "total": len(images),
        }

    print("Prepared counts:")
    for class_name, counts in total_counts.items():
        print(
            f"{class_name:8} "
            f"train={counts['train']:4} "
            f"val={counts['val']:4} "
            f"test={counts['test']:4} "
            f"total={counts['total']:4}"
        )

    print()
    print("Saved under:")
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()