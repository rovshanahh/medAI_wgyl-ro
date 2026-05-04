from pathlib import Path
import random
import shutil


RAW_ROOT = Path("raw_datasets/ham10000")

SOURCE_DIRS = [
    RAW_ROOT / "HAM10000_images_part_1",
    RAW_ROOT / "HAM10000_images_part_2",
]

OUTPUT_ROOT = Path("datasets/routing")
TARGET_CLASS = "skin_dermoscopy"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

LIMITS = {
    "train": 1000,
    "val": 250,
    "test": 250,
}

SEED = 42


def collect_images() -> list[Path]:
    images = []

    for folder in SOURCE_DIRS:
        if not folder.exists():
            raise FileNotFoundError(f"Missing source folder: {folder}")

        images.extend(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    if not images:
        raise RuntimeError("No HAM10000 images found.")

    return images


def clear_folder(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    for item in folder.iterdir():
        if item.is_file():
            item.unlink()


def copy_images(images: list[Path], split: str) -> int:
    output_dir = OUTPUT_ROOT / split / TARGET_CLASS
    clear_folder(output_dir)

    limit = LIMITS[split]
    selected = images[:limit]

    for index, source_path in enumerate(selected):
        target_name = f"skin_{index:05d}{source_path.suffix.lower()}"
        shutil.copy2(source_path, output_dir / target_name)

    del images[:limit]
    return len(selected)


def main() -> None:
    random.seed(SEED)

    images = collect_images()
    random.shuffle(images)

    total_needed = sum(LIMITS.values())

    if len(images) < total_needed:
        raise RuntimeError(
            f"Not enough HAM10000 images. Need {total_needed}, found {len(images)}."
        )

    print("Preparing skin_dermoscopy routing dataset from HAM10000...\n")

    counts = {}

    for split in ["train", "val", "test"]:
        count = copy_images(images, split)
        counts[split] = count
        print(f"{split}/skin_dermoscopy: {count}")

    print("\nSaved under:")
    print(OUTPUT_ROOT)

    print("\nExpected next step:")
    print("Update train_route_detector.py to include skin_dermoscopy")
    print("Then run: python3 train_route_detector.py")


if __name__ == "__main__":
    main()