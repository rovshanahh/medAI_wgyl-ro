from pathlib import Path
import random
import shutil


RAW_ROOT = Path("raw_datasets/aptos2019")

SOURCE_SPLITS = {
    "train": RAW_ROOT / "train_images",
    "val": RAW_ROOT / "val_images",
    "test": RAW_ROOT / "test_images",
}

OUTPUT_ROOT = Path("datasets/routing")
TARGET_CLASS = "retina_fundus"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Keep it balanced with your other routing classes.
# Earlier you had around 1000 train, 250 val, 250 test per class.
LIMITS = {
    "train": 1000,
    "val": 250,
    "test": 250,
}

SEED = 42


def collect_images(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Missing source folder: {folder}")

    return [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def clear_folder(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    for item in folder.iterdir():
        if item.is_file():
            item.unlink()


def copy_split(split: str) -> int:
    source_dir = SOURCE_SPLITS[split]
    output_dir = OUTPUT_ROOT / split / TARGET_CLASS

    images = collect_images(source_dir)
    random.shuffle(images)

    limit = LIMITS[split]
    selected = images[:limit]

    clear_folder(output_dir)

    for index, source_path in enumerate(selected):
        target_name = f"retina_{index:05d}{source_path.suffix.lower()}"
        shutil.copy2(source_path, output_dir / target_name)

    return len(selected)


def main() -> None:
    random.seed(SEED)

    print("Preparing retina_fundus routing dataset from APTOS 2019...\n")

    counts = {}

    for split in ["train", "val", "test"]:
        count = copy_split(split)
        counts[split] = count
        print(f"{split}/retina_fundus: {count}")

    print("\nSaved under:")
    print(OUTPUT_ROOT)

    print("\nExpected next step:")
    print("python3 train_route_detector.py")


if __name__ == "__main__":
    main()