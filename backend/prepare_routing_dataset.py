from pathlib import Path
import random
import shutil

SEED = 42

RAW_ROOT = Path("raw_datasets")
OUT_ROOT = Path("datasets/routing")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

TRAIN_LIMIT = 1000
VAL_LIMIT = 250
TEST_LIMIT = 250


def collect_images(root: Path) -> list[Path]:
    if not root.exists():
        return []

    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def clear_class_folder(split: str, class_name: str) -> None:
    target = OUT_ROOT / split / class_name

    if target.exists():
        shutil.rmtree(target)

    target.mkdir(parents=True, exist_ok=True)


def copy_images(images: list[Path], split: str, class_name: str, prefix: str) -> int:
    target = OUT_ROOT / split / class_name

    for index, src in enumerate(images):
        dst = target / f"{prefix}_{index:05d}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    return len(images)


def find_brain_images() -> list[Path]:
    images = []

    for root in RAW_ROOT.rglob("*"):
        if not root.is_dir():
            continue

        root_str = str(root).lower()

        if "brain" in root_str or "tumor" in root_str:
            images.extend(collect_images(root))

    return images


def find_chest_images() -> list[Path]:
    images = []

    for root in RAW_ROOT.rglob("*"):
        if not root.is_dir():
            continue

        root_str = str(root).lower()

        if "chest" in root_str or "pneumonia" in root_str:
            images.extend(collect_images(root))

    return images


def find_bone_images() -> list[Path]:
    images = []

    for root in RAW_ROOT.rglob("*"):
        if not root.is_dir():
            continue

        root_str = str(root).lower()

        if "mura" in root_str:
            images.extend(collect_images(root))

    return images


def split_images(images: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    random.shuffle(images)

    train = images[:TRAIN_LIMIT]
    val = images[TRAIN_LIMIT : TRAIN_LIMIT + VAL_LIMIT]
    test = images[TRAIN_LIMIT + VAL_LIMIT : TRAIN_LIMIT + VAL_LIMIT + TEST_LIMIT]

    return train, val, test


def main() -> None:
    random.seed(SEED)

    for split in ["train", "val", "test"]:
        for class_name in ["brain_mri", "bone_xray", "chest_xray"]:
            clear_class_folder(split, class_name)

    brain_images = find_brain_images()
    chest_images = find_chest_images()
    bone_images = find_bone_images()

    brain_train, brain_val, brain_test = split_images(brain_images)
    chest_train, chest_val, chest_test = split_images(chest_images)
    bone_train, bone_val, bone_test = split_images(bone_images)

    copied = {
        "train/brain_mri": copy_images(brain_train, "train", "brain_mri", "brain_train"),
        "val/brain_mri": copy_images(brain_val, "val", "brain_mri", "brain_val"),
        "test/brain_mri": copy_images(brain_test, "test", "brain_mri", "brain_test"),
        "train/chest_xray": copy_images(chest_train, "train", "chest_xray", "chest_train"),
        "val/chest_xray": copy_images(chest_val, "val", "chest_xray", "chest_val"),
        "test/chest_xray": copy_images(chest_test, "test", "chest_xray", "chest_test"),
        "train/bone_xray": copy_images(bone_train, "train", "bone_xray", "bone_train"),
        "val/bone_xray": copy_images(bone_val, "val", "bone_xray", "bone_val"),
        "test/bone_xray": copy_images(bone_test, "test", "bone_xray", "bone_test"),
    }

    print("\nPrepared routing dataset:")
    for key, value in copied.items():
        print(f"{key}: {value}")

    print("\nUnknown folders were not touched.")
    print("Saved under: datasets/routing")


if __name__ == "__main__":
    main()