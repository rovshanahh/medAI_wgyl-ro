from pathlib import Path
import random
import shutil

SEED = 42

RAW_ROOT = Path("raw_datasets")
OUT_ROOT = Path("datasets/input_gates/brain_mri")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

TRAIN_POS_LIMIT = 1200
VAL_POS_LIMIT = 300
TEST_POS_LIMIT = 300

TRAIN_NEG_LIMIT = 1200
VAL_NEG_LIMIT = 300
TEST_NEG_LIMIT = 300


def collect_images(root: Path) -> list[Path]:
    if not root.exists():
        return []

    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def reset_output_dirs() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    for split in ["train", "val", "test"]:
        for label in ["brain_mri", "not_brain_mri"]:
            (OUT_ROOT / split / label).mkdir(parents=True, exist_ok=True)


def copy_images(images: list[Path], target_dir: Path, prefix: str, limit: int) -> int:
    selected = images[:limit]

    for index, src in enumerate(selected):
        dst = target_dir / f"{prefix}_{index:05d}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    return len(selected)


def find_brain_roots() -> tuple[list[Path], list[Path]]:
    train_roots = []
    test_roots = []

    for root in RAW_ROOT.rglob("*"):
        if not root.is_dir():
            continue

        root_str = str(root).lower()
        name = root.name.lower()

        if "brain" not in root_str and "tumor" not in root_str:
            continue

        if name == "training":
            train_roots.append(root)

        if name == "testing":
            test_roots.append(root)

    return train_roots, test_roots


def find_chest_roots() -> tuple[list[Path], list[Path], list[Path]]:
    train_roots = []
    val_roots = []
    test_roots = []

    for root in RAW_ROOT.rglob("*"):
        if not root.is_dir():
            continue

        root_str = str(root).lower()
        name = root.name.lower()

        if "chest" not in root_str and "pneumonia" not in root_str:
            continue

        if name == "train":
            train_roots.append(root)

        elif name == "val":
            val_roots.append(root)

        elif name == "test":
            test_roots.append(root)

    return train_roots, val_roots, test_roots


def find_mura_roots() -> tuple[list[Path], list[Path]]:
    train_roots = []
    valid_roots = []

    for root in RAW_ROOT.rglob("*"):
        if not root.is_dir():
            continue

        root_str = str(root).lower()
        name = root.name.lower()

        if "mura" not in root_str:
            continue

        if name == "train":
            train_roots.append(root)

        elif name in {"valid", "val", "validation"}:
            valid_roots.append(root)

    return train_roots, valid_roots


def collect_from_roots(roots: list[Path]) -> list[Path]:
    images = []

    for root in roots:
        images.extend(collect_images(root))

    return images


def main() -> None:
    random.seed(SEED)
    reset_output_dirs()

    brain_train_roots, brain_test_roots = find_brain_roots()
    chest_train_roots, chest_val_roots, chest_test_roots = find_chest_roots()
    mura_train_roots, mura_valid_roots = find_mura_roots()

    brain_train = collect_from_roots(brain_train_roots)
    brain_test = collect_from_roots(brain_test_roots)

    chest_train = collect_from_roots(chest_train_roots)
    chest_val = collect_from_roots(chest_val_roots)
    chest_test = collect_from_roots(chest_test_roots)

    mura_train = collect_from_roots(mura_train_roots)
    mura_valid = collect_from_roots(mura_valid_roots)

    random.shuffle(brain_train)
    random.shuffle(brain_test)
    random.shuffle(chest_train)
    random.shuffle(chest_val)
    random.shuffle(chest_test)
    random.shuffle(mura_train)
    random.shuffle(mura_valid)

    train_pos = brain_train
    val_pos = brain_test[:VAL_POS_LIMIT]
    test_pos = brain_test[VAL_POS_LIMIT : VAL_POS_LIMIT + TEST_POS_LIMIT]

    train_neg = chest_train + mura_train
    val_neg = chest_val + mura_valid[: len(mura_valid) // 2]
    test_neg = chest_test + mura_valid[len(mura_valid) // 2 :]

    random.shuffle(train_neg)
    random.shuffle(val_neg)
    random.shuffle(test_neg)

    copied = {
        "train/brain_mri": copy_images(
            train_pos,
            OUT_ROOT / "train" / "brain_mri",
            "brain_train",
            TRAIN_POS_LIMIT,
        ),
        "val/brain_mri": copy_images(
            val_pos,
            OUT_ROOT / "val" / "brain_mri",
            "brain_val",
            VAL_POS_LIMIT,
        ),
        "test/brain_mri": copy_images(
            test_pos,
            OUT_ROOT / "test" / "brain_mri",
            "brain_test",
            TEST_POS_LIMIT,
        ),
        "train/not_brain_mri": copy_images(
            train_neg,
            OUT_ROOT / "train" / "not_brain_mri",
            "neg_train",
            TRAIN_NEG_LIMIT,
        ),
        "val/not_brain_mri": copy_images(
            val_neg,
            OUT_ROOT / "val" / "not_brain_mri",
            "neg_val",
            VAL_NEG_LIMIT,
        ),
        "test/not_brain_mri": copy_images(
            test_neg,
            OUT_ROOT / "test" / "not_brain_mri",
            "neg_test",
            TEST_NEG_LIMIT,
        ),
    }

    print("\nPrepared Brain MRI input gate dataset:")
    for key, value in copied.items():
        print(f"{key}: {value}")

    if copied["train/brain_mri"] == 0:
        print("\nWARNING: No brain MRI training images found.")

    if copied["train/not_brain_mri"] == 0:
        print("\nWARNING: No negative training images found.")

    print(f"\nSaved under: {OUT_ROOT}")


if __name__ == "__main__":
    main()