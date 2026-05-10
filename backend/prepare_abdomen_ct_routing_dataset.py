import random
import shutil
from pathlib import Path


SOURCE_ROOT = Path("datasets/abdomen_ct")
ROUTING_ROOT = Path("datasets/routing")

ROUTE_NAME = "abdomen_ct"
CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]

TRAIN_COUNT = 1000
VAL_COUNT = 250
TEST_COUNT = 250

SEED = 42


def collect_images() -> list[Path]:
    images = []

    for split in ["train", "val", "test"]:
        for class_name in CLASS_NAMES:
            class_dir = SOURCE_ROOT / split / class_name

            if not class_dir.exists():
                raise RuntimeError(f"Missing folder: {class_dir}")

            images.extend(
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )

    random.shuffle(images)
    return images


def reset_route_folder() -> None:
    for split in ["train", "val", "test"]:
        route_dir = ROUTING_ROOT / split / ROUTE_NAME

        if route_dir.exists():
            shutil.rmtree(route_dir)

        route_dir.mkdir(parents=True, exist_ok=True)


def copy_images(images: list[Path], split: str) -> None:
    output_dir = ROUTING_ROOT / split / ROUTE_NAME

    for index, source_path in enumerate(images, start=1):
        target_path = output_dir / f"{ROUTE_NAME}_{index:05d}{source_path.suffix.lower()}"
        shutil.copy2(source_path, target_path)


def main() -> None:
    random.seed(SEED)

    images = collect_images()
    total_needed = TRAIN_COUNT + VAL_COUNT + TEST_COUNT

    print("Abdomen CT routing preparation")
    print(f"Available images: {len(images)}")
    print(f"Needed images:    {total_needed}")

    if len(images) < total_needed:
        raise RuntimeError(
            f"Not enough abdomen CT images. Need {total_needed}, found {len(images)}."
        )

    selected = images[:total_needed]

    train_images = selected[:TRAIN_COUNT]
    val_images = selected[TRAIN_COUNT : TRAIN_COUNT + VAL_COUNT]
    test_images = selected[TRAIN_COUNT + VAL_COUNT :]

    reset_route_folder()

    copy_images(train_images, "train")
    copy_images(val_images, "val")
    copy_images(test_images, "test")

    print()
    print("Prepared abdomen_ct routing dataset:")
    print(f"train/{ROUTE_NAME}: {len(train_images)}")
    print(f"val/{ROUTE_NAME}:   {len(val_images)}")
    print(f"test/{ROUTE_NAME}:  {len(test_images)}")

    print()
    print("Saved under:")
    print(ROUTING_ROOT)


if __name__ == "__main__":
    main()