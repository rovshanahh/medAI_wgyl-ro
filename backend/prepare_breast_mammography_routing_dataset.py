from pathlib import Path
import csv
import random
import shutil


RAW_ROOT = Path("raw_datasets/cbis_ddsm")
CSV_ROOT = RAW_ROOT / "csv"
JPEG_ROOT = RAW_ROOT / "jpeg"

DICOM_INFO_CSV = CSV_ROOT / "dicom_info.csv"

OUTPUT_ROOT = Path("datasets/routing")
TARGET_CLASS = "breast_mammography"

LIMITS = {
    "train": 1000,
    "val": 250,
    "test": 250,
}

SEED = 42


def normalize_path(value: str) -> str:
    return (value or "").strip().replace("\\", "/").replace("\n", "")


def collect_mammography_jpegs() -> list[Path]:
    if not DICOM_INFO_CSV.exists():
        raise FileNotFoundError(f"Missing file: {DICOM_INFO_CSV}")

    images = []

    with DICOM_INFO_CSV.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)

        for row in reader:
            body_part = normalize_path(row.get("BodyPartExamined", "")).upper()
            modality = normalize_path(row.get("Modality", "")).upper()
            series_description = normalize_path(row.get("SeriesDescription", "")).lower()
            image_path = normalize_path(row.get("image_path", ""))

            if body_part != "BREAST":
                continue

            if modality != "MG":
                continue

            # For routing, use full mammogram images, not cropped ROI/mask images.
            if "full mammogram" not in series_description:
                continue

            if not image_path:
                continue

            jpeg_relative = image_path.replace("CBIS-DDSM/jpeg/", "")
            jpeg_path = JPEG_ROOT / jpeg_relative

            if jpeg_path.exists():
                images.append(jpeg_path)

    images = list(dict.fromkeys(images))

    if not images:
        raise RuntimeError("No full mammography JPEG images were found from dicom_info.csv.")

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
        target_name = f"breast_{index:05d}{source_path.suffix.lower()}"
        shutil.copy2(source_path, output_dir / target_name)

    del images[:limit]
    return len(selected)


def main() -> None:
    random.seed(SEED)

    images = collect_mammography_jpegs()
    random.shuffle(images)

    total_needed = sum(LIMITS.values())

    print("CBIS-DDSM mammography routing preparation")
    print(f"Matched full mammography JPEG images: {len(images)}")
    print(f"Needed images: {total_needed}")

    if len(images) < total_needed:
        raise RuntimeError(
            f"Not enough mammography images. Need {total_needed}, found {len(images)}."
        )

    print("\nPreparing breast_mammography routing dataset...\n")

    for split in ["train", "val", "test"]:
        count = copy_images(images, split)
        print(f"{split}/breast_mammography: {count}")

    print("\nSaved under:")
    print(OUTPUT_ROOT)

    print("\nNext:")
    print("Add breast_mammography to EXPECTED_CLASSES in train_route_detector.py")
    print("Then run: python3 train_route_detector.py")


if __name__ == "__main__":
    main()