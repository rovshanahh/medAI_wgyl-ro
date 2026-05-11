from pathlib import Path

from PIL import Image, ImageFilter
import numpy as np

from medalix.api.orchestrator import Orchestrator


GENERATED_DIR = Path("test_samples/generated_quality")
SOURCE_IMAGE = Path("test_samples/chest_xray.jpg")


def create_quality_samples() -> dict[str, Path]:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE_IMAGE.exists():
        raise RuntimeError(f"Missing source image: {SOURCE_IMAGE}")

    image = Image.open(SOURCE_IMAGE).convert("RGB")

    blurred_path = GENERATED_DIR / "blurred_chest_xray.jpg"
    noisy_path = GENERATED_DIR / "noisy_chest_xray.jpg"
    low_contrast_path = GENERATED_DIR / "low_contrast_chest_xray.jpg"
    corrupted_path = GENERATED_DIR / "corrupted_file.jpg"

    image.filter(ImageFilter.GaussianBlur(radius=8)).save(blurred_path)

    array = np.array(image).astype(np.float32)
    noise = np.random.normal(0, 90, array.shape)
    noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
    Image.fromarray(noisy).save(noisy_path)

    low_contrast = np.clip(array * 0.08 + 115, 0, 255).astype(np.uint8)
    Image.fromarray(low_contrast).save(low_contrast_path)

    corrupted_path.write_bytes(b"this is not a real image file")

    return {
        "blurred": blurred_path,
        "noisy": noisy_path,
        "low_contrast": low_contrast_path,
        "corrupted": corrupted_path,
    }


def safe_get(data, *keys, default=None):
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)

    return current if current is not None else default


def run_case(orchestrator: Orchestrator, name: str, path: Path) -> dict:
    result = orchestrator.execute(
        filename=path.name,
        content_type=None,
        content=path.read_bytes(),
    )

    route = safe_get(result, "input_gate", "selected_route", default="—")
    policy = safe_get(result, "policy", "action", default="—")
    message = result.get("message", "—")

    quality_status = safe_get(result, "quality", "status", default=None)
    quality_reason = safe_get(result, "quality", "reason", default=None)
    requires_reupload = safe_get(result, "quality", "requires_reupload", default=False)
    blocking = safe_get(result, "quality", "blocking", default=False)

    if quality_status is None and route == "unknown":
        quality_status = "stopped_before_quality"
        quality_reason = safe_get(
            result,
            "policy",
            "reason",
            default="Input was stopped before quality assessment.",
        )

    if quality_status is None:
        quality_status = "—"

    if quality_reason is None:
        quality_reason = "—"

    expected_behavior_passed = False

    if name == "corrupted":
        expected_behavior_passed = policy == "STOP"

    elif name == "low_contrast":
        expected_behavior_passed = (
            quality_status in {"warning", "failed", "stopped_before_quality"}
            or policy == "STOP"
        )

    elif name in {"blurred", "noisy"}:
        expected_behavior_passed = quality_status in {"warning", "failed"} or policy == "STOP"

    return {
        "case": name,
        "file": str(path),
        "route": route,
        "quality_status": quality_status,
        "quality_reason": quality_reason,
        "requires_reupload": requires_reupload,
        "blocking": blocking,
        "policy": policy,
        "message": message,
        "passed": expected_behavior_passed,
    }


def print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 130)
    print("QUALITY CONTROL EVALUATION")
    print("=" * 130)
    print(
        f"{'Case':16} | {'Route':14} | {'Quality':24} | {'Policy':10} | "
        f"{'Reupload':8} | {'Blocking':8} | {'Pass':5} | Reason"
    )
    print("-" * 130)

    for row in rows:
        print(
            f"{row['case'][:16]:16} | "
            f"{row['route'][:14]:14} | "
            f"{row['quality_status'][:24]:24} | "
            f"{row['policy'][:10]:10} | "
            f"{str(row['requires_reupload'])[:8]:8} | "
            f"{str(row['blocking'])[:8]:8} | "
            f"{'Yes' if row['passed'] else 'No':5} | "
            f"{row['quality_reason']}"
        )

    print("=" * 130)


def main() -> None:
    samples = create_quality_samples()
    orchestrator = Orchestrator()

    rows = [run_case(orchestrator, name, path) for name, path in samples.items()]

    print_table(rows)

    failed = [row for row in rows if not row["passed"]]

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()