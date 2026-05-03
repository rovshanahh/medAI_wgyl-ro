from pathlib import Path

from medalix.api.orchestrator import Orchestrator


TEST_FILES = [
    "test_samples/brain_mri.png",
    "test_samples/bone_xray.png",
    "test_samples/chest_xray.png",
    "test_samples/random.png",
    "test_samples/test_chest_xray.dcm",
]

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8"


def guess_content_type(path: Path, content: bytes) -> str:
    suffix = path.suffix.lower()

    if suffix == ".dcm":
        return "application/dicom"

    if content.startswith(PNG_MAGIC):
        return "image/png"

    if content.startswith(JPEG_MAGIC):
        return "image/jpeg"

    if content[:4] in {b"II*\x00", b"MM\x00*"}:
        return "image/tiff"

    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"

    if suffix == ".png":
        return "image/png"

    if suffix in {".tif", ".tiff"}:
        return "image/tiff"

    return "application/octet-stream"


def main() -> None:
    orchestrator = Orchestrator()

    for file_name in TEST_FILES:
        path = Path(file_name)

        print("\n" + "=" * 80)
        print("File:", file_name)

        if not path.exists():
            print("Missing file. Skipped.")
            continue

        content = path.read_bytes()

        result = orchestrator.execute(
            filename=path.name,
            content_type=guess_content_type(path, content),
            content=content,
        )

        input_gate = result.get("input_gate", {})
        detection = result.get("detection", {})
        routing = result.get("routing", {})
        inference = result.get("inference", {})
        policy = result.get("policy", {})
        ood = result.get("ood", {})
        explainability = result.get("explainability", {})

        print("Accepted:", input_gate.get("accepted_for_analysis"))
        print("Selected route:", input_gate.get("selected_route"))
        print("Region:", detection.get("region"))
        print("Modality:", detection.get("modality"))
        print("Selected model:", routing.get("selected_model"))
        print("Inference label:", inference.get("top_label", "No inference"))
        print("Inference confidence:", inference.get("top_probability", "—"))
        print("OOD tier:", ood.get("tier"))
        print("Policy action:", policy.get("action"))
        print("Policy reason:", policy.get("reason"))
        print("Heatmap:", explainability.get("heatmap_path"))
        print("Message:", result.get("message"))


if __name__ == "__main__":
    main()