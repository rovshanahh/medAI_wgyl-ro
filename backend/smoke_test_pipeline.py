from pathlib import Path

from medalix.api.orchestrator import Orchestrator


TEST_FILES = [
    "test_samples/abdomen_ct.jpg",
    "test_samples/brain_mri.jpg",
    "test_samples/bone_xray.png",
    "test_samples/chest_xray.jpg",
    "test_samples/random.jpg",
    "test_samples/retina_fundus.jpg",
    "test_samples/skin_dermoscopy.jpg",
    "test_samples/test_chest_xray.dcm",
    "test_samples/breast_mammography.jpg",
]


def value_or_dash(value):
    if value is None:
        return "—"
    return value


def main():
    orchestrator = Orchestrator()

    for file_path in TEST_FILES:
        path = Path(file_path)

        if not path.exists():
            print("\n" + "=" * 80)
            print(f"Missing file: {file_path}")
            continue

        result = orchestrator.execute(
            filename=path.name,
            content_type=None,
            content=path.read_bytes(),
        )

        input_gate = result.get("input_gate", {})
        detection = result.get("detection", {})
        routing = result.get("routing", {})
        inference = result.get("inference", {})
        ood = result.get("ood", {})
        policy = result.get("policy", {})
        explainability = result.get("explainability", {})

        print("\n" + "=" * 80)
        print(f"File: {file_path}")
        print(f"Accepted: {input_gate.get('accepted_for_analysis')}")
        print(f"Selected route: {input_gate.get('selected_route')}")
        print(f"Region: {detection.get('region')}")
        print(f"Modality: {detection.get('modality')}")
        print(f"Selected model: {routing.get('selected_model')}")
        print(f"Inference label: {inference.get('top_label', 'No inference')}")
        print(f"Inference confidence: {value_or_dash(inference.get('top_probability'))}")
        print(f"OOD tier: {value_or_dash(ood.get('tier'))}")
        print(f"Policy action: {policy.get('action')}")
        print(f"Policy reason: {policy.get('reason')}")
        print(f"Heatmap: {explainability.get('heatmap_path')}")
        print(f"Message: {result.get('message')}")


if __name__ == "__main__":
    main()