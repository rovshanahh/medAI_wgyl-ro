from pathlib import Path
from statistics import mean

from medalix.api.orchestrator import Orchestrator


TEST_FILES = {
    "brain_mri": "test_samples/brain_mri.jpg",
    "bone_xray": "test_samples/bone_xray.png",
    "chest_xray": "test_samples/chest_xray.jpg",
    "retina_fundus": "test_samples/retina_fundus.jpg",
    "unknown": "test_samples/random.jpg",
    "dicom_chest_xray": "test_samples/test_chest_xray.dcm",
}


def format_percent(value):
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def safe_get(data, *keys, default=None):
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)

    return current if current is not None else default


def run_case(orchestrator: Orchestrator, expected_route: str, file_path: str) -> dict:
    path = Path(file_path)

    if not path.exists():
        return {
            "case": expected_route,
            "file": file_path,
            "status": "MISSING",
            "route": "—",
            "model": "—",
            "policy": "—",
            "output": "—",
            "confidence": "—",
            "heatmap": "—",
        }

    result = orchestrator.execute(
        filename=path.name,
        content_type=None,
        content=path.read_bytes(),
    )

    route = safe_get(result, "input_gate", "selected_route", default="—")
    model = safe_get(result, "routing", "selected_model", default="—")
    policy = safe_get(result, "policy", "action", default="—")
    output = safe_get(result, "inference", "top_label", default="No inference")
    confidence = safe_get(result, "inference", "top_probability", default=None)
    heatmap = safe_get(result, "explainability", "heatmap_path", default=None)

    stage_count = len(result.get("pipeline_stages", []))
    timing_summary = result.get("timing_summary", [])

    return {
        "case": expected_route,
        "file": file_path,
        "status": "OK",
        "route": route,
        "model": model or "—",
        "policy": policy,
        "output": output,
        "confidence": format_percent(confidence),
        "heatmap": "Yes" if heatmap else "No",
        "stage_count": stage_count,
        "timing_count": len(timing_summary),
    }


def print_table(rows: list[dict]) -> None:
    headers = [
        "Case",
        "Route",
        "Model",
        "Policy",
        "Output",
        "Confidence",
        "Heatmap",
    ]

    table_rows = [
        [
            row["case"],
            row["route"],
            row["model"],
            row["policy"],
            row["output"],
            row["confidence"],
            row["heatmap"],
        ]
        for row in rows
    ]

    widths = [
        max(len(str(value)) for value in [header] + [row[i] for row in table_rows])
        for i, header in enumerate(headers)
    ]

    print("\n" + "=" * 100)
    print("ACTIVE ROUTE EVALUATION SUMMARY")
    print("=" * 100)

    header_line = " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))

    for row in table_rows:
        print(" | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))

    print("=" * 100)


def print_checks(rows: list[dict]) -> None:
    total = len(rows)
    completed = sum(1 for row in rows if row["status"] == "OK")
    heatmaps = sum(1 for row in rows if row["heatmap"] == "Yes")
    stopped = sum(1 for row in rows if row["policy"] == "STOP")
    answered = sum(1 for row in rows if row["policy"] == "ANSWER")

    print("\nChecks:")
    print(f"- Completed cases: {completed}/{total}")
    print(f"- ANSWER decisions: {answered}")
    print(f"- STOP decisions: {stopped}")
    print(f"- Heatmaps generated: {heatmaps}")
    print("- Unknown/random input should STOP before inference.")
    print("- Active medical routes should produce inference and heatmap.")


def main() -> None:
    orchestrator = Orchestrator()
    rows = []

    for expected_route, file_path in TEST_FILES.items():
        row = run_case(orchestrator, expected_route, file_path)
        rows.append(row)

    print_table(rows)
    print_checks(rows)


if __name__ == "__main__":
    main()