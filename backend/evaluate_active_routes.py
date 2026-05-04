import json
from pathlib import Path

from medalix.api.orchestrator import Orchestrator


TEST_FILES = {
    "brain_mri": "test_samples/brain_mri.jpg",
    "bone_xray": "test_samples/bone_xray.png",
    "chest_xray": "test_samples/chest_xray.jpg",
    "retina_fundus": "test_samples/retina_fundus.jpg",
    "skin_dermoscopy": "test_samples/skin_dermoscopy.jpg",
    "unknown": "test_samples/random.jpg",
    "dicom_chest_xray": "test_samples/test_chest_xray.dcm",
}

OUTPUT_PATH = Path("evaluation/active_route_evaluation.json")


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
            "message": "Test file missing",
            "stage_count": 0,
            "timing_count": 0,
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
    message = result.get("message", "—")

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
        "raw_confidence": confidence,
        "heatmap": "Yes" if heatmap else "No",
        "heatmap_path": heatmap,
        "message": message,
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


def build_summary(rows: list[dict]) -> dict:
    total = len(rows)
    completed = sum(1 for row in rows if row["status"] == "OK")
    heatmaps = sum(1 for row in rows if row["heatmap"] == "Yes")
    stopped = sum(1 for row in rows if row["policy"] == "STOP")
    answered = sum(1 for row in rows if row["policy"] == "ANSWER")

    return {
        "total_cases": total,
        "completed_cases": completed,
        "answer_decisions": answered,
        "stop_decisions": stopped,
        "heatmaps_generated": heatmaps,
        "expected_behavior": {
            "active_medical_routes": "Should produce inference and heatmap",
            "unknown_random_input": "Should STOP before inference",
            "dicom_input": "Should convert, route, infer, and generate heatmap",
        },
        "rows": rows,
    }


def print_checks(summary: dict) -> None:
    print("\nChecks:")
    print(f"- Completed cases: {summary['completed_cases']}/{summary['total_cases']}")
    print(f"- ANSWER decisions: {summary['answer_decisions']}")
    print(f"- STOP decisions: {summary['stop_decisions']}")
    print(f"- Heatmaps generated: {summary['heatmaps_generated']}")
    print("- Unknown/random input should STOP before inference.")
    print("- Active medical routes should produce inference and heatmap.")


def save_report(summary: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation report to: {OUTPUT_PATH}")


def main() -> None:
    orchestrator = Orchestrator()
    rows = []

    for expected_route, file_path in TEST_FILES.items():
        row = run_case(orchestrator, expected_route, file_path)
        rows.append(row)

    summary = build_summary(rows)

    print_table(rows)
    print_checks(summary)
    save_report(summary)


if __name__ == "__main__":
    main()