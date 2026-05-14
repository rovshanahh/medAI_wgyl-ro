from pathlib import Path
import csv
import json
from datetime import datetime

from medalix.api.orchestrator import Orchestrator


VALID_CASES = [
    ("brain_mri", "test_samples/brain_mri.jpg"),
    ("abdomen_ct", "test_samples/abdomen_ct.jpg"),
    ("skin_dermoscopy", "test_samples/skin_dermoscopy.jpg"),
    ("retina_fundus", "test_samples/retina_fundus.jpg"),
    ("chest_xray", "test_samples/chest_xray.jpg"),
    ("breast_mammography", "test_samples/breast_mammography.jpg"),
]

OOD_CASES = [
    ("unknown", "test_samples/random.jpg"),
]

for image_path in sorted(Path("test_samples/ood_synthetic").glob("*.jpg")):
    OOD_CASES.append(("unknown", str(image_path)))


OUT_DIR = Path("evaluation/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_case(orch, expected_group, route, file_path):
    path = Path(file_path)

    if not path.exists():
        return {
            "expected_group": expected_group,
            "route": route,
            "file": file_path,
            "exists": False,
            "error": "missing file",
        }

    kwargs = {
        "filename": path.name,
        "content_type": "image/jpeg",
        "content": path.read_bytes(),
    }

    if route != "unknown":
        kwargs["route_override"] = route

    result = orch.execute(**kwargs)

    ood = result.get("ood", {}) or {}
    metrics = ood.get("metrics", {}) or {}
    policy = result.get("policy", {}) or {}
    pre_safety = result.get("pre_inference_safety", {}) or {}

    tier = ood.get("tier")
    action = policy.get("action")
    inference_exists = result.get("inference") is not None

    if expected_group == "valid":
        correct = inference_exists and tier in {"IN_DISTRIBUTION", None} and action != "STOP"
    else:
        correct = (not inference_exists) or action in {"STOP", "REFUSE"} or tier in {"HARD_OOD", "NEAR_OOD"}

    return {
        "expected_group": expected_group,
        "route": route,
        "file": file_path,
        "exists": True,
        "correct": correct,
        "message": result.get("message"),
        "policy_action": action,
        "policy_risk": policy.get("risk_category"),
        "policy_reason": policy.get("reason"),
        "pre_safety_decision": pre_safety.get("decision"),
        "pre_safety_passed": pre_safety.get("passed"),
        "ood_tier": tier,
        "ood_score": ood.get("score"),
        "diffusion_method": metrics.get("diffusion_method"),
        "diffusion_model_available": metrics.get("diffusion_model_available"),
        "diffusion_near_threshold": metrics.get("diffusion_near_threshold"),
        "diffusion_hard_threshold": metrics.get("diffusion_hard_threshold"),
        "inference_exists": inference_exists,
    }


def main():
    orch = Orchestrator()
    rows = []

    for route, path in VALID_CASES:
        rows.append(run_case(orch, "valid", route, path))

    for route, path in OOD_CASES:
        rows.append(run_case(orch, "ood", route, path))

    valid_rows = [r for r in rows if r["expected_group"] == "valid" and r.get("exists")]
    ood_rows = [r for r in rows if r["expected_group"] == "ood" and r.get("exists")]

    valid_correct = sum(1 for r in valid_rows if r.get("correct"))
    ood_correct = sum(1 for r in ood_rows if r.get("correct"))

    summary = {
        "valid_acceptance_rate": valid_correct / len(valid_rows) if valid_rows else None,
        "ood_rejection_rate": ood_correct / len(ood_rows) if ood_rows else None,
        "valid_cases": len(valid_rows),
        "ood_cases": len(ood_rows),
        "rows": rows,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUT_DIR / f"ood_evaluation_{timestamp}.json"
    csv_path = OUT_DIR / f"ood_evaluation_{timestamp}.csv"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Valid acceptance rate:", summary["valid_acceptance_rate"])
    print("OOD rejection rate:", summary["ood_rejection_rate"])
    print("Saved:", json_path)
    print("Saved:", csv_path)

    for row in rows:
        print(
            row["expected_group"],
            row["route"],
            "correct=", row.get("correct"),
            "policy=", row.get("policy_action"),
            "ood=", row.get("ood_tier"),
            "score=", row.get("ood_score"),
        )


if __name__ == "__main__":
    main()
