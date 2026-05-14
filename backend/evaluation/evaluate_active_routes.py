from pathlib import Path
import csv
import json
from datetime import datetime

from medalix.api.orchestrator import Orchestrator


TEST_CASES = [
    {
        "route": "skin_dermoscopy",
        "file": "test_samples/skin_dermoscopy.jpg",
        "content_type": "image/jpeg",
    },
    {
        "route": "retina_fundus",
        "file": "test_samples/retina_fundus.jpg",
        "content_type": "image/jpeg",
    },
    {
        "route": "brain_mri",
        "file": "test_samples/brain_mri.jpg",
        "content_type": "image/jpeg",
    },
    {
        "route": "abdomen_ct",
        "file": "test_samples/abdomen_ct.jpg",
        "content_type": "image/jpeg",
    },
    {
        "route": "breast_mammography",
        "file": "test_samples/breast_mammography.jpg",
        "content_type": "image/jpeg",
    },
    {
        "route": "chest_xray",
        "file": "test_samples/chest_xray.jpg",
        "content_type": "image/jpeg",
    },
]


OUT_DIR = Path("evaluation/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_get(data: dict, *keys, default=None):
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)

    return current if current is not None else default


def evaluate_case(orch: Orchestrator, case: dict) -> dict:
    path = Path(case["file"])

    if not path.exists():
        return {
            "route": case["route"],
            "file": case["file"],
            "exists": False,
            "error": "File not found",
        }

    result = orch.execute(
        filename=path.name,
        content_type=case["content_type"],
        content=path.read_bytes(),
        route_override=case["route"],
    )

    inference = result.get("inference", {})
    policy = result.get("policy", {})
    detection = result.get("detection", {})
    input_gate = result.get("input_gate", {})
    ood = result.get("ood", {})
    quality = result.get("quality", {})
    pre_safety = result.get("pre_inference_safety", {})

    return {
        "route": case["route"],
        "file": case["file"],
        "exists": True,
        "analysis_id": result.get("analysis_id"),
        "message": result.get("message"),

        "input_accepted": input_gate.get("accepted_for_analysis"),
        "detected_region": detection.get("region"),
        "detected_modality": detection.get("modality"),
        "detection_supported": detection.get("supported"),

        "top_label": inference.get("top_label"),
        "top_probability": inference.get("top_probability"),
        "deep_ensemble_enabled": inference.get("deep_ensemble_enabled"),
        "ensemble_member_count": inference.get("ensemble_member_count"),
        "uncertainty_method": inference.get("uncertainty_method"),
        "reliability_score": inference.get("reliability_score"),
        "disagreement_score": inference.get("disagreement_score"),
        "uncertainty_level": safe_get(inference, "uncertainty_summary", "level"),
        "secondary_verification_triggered": inference.get("secondary_verification_triggered"),
        "secondary_verification_reason": inference.get("secondary_verification_reason"),

        "pre_safety_passed": pre_safety.get("passed"),
        "pre_safety_decision": pre_safety.get("decision"),
        "pre_safety_risk": pre_safety.get("risk"),
        "pre_safety_reason": pre_safety.get("reason"),
        "pre_safety_reasons": "; ".join(pre_safety.get("reasons", [])),
        "ood_tier": ood.get("tier"),
        "ood_score": ood.get("score"),
        "quality_status": quality.get("status"),
        "quality_blocking": quality.get("blocking"),

        "policy_action": policy.get("action"),
        "policy_risk": policy.get("risk_category"),
        "policy_reason": policy.get("reason"),
        "warnings": "; ".join(result.get("warnings", [])),
    }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUT_DIR / f"active_route_evaluation_{timestamp}.json"
    csv_path = OUT_DIR / f"active_route_evaluation_{timestamp}.csv"

    orch = Orchestrator()
    rows = []

    for case in TEST_CASES:
        print(f"\nEvaluating {case['route']} → {case['file']}")
        row = evaluate_case(orch, case)
        rows.append(row)

        print("  policy:", row.get("policy_action"))
        print("  label:", row.get("top_label"))
        print("  probability:", row.get("top_probability"))
        print("  ensemble:", row.get("deep_ensemble_enabled"), row.get("ensemble_member_count"))

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()