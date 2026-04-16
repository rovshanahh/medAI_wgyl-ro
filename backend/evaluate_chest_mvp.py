from pathlib import Path
import json

from medalix.ingestion.chest_xray_input_gate import ChestXrayInputGate
from medalix.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from medalix.inference.ensemble_model import EnsembleModel
from medalix.quality.quality_assessor import QualityAssessor
from medalix.ood.ood_detector import OODDetector
from medalix.detection.conformal_router import ConformalRouter
from medalix.policy.governed_decision_policy import GovernedDecisionPolicy
from medalix.policy.policy_models import PolicyInput


def evaluate_one(image_path: Path, expected_group: str):
    raw_bytes = image_path.read_bytes()

    input_gate = ChestXrayInputGate()
    input_gate_result = input_gate.evaluate(raw_bytes)

    if input_gate_result["hard_reject"]:
        return {
            "filename": image_path.name,
            "expected_group": expected_group,
            "stage": "input_gate",
            "final_action": "STOP",
            "input_gate_label": input_gate_result["predicted_label"],
            "input_gate_confidence": input_gate_result["confidence"],
            "ood_tier": None,
            "risk_category": None,
        }

    preprocessor = PreprocessingPipeline()
    _, tensor = preprocessor.run(raw_bytes)

    model = EnsembleModel()
    inference_result = model.predict(tensor)

    quality_assessor = QualityAssessor()
    quality_result = quality_assessor.assess(tensor)

    ood_detector = OODDetector()
    ood_result = ood_detector.evaluate(tensor, inference_result)

    conformal_router = ConformalRouter()
    routing_result = conformal_router.decide(inference_result)

    policy_input = PolicyInput(
        ood_result=ood_result,
        routing_result=routing_result,
        inference_result=inference_result,
        quality_result=quality_result,
    )

    policy = GovernedDecisionPolicy()
    policy_output = policy.evaluate(policy_input)

    return {
        "filename": image_path.name,
        "expected_group": expected_group,
        "stage": "full_pipeline",
        "final_action": policy_output.action,
        "input_gate_label": input_gate_result["predicted_label"],
        "input_gate_confidence": input_gate_result["confidence"],
        "ood_tier": ood_result.get("tier"),
        "risk_category": policy_output.risk_category,
    }


def main():
    eval_root = Path("evaluation")
    chest_dir = eval_root / "chest_xray"
    nonchest_dir = eval_root / "not_chest_xray"

    results = []

    for image_path in sorted(chest_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        results.append(evaluate_one(image_path, "chest_xray"))

    for image_path in sorted(nonchest_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        results.append(evaluate_one(image_path, "not_chest_xray"))

    output_path = Path("evaluation/evaluation_results.json")
    output_path.write_text(json.dumps(results, indent=2))

    chest_total = sum(r["expected_group"] == "chest_xray" for r in results)
    nonchest_total = sum(r["expected_group"] == "not_chest_xray" for r in results)

    chest_answer_like = sum(
        r["expected_group"] == "chest_xray" and r["final_action"] != "STOP"
        for r in results
    )
    nonchest_stopped = sum(
        r["expected_group"] == "not_chest_xray" and r["final_action"] == "STOP"
        for r in results
    )

    print(f"Chest images accepted (not STOP): {chest_answer_like}/{chest_total}")
    print(f"Non-chest images stopped: {nonchest_stopped}/{nonchest_total}")
    print(f"Saved detailed results to {output_path}")


if __name__ == "__main__":
    main()