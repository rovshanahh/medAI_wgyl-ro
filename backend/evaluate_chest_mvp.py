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
from medalix.registry.model_registry import ModelRegistry
from medalix.registry.model_router import ModelRouter


def get_chest_model_metadata():
    registry = ModelRegistry("reference_data/model_registry.json")
    router = ModelRouter(registry)
    return router.resolve(
        region="chest",
        modality="xray",
        input_shape=(1, 3, 224, 224),
    )


def evaluate_one(image_path: Path, expected_group: str, model_metadata):
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

    model = EnsembleModel(model_metadata=model_metadata)
    inference_result = model.predict(tensor)

    quality_assessor = QualityAssessor()
    quality_result = quality_assessor.assess(tensor)

    ood_detector = OODDetector()
    ood_result = ood_detector.evaluate(tensor, inference_result)

    conformal_router = ConformalRouter()
    detection_result = {
        "region": "chest",
        "modality": "xray",
        "confidence": input_gate_result["confidence"],
        "supported": True,
        "requires_confirmation": False,
    }
    routing_result = conformal_router.decide(detection_result)

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

    print("Loading chest model metadata from registry...")
    model_metadata = get_chest_model_metadata()
    print(f"Model: {model_metadata.model_id} v{model_metadata.version}")

    results = []

    print(f"\nEvaluating chest X-rays from {chest_dir}...")
    for image_path in sorted(chest_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        result = evaluate_one(image_path, "chest_xray", model_metadata)
        results.append(result)
        print(f"  {image_path.name}: {result['final_action']} ({result.get('risk_category', '—')})")

    print(f"\nEvaluating non-chest images from {nonchest_dir}...")
    for image_path in sorted(nonchest_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        result = evaluate_one(image_path, "not_chest_xray", model_metadata)
        results.append(result)
        print(f"  {image_path.name}: {result['final_action']} ({result.get('risk_category', '—')})")

    output_path = Path("evaluation/evaluation_results.json")
    output_path.write_text(json.dumps(results, indent=2))

    chest_total = sum(r["expected_group"] == "chest_xray" for r in results)
    nonchest_total = sum(r["expected_group"] == "not_chest_xray" for r in results)

    chest_accepted = sum(
        r["expected_group"] == "chest_xray" and r["final_action"] != "STOP"
        for r in results
    )
    nonchest_stopped = sum(
        r["expected_group"] == "not_chest_xray" and r["final_action"] == "STOP"
        for r in results
    )

    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Chest images accepted (not STOP):  {chest_accepted}/{chest_total}")
    print(f"Non-chest images stopped:          {nonchest_stopped}/{nonchest_total}")
    if chest_total:
        print(f"Chest acceptance rate:             {chest_accepted/chest_total*100:.1f}%")
    if nonchest_total:
        print(f"Non-chest rejection rate:          {nonchest_stopped/nonchest_total*100:.1f}%")
    print(f"{'='*50}")
    print(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    main()