from fastapi import APIRouter, File, HTTPException, UploadFile

from medalix.ingestion.ingestion_validator import IngestionValidator
from medalix.ingestion.chest_xray_input_gate import ChestXrayInputGate
from medalix.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from medalix.inference.ensemble_model import EnsembleModel
from medalix.quality.quality_assessor import QualityAssessor
from medalix.ood.ood_detector import OODDetector
from medalix.detection.conformal_router import ConformalRouter
from medalix.policy.governed_decision_policy import GovernedDecisionPolicy
from medalix.policy.policy_models import PolicyInput
from medalix.audit.logger import Logger
from medalix.audit.audit_trace_builder import AuditTraceBuilder
from medalix.explainability.explainability_engine import ExplainabilityEngine

router = APIRouter()

NON_DIAGNOSTIC_DISCLAIMER = (
    "Research-use only. This output is non-diagnostic and must not be used "
    "for clinical decision-making."
)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/")
def root():
    return {"message": "MedAIx backend is running"}


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        content = await file.read()

        logger = Logger()
        trace_builder = AuditTraceBuilder()

        validator = IngestionValidator()
        validator.validate(file.filename, file.content_type, content)

        input_gate = ChestXrayInputGate()
        input_gate_result = input_gate.evaluate(content)

        safe_input_gate_result = {
            "is_chest_xray_like": input_gate_result.get("is_chest_xray_like"),
            "predicted_label": input_gate_result.get("predicted_label"),
            "confidence": input_gate_result.get("confidence"),
            "hard_reject": input_gate_result.get("hard_reject"),
        }

        if input_gate_result["hard_reject"]:
            policy_result = {
                "action": "STOP",
                "reason": "Input is not a chest X-ray-like image",
            }

            trace = trace_builder.build(
                filename=file.filename,
                input_gate=safe_input_gate_result,
                quality={},
                ood={},
                routing={},
                policy=policy_result,
                inference_summary={},
            )
            logger.info({"event": "analysis_completed", "trace": trace.__dict__})

            return {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(content),
                "input_gate": safe_input_gate_result,
                "policy": policy_result,
                "disclaimer": NON_DIAGNOSTIC_DISCLAIMER,
                "message": "Analysis stopped before inference",
            }

        preprocessor = PreprocessingPipeline()
        original_image, tensor = preprocessor.run(content)

        model = EnsembleModel()
        inference_result = model.predict(tensor)

        explainability_engine = ExplainabilityEngine(model.model)
        explainability_result = explainability_engine.generate(
            original_image=original_image,
            tensor=tensor,
            inference_result=inference_result,
            filename=file.filename,
        )

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

        safe_inference_result = {
            "top_label": inference_result.get("top_label"),
            "top_probability": inference_result.get("top_probability"),
            "positive_findings": inference_result.get("positive_findings", []),
            "epistemic_uncertainty": inference_result.get("epistemic_uncertainty", {}),
        }

        safe_policy_result = {
            "action": policy_output.action,
            "reason": policy_output.reason,
            "risk_category": policy_output.risk_category,
            "warnings": policy_output.warnings,
        }

        trace = trace_builder.build(
            filename=file.filename,
            input_gate=safe_input_gate_result,
            quality=quality_result,
            ood=ood_result,
            routing=routing_result,
            policy=safe_policy_result,
            inference_summary=safe_inference_result,
        )
        logger.info({"event": "analysis_completed", "trace": trace.__dict__})

        if policy_output.action == "STOP":
            return {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(content),
                "input_gate": safe_input_gate_result,
                "ood": ood_result,
                "policy": safe_policy_result,
                "disclaimer": NON_DIAGNOSTIC_DISCLAIMER,
                "message": "Analysis stopped due to out-of-distribution input",
            }

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(content),
            "input_gate": safe_input_gate_result,
            "tensor_shape": list(tensor.shape),
            "inference": safe_inference_result,
            "explainability": explainability_result,
            "quality": quality_result,
            "ood": ood_result,
            "routing": routing_result,
            "policy": safe_policy_result,
            "disclaimer": NON_DIAGNOSTIC_DISCLAIMER,
            "message": "Governed pipeline completed successfully",
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}") from e