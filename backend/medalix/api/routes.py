from fastapi import APIRouter, File, HTTPException, UploadFile

from medalix.audit.audit_trace_builder import AuditTraceBuilder
from medalix.audit.logger import Logger
from medalix.detection.conformal_router import ConformalRouter
from medalix.explainability.explainability_engine import ExplainabilityEngine
from medalix.ingestion.chest_xray_input_gate import ChestXrayInputGate
from medalix.ingestion.ingestion_validator import IngestionValidator
from medalix.inference.ensemble_model import EnsembleModel
from medalix.ood.ood_detector import OODDetector
from medalix.policy.governed_decision_policy import GovernedDecisionPolicy
from medalix.policy.policy_models import PolicyInput
from medalix.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from medalix.quality.quality_assessor import QualityAssessor
from medalix.retention.data_retention_manager import DataRetentionManager
from medalix.retention.temp_file_manager import TempFileManager

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


def _build_input_gate_result(input_gate_result: dict | None = None) -> dict:
    input_gate_result = input_gate_result or {}
    hard_reject = input_gate_result.get("hard_reject", False)

    return {
        "accepted_for_analysis": not hard_reject,
        "confidence": input_gate_result.get("confidence"),
        "message": (
            "Input accepted by the active workflow."
            if not hard_reject
            else "Input rejected by the active workflow."
        ),
    }


def _build_detection_result(input_gate_result: dict | None = None) -> dict:
    input_gate_result = input_gate_result or {}
    return {
        "region": "chest",
        "modality": "xray",
        "confidence": input_gate_result.get("confidence"),
        "requires_confirmation": False,
    }


def _build_routing_result(raw_routing_result: dict | None = None) -> dict:
    raw_routing_result = raw_routing_result or {}
    accepted_findings_set = raw_routing_result.get("accepted_findings_set", [])

    return {
        "accepted_findings_set": accepted_findings_set,
        "selected_model": "chest_xray_ensemble_v1",
        "set_size": raw_routing_result.get("set_size", len(accepted_findings_set)),
        "requires_confirmation": raw_routing_result.get("requires_confirmation", False),
    }


def _build_policy_result(
    action: str,
    reason: str,
    risk_category: str | None = None,
    warnings: list[str] | None = None,
) -> dict:
    return {
        "action": action,
        "reason": reason,
        "risk_category": risk_category,
        "warnings": warnings or [],
    }


def _build_ood_result(ood_result: dict | None = None) -> dict:
    ood_result = ood_result or {}
    return {
        "tier": ood_result.get("tier"),
        "score": ood_result.get("score"),
        "is_hard_ood": ood_result.get("is_hard_ood", False),
    }


def _build_explainability_result(explainability_result: dict | None = None) -> dict:
    explainability_result = explainability_result or {}
    return {
        "method": explainability_result.get("method"),
        "heatmap_path": explainability_result.get("heatmap_path"),
        "warning": explainability_result.get("warning"),
    }


def _build_quality_result(quality_result: dict | None = None) -> dict:
    quality_result = quality_result or {}
    return {
        "status": quality_result.get("status"),
        "warnings": quality_result.get("warnings", []),
        "requires_reupload": quality_result.get("requires_reupload"),
        "blocking": quality_result.get("blocking", False),
        "reason": quality_result.get("reason"),
    }


def _build_inference_result(inference_result: dict | None = None) -> dict:
    inference_result = inference_result or {}
    return {
        "top_label": inference_result.get("top_label"),
        "top_probability": inference_result.get("top_probability"),
        "positive_findings": inference_result.get("positive_findings", []),
        "epistemic_uncertainty": inference_result.get("epistemic_uncertainty", {}),
    }


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    temp_path = None
    logger = Logger()
    retention_manager = DataRetentionManager()

    try:
        content = await file.read()

        trace_builder = AuditTraceBuilder()
        temp_file_manager = TempFileManager()

        temp_path = temp_file_manager.save_upload(file.filename, content)

        validator = IngestionValidator()
        validator.validate(file.filename, file.content_type, content)

        input_gate = ChestXrayInputGate()
        input_gate_result = input_gate.evaluate(content)

        safe_input_gate_result = _build_input_gate_result(input_gate_result)
        safe_detection_result = _build_detection_result(input_gate_result)

        if input_gate_result.get("hard_reject", False):
            safe_policy_result = _build_policy_result(
                action="STOP",
                reason="Input was rejected by the active workflow before inference.",
                risk_category="HIGH",
                warnings=["Input not accepted for analysis"],
            )

            trace = trace_builder.build(
                filename=file.filename,
                input_gate=safe_input_gate_result,
                quality={},
                ood={},
                routing={},
                policy=safe_policy_result,
                inference_summary={},
            )
            logger.info({"event": "analysis_completed", "trace": trace.__dict__})

            return {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(content),
                "input_gate": safe_input_gate_result,
                "detection": safe_detection_result,
                "routing": {},
                "quality": {},
                "ood": {},
                "policy": safe_policy_result,
                "disclaimer": NON_DIAGNOSTIC_DISCLAIMER,
                "message": "Analysis stopped before inference",
            }

        preprocessor = PreprocessingPipeline()
        original_image, tensor = preprocessor.run(content)

        model = EnsembleModel()
        inference_result = model.predict(tensor)
        safe_inference_result = _build_inference_result(inference_result)

        explainability_engine = ExplainabilityEngine(model.model)
        raw_explainability_result = explainability_engine.generate(
            original_image=original_image,
            tensor=tensor,
            inference_result=inference_result,
            filename=file.filename,
        )
        safe_explainability_result = _build_explainability_result(raw_explainability_result)

        quality_assessor = QualityAssessor()
        raw_quality_result = quality_assessor.assess(tensor)
        safe_quality_result = _build_quality_result(raw_quality_result)

        ood_detector = OODDetector()
        raw_ood_result = ood_detector.evaluate(tensor, inference_result)
        safe_ood_result = _build_ood_result(raw_ood_result)

        conformal_router = ConformalRouter()
        raw_routing_result = conformal_router.decide(inference_result)
        safe_routing_result = _build_routing_result(raw_routing_result)

        policy_input = PolicyInput(
            ood_result=safe_ood_result,
            routing_result=safe_routing_result,
            inference_result=inference_result,
            quality_result=safe_quality_result,
        )

        policy = GovernedDecisionPolicy()
        policy_output = policy.evaluate(policy_input)

        safe_policy_result = _build_policy_result(
            action=policy_output.action,
            reason=policy_output.reason,
            risk_category=policy_output.risk_category,
            warnings=policy_output.warnings,
        )

        trace = trace_builder.build(
            filename=file.filename,
            input_gate=safe_input_gate_result,
            quality=safe_quality_result,
            ood=safe_ood_result,
            routing=safe_routing_result,
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
                "detection": safe_detection_result,
                "routing": safe_routing_result,
                "quality": safe_quality_result,
                "ood": safe_ood_result,
                "policy": safe_policy_result,
                "disclaimer": NON_DIAGNOSTIC_DISCLAIMER,
                "message": "Analysis stopped due to policy or distribution checks",
            }

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(content),
            "input_gate": safe_input_gate_result,
            "detection": safe_detection_result,
            "tensor_shape": list(tensor.shape),
            "inference": safe_inference_result,
            "explainability": safe_explainability_result,
            "quality": safe_quality_result,
            "ood": safe_ood_result,
            "routing": safe_routing_result,
            "policy": safe_policy_result,
            "disclaimer": NON_DIAGNOSTIC_DISCLAIMER,
            "message": "Governed pipeline completed successfully",
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}") from e
    finally:
        if temp_path is not None:
            retention_result = retention_manager.delete_now(temp_path)
            logger.info({"event": "retention_cleanup", "result": retention_result})