from pathlib import Path

from medalix.audit.audit_trace_builder import AuditTraceBuilder
from medalix.audit.logger import Logger
from medalix.explainability.explainability_engine import ExplainabilityEngine
from medalix.ingestion.bone_xray_input_gate import BoneXrayInputGate
from medalix.ingestion.brain_mri_input_gate import BrainMriInputGate
from medalix.ingestion.chest_xray_input_gate import ChestXrayInputGate
from medalix.ingestion.ingestion_validator import IngestionValidator
from medalix.ingestion.medical_xray_input_gate import MedicalXrayInputGate
from medalix.inference.ensemble_model import EnsembleModel
from medalix.ood.ood_detector import OODDetector
from medalix.policy.governed_decision_policy import GovernedDecisionPolicy
from medalix.policy.policy_models import PolicyInput
from medalix.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from medalix.quality.quality_assessor import QualityAssessor
from medalix.registry.model_registry import ModelRegistry
from medalix.registry.model_router import ModelRouter
from medalix.retention.data_retention_manager import DataRetentionManager
from medalix.retention.temp_file_manager import TempFileManager

from .cleanup_coordinator import CleanupCoordinator
from .error_response_builder import ErrorResponseBuilder
from .pipeline_state import PipelineState
from .session_store import SessionStore

NON_DIAGNOSTIC_DISCLAIMER = (
    "Research-use only. This output is non-diagnostic and must not be used "
    "for clinical decision-making."
)


class Orchestrator:
    def __init__(self) -> None:
        self.logger = Logger()
        self.trace_builder = AuditTraceBuilder()
        self.temp_file_manager = TempFileManager()
        self.retention_manager = DataRetentionManager()
        self.cleanup_coordinator = CleanupCoordinator(
            self.retention_manager,
            self.logger,
        )
        self.session_store = SessionStore(ttl_seconds=1800)

        registry_path = Path("reference_data/model_registry.json")
        self.model_registry = ModelRegistry(str(registry_path))
        self.model_router = ModelRouter(self.model_registry)

        self.medical_gate = MedicalXrayInputGate()
        self.chest_gate = ChestXrayInputGate()
        self.bone_gate = BoneXrayInputGate()
        self.brain_gate = BrainMriInputGate()

    def get_result(self, analysis_id: str) -> dict | None:
        result = self.session_store.get_result(analysis_id)
        if result is None:
            self.logger.info({"event": "result_lookup_miss", "analysis_id": analysis_id})
        else:
            self.logger.info({"event": "result_lookup_hit", "analysis_id": analysis_id})
        return result

    def _finalize_and_store(self, state: PipelineState, payload: dict) -> dict:
        self.session_store.save_result(state.analysis_id, payload)
        self.logger.info({"event": "result_stored", "analysis_id": state.analysis_id})
        return payload

    def _build_input_gate_result(self, route_decision: dict | None = None) -> dict:
        route_decision = route_decision or {}
        hard_reject = route_decision.get("hard_reject", False)
        return {
            "accepted_for_analysis": not hard_reject,
            "confidence": route_decision.get("confidence"),
            "message": (
                route_decision.get("message")
                or (
                    "Input accepted by the active workflow."
                    if not hard_reject
                    else "Input rejected by the active workflow."
                )
            ),
            "selected_route": route_decision.get("selected_route"),
            "route_scores": route_decision.get("route_scores", {}),
            "top_level_gate": route_decision.get("top_level_gate", {}),
        }

    def _build_detection_result(self, raw_detection_result: dict | None = None) -> dict:
        raw_detection_result = raw_detection_result or {}
        return {
            "region": raw_detection_result.get("region"),
            "modality": raw_detection_result.get("modality"),
            "confidence": raw_detection_result.get("confidence"),
            "requires_confirmation": bool(
                raw_detection_result.get("requires_confirmation", False)
            ),
            "supported": bool(raw_detection_result.get("supported", False)),
            "reason": raw_detection_result.get("reason"),
        }

    def _build_routing_result(
        self,
        raw_routing_result: dict | None = None,
        selected_model: str | None = None,
    ) -> dict:
        raw_routing_result = raw_routing_result or {}
        accepted_findings_set = raw_routing_result.get("accepted_findings_set", [])
        return {
            "accepted_findings_set": accepted_findings_set,
            "selected_model": selected_model or raw_routing_result.get("selected_model"),
            "set_size": raw_routing_result.get("set_size", len(accepted_findings_set)),
            "requires_confirmation": raw_routing_result.get("requires_confirmation", False),
            "alpha": raw_routing_result.get("alpha"),
            "threshold": raw_routing_result.get("threshold"),
            "top_probability": raw_routing_result.get("top_probability"),
            "nonconformity": raw_routing_result.get("nonconformity"),
            "routing_candidates": raw_routing_result.get(
                "routing_candidates", accepted_findings_set
            ),
        }

    def _build_policy_result(
        self,
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

    def _build_ood_result(self, ood_result: dict | None = None) -> dict:
        ood_result = ood_result or {}
        return {
            "tier": ood_result.get("tier"),
            "score": ood_result.get("score"),
            "is_hard_ood": ood_result.get("is_hard_ood", False),
            "reason": ood_result.get("reason"),
        }

    def _build_explainability_result(self, explainability_result: dict | None = None) -> dict:
        explainability_result = explainability_result or {}
        return {
            "method": explainability_result.get("method"),
            "heatmap_path": explainability_result.get("heatmap_path"),
            "warning": explainability_result.get("warning"),
        }

    def _build_quality_result(self, quality_result: dict | None = None) -> dict:
        quality_result = quality_result or {}
        return {
            "status": quality_result.get("status"),
            "warnings": quality_result.get("warnings", []),
            "requires_reupload": quality_result.get("requires_reupload"),
            "blocking": quality_result.get("blocking", False),
            "reason": quality_result.get("reason"),
            "metrics": quality_result.get("metrics", {}),
        }

    def _build_inference_result(self, inference_result: dict | None = None) -> dict:
        inference_result = inference_result or {}
        return {
            "top_label": inference_result.get("top_label"),
            "top_probability": inference_result.get("top_probability"),
            "positive_findings": inference_result.get("positive_findings", []),
            "probabilities": inference_result.get("probabilities", {}),
            "epistemic_uncertainty": inference_result.get("epistemic_uncertainty", {}),
            "aleatoric_uncertainty": inference_result.get("aleatoric_uncertainty", {}),
            "reliability_score": inference_result.get("reliability_score"),
            "disagreement_score": inference_result.get("disagreement_score"),
            "secondary_verification_triggered": inference_result.get(
                "secondary_verification_triggered", False
            ),
            "ensemble_member_count": inference_result.get("ensemble_member_count"),
        }

    def _selected_model_payload(self, selected_model) -> dict:
        if selected_model is None:
            return {}
        return {
            "model_id": getattr(selected_model, "model_id", None),
            "version": getattr(selected_model, "version", None),
            "architecture": getattr(selected_model, "architecture", None),
            "region": getattr(selected_model, "region", None),
            "modality": getattr(selected_model, "modality", None),
            "input_shape": list(getattr(selected_model, "input_shape", [])),
            "status": getattr(selected_model, "status", None),
        }

    def _build_trace(self, state: PipelineState):
        return self.trace_builder.build(
            filename=state.filename,
            input_gate=state.input_gate_result or {},
            detection=state.detection_result or {},
            routing=state.routing_result or {},
            selected_model=self._selected_model_payload(state.selected_model),
            ood=state.ood_result or {},
            quality=state.quality_result or {},
            inference_summary=state.inference_result or {},
            policy=state.policy_result or {},
            explainability=state.explainability_result or {},
            pipeline_stages=state.stage_history,
        )

    def _select_route(self, filename: str, content: bytes) -> dict:
        brain_result = self.brain_gate.evaluate(content)
        brain_conf = float(brain_result.get("confidence", 0.0) or 0.0)
        brain_is_brain = bool(brain_result.get("is_brain_mri_like", False))

        xray_result = self.medical_gate.evaluate(content)
        xray_conf = float(xray_result.get("confidence", 0.0) or 0.0)
        xray_is_medical = bool(xray_result.get("is_medical_xray_like", False))
        xray_hard_reject = bool(xray_result.get("hard_reject", False))

        chest_result = self.chest_gate.evaluate(content)
        bone_result = self.bone_gate.evaluate(content, filename=filename)

        chest_conf = float(chest_result.get("confidence", 0.0) or 0.0)
        bone_conf = float(bone_result.get("confidence", 0.0) or 0.0)

        chest_is_chest = bool(chest_result.get("is_chest_xray_like", False))
        chest_hard_reject = bool(chest_result.get("hard_reject", False))
        bone_is_bone = bool(bone_result.get("is_bone_xray_like", False))

        route_scores = {
            "brain_mri": brain_conf,
            "medical_xray_gate": xray_conf,
            "chest_xray": chest_conf,
            "bone_xray": bone_conf,
        }

        if brain_is_brain and brain_conf >= 0.80 and brain_conf >= xray_conf:
            return {
                "hard_reject": False,
                "confidence": brain_conf,
                "selected_route": "brain_mri",
                "route_scores": route_scores,
                "top_level_gate": {
                    "brain_mri_gate": brain_result,
                    "medical_xray_gate": xray_result,
                },
                "message": "Routed to brain MRI pipeline.",
            }

        if xray_hard_reject and not brain_is_brain:
            return {
                "hard_reject": True,
                "confidence": max(brain_conf, xray_conf),
                "selected_route": None,
                "route_scores": route_scores,
                "top_level_gate": {
                    "brain_mri_gate": brain_result,
                    "medical_xray_gate": xray_result,
                },
                "message": "Input rejected by top-level medical image gates.",
            }

        if xray_is_medical:
            if chest_is_chest and chest_conf >= 0.80:
                return {
                    "hard_reject": False,
                    "confidence": chest_conf,
                    "selected_route": "chest_xray",
                    "route_scores": route_scores,
                    "top_level_gate": {
                        "brain_mri_gate": brain_result,
                        "medical_xray_gate": xray_result,
                    },
                    "message": "Routed to chest X-ray pipeline.",
                }

            if bone_is_bone and bone_conf >= 0.80:
                return {
                    "hard_reject": False,
                    "confidence": bone_conf,
                    "selected_route": "bone_xray",
                    "route_scores": route_scores,
                    "top_level_gate": {
                        "brain_mri_gate": brain_result,
                        "medical_xray_gate": xray_result,
                    },
                    "message": "Routed to bone X-ray pipeline.",
                }

            if chest_hard_reject and bone_is_bone and bone_conf >= 0.60:
                return {
                    "hard_reject": False,
                    "confidence": bone_conf,
                    "selected_route": "bone_xray",
                    "route_scores": route_scores,
                    "top_level_gate": {
                        "brain_mri_gate": brain_result,
                        "medical_xray_gate": xray_result,
                    },
                    "message": "Routed to bone X-ray pipeline after chest rejection.",
                }

        return {
            "hard_reject": True,
            "confidence": max(brain_conf, xray_conf, chest_conf, bone_conf),
            "selected_route": None,
            "route_scores": route_scores,
            "top_level_gate": {
                "brain_mri_gate": brain_result,
                "medical_xray_gate": xray_result,
            },
            "message": "Could not safely route input to brain MRI, chest X-ray, or bone X-ray pipeline.",
        }

    def _route_to_detection(self, selected_route: str | None, confidence: float) -> dict:
        if selected_route == "chest_xray":
            return {
                "region": "chest",
                "modality": "xray",
                "confidence": confidence,
                "requires_confirmation": False,
                "supported": True,
                "reason": "Chest X-ray route selected by medical image dispatcher.",
            }

        if selected_route == "bone_xray":
            return {
                "region": "bone",
                "modality": "xray",
                "confidence": confidence,
                "requires_confirmation": False,
                "supported": True,
                "reason": "Bone X-ray route selected by medical image dispatcher.",
            }

        if selected_route == "brain_mri":
            return {
                "region": "brain",
                "modality": "mri",
                "confidence": confidence,
                "requires_confirmation": False,
                "supported": True,
                "reason": "Brain MRI route selected by medical image dispatcher.",
            }

        return {
            "region": None,
            "modality": None,
            "confidence": confidence,
            "requires_confirmation": True,
            "supported": False,
            "reason": "No valid medical image route selected.",
        }

    def execute(self, filename: str, content_type: str | None, content: bytes) -> dict:
        state = PipelineState(
            analysis_id=self.session_store.create_analysis_id(),
            filename=filename,
            content_type=content_type or "application/octet-stream",
            size_bytes=len(content),
        )

        try:
            state.set_stage("temp_save")
            state.temp_path = self.temp_file_manager.save_upload(filename, content)

            state.set_stage("ingestion_validation")
            validator = IngestionValidator()
            validator.validate(filename, content_type, content)

            state.set_stage("input_gate")
            raw_input_gate_result = self._select_route(filename=filename, content=content)
            state.input_gate_result = self._build_input_gate_result(raw_input_gate_result)

            if raw_input_gate_result.get("hard_reject", False):
                state.detection_result = {}
                state.routing_result = {}
                state.quality_result = {}
                state.ood_result = {}
                state.explainability_result = {}
                state.policy_result = self._build_policy_result(
                    action="STOP",
                    reason=raw_input_gate_result.get("message", "Input rejected by route selector."),
                    risk_category="HIGH",
                    warnings=["Input not accepted for analysis"],
                )
                state.trace = self._build_trace(state)
                self.logger.info(
                    {
                        "event": "analysis_completed",
                        "analysis_id": state.analysis_id,
                        "trace": state.trace.to_dict(),
                    }
                )
                return self._finalize_and_store(
                    state,
                    {
                        "analysis_id": state.analysis_id,
                        "filename": state.filename,
                        "content_type": state.content_type,
                        "size_bytes": state.size_bytes,
                        "input_gate": state.input_gate_result,
                        "detection": state.detection_result,
                        "routing": state.routing_result,
                        "quality": state.quality_result,
                        "ood": state.ood_result,
                        "explainability": state.explainability_result,
                        "selected_model": self._selected_model_payload(state.selected_model),
                        "policy": state.policy_result,
                        "warnings": state.warnings,
                        "disclaimer": NON_DIAGNOSTIC_DISCLAIMER,
                        "message": "Analysis stopped before inference",
                    },
                )

            selected_route = raw_input_gate_result.get("selected_route")

            state.set_stage("preprocessing")
            preprocessor = PreprocessingPipeline()
            modality_for_preprocessing = "mri" if selected_route == "brain_mri" else "xray"
            state.original_image, state.tensor = preprocessor.run(
                content,
                modality=modality_for_preprocessing,
            )

            state.set_stage("quality")
            quality_assessor = QualityAssessor()
            raw_quality_result = quality_assessor.assess(state.tensor)
            state.quality_result = self._build_quality_result(raw_quality_result)

            state.set_stage("detection")
            raw_detection_result = self._route_to_detection(
                selected_route=selected_route,
                confidence=float(raw_input_gate_result.get("confidence", 0.0) or 0.0),
            )
            state.detection_result = self._build_detection_result(raw_detection_result)

            state.set_stage("routing")
            route_name = f"{state.detection_result['region']}:{state.detection_result['modality']}"
            raw_routing_result = {
                "accepted_findings_set": [route_name],
                "set_size": 1,
                "requires_confirmation": False,
                "alpha": 0.0,
                "threshold": 0.0,
                "top_probability": state.detection_result["confidence"],
                "nonconformity": 1.0 - float(state.detection_result["confidence"] or 0.0),
                "routing_candidates": [route_name],
                "selected_model": None,
            }
            state.routing_result = self._build_routing_result(raw_routing_result)

            state.set_stage("registry")
            try:
                selected_model_metadata = self.model_router.resolve(
                    region=state.detection_result["region"],
                    modality=state.detection_result["modality"],
                    input_shape=tuple(state.tensor.shape),
                )
            except ValueError as exc:
                state.ood_result = {}
                state.explainability_result = {}
                state.policy_result = self._build_policy_result(
                    action="STOP",
                    reason=str(exc),
                    risk_category="HIGH",
                    warnings=["Route is known but no active compatible model is available"],
                )
                state.trace = self._build_trace(state)
                self.logger.warn(
                    {
                        "event": "registry_resolution_blocked",
                        "analysis_id": state.analysis_id,
                        "filename": state.filename,
                        "stage": state.current_stage,
                        "message": str(exc),
                    }
                )
                return self._finalize_and_store(
                    state,
                    {
                        "analysis_id": state.analysis_id,
                        "filename": state.filename,
                        "content_type": state.content_type,
                        "size_bytes": state.size_bytes,
                        "input_gate": state.input_gate_result,
                        "detection": state.detection_result,
                        "routing": state.routing_result,
                        "quality": state.quality_result,
                        "ood": state.ood_result,
                        "explainability": state.explainability_result,
                        "selected_model": self._selected_model_payload(state.selected_model),
                        "policy": state.policy_result,
                        "warnings": state.warnings,
                        "disclaimer": NON_DIAGNOSTIC_DISCLAIMER,
                        "message": "Analysis stopped because the routed model is not available yet",
                    },
                )

            state.selected_model = selected_model_metadata
            state.routing_result = self._build_routing_result(
                raw_routing_result=raw_routing_result,
                selected_model=selected_model_metadata.model_id,
            )

            state.set_stage("inference")
            try:
                model = EnsembleModel(model_metadata=selected_model_metadata)
                raw_inference_result = model.predict(state.tensor)
                state.inference_result = self._build_inference_result(raw_inference_result)
            except Exception as exc:
                self.logger.error(
                    {
                        "event": "inference_failed",
                        "analysis_id": state.analysis_id,
                        "filename": state.filename,
                        "stage": state.current_stage,
                        "message": str(exc),
                    }
                )
                return self._finalize_and_store(
                    state,
                    ErrorResponseBuilder.build_processing_error(
                        analysis_id=state.analysis_id,
                        stage=state.current_stage,
                        message=str(exc),
                    ),
                )

            state.set_stage("ood")
            if state.detection_result.get("region") in {"bone", "brain"}:
                raw_ood_result = {
                    "score": 0.0,
                    "tier": "IN_DISTRIBUTION",
                    "is_hard_ood": False,
                    "reason": (
                        f"{state.detection_result.get('region', 'Route').title()} route "
                        "currently bypasses OOD until route-specific feature stats are available."
                    ),
                }
            else:
                ood_detector = OODDetector()
                raw_ood_result = ood_detector.evaluate(state.tensor, raw_inference_result)
            state.ood_result = self._build_ood_result(raw_ood_result)

            state.set_stage("explainability")
            try:
                explainability_engine = ExplainabilityEngine(model.model)
                raw_explainability_result = explainability_engine.generate(
                    original_image=state.original_image,
                    tensor=state.tensor,
                    inference_result=raw_inference_result,
                    filename=filename,
                )
                state.explainability_result = self._build_explainability_result(
                    raw_explainability_result
                )
            except Exception as exc:
                warning_message = f"Explainability generation failed: {str(exc)}"
                state.warnings.append(warning_message)
                self.logger.warn(
                    {
                        "event": "explainability_failed",
                        "analysis_id": state.analysis_id,
                        "filename": state.filename,
                        "stage": state.current_stage,
                        "message": str(exc),
                    }
                )
                state.explainability_result = self._build_explainability_result(
                    {
                        "method": None,
                        "heatmap_path": None,
                        "warning": warning_message,
                    }
                )

            state.set_stage("policy")
            policy_input = PolicyInput(
                ood_result=state.ood_result,
                routing_result=state.routing_result,
                inference_result=raw_inference_result,
                quality_result=state.quality_result,
            )
            policy_output = GovernedDecisionPolicy().evaluate(policy_input)
            state.policy_result = self._build_policy_result(
                action=policy_output.action,
                reason=policy_output.reason,
                risk_category=policy_output.risk_category,
                warnings=policy_output.warnings,
            )

            state.trace = self._build_trace(state)
            self.logger.info(
                {
                    "event": "analysis_completed",
                    "analysis_id": state.analysis_id,
                    "trace": state.trace.to_dict(),
                }
            )

            payload = {
                "analysis_id": state.analysis_id,
                "filename": state.filename,
                "content_type": state.content_type,
                "size_bytes": state.size_bytes,
                "input_gate": state.input_gate_result,
                "detection": state.detection_result,
                "quality": state.quality_result,
                "ood": state.ood_result,
                "routing": state.routing_result,
                "selected_model": self._selected_model_payload(state.selected_model),
                "explainability": state.explainability_result,
                "policy": state.policy_result,
                "warnings": state.warnings,
                "disclaimer": NON_DIAGNOSTIC_DISCLAIMER,
            }

            if policy_output.action != "STOP":
                payload.update(
                    {
                        "tensor_shape": list(state.tensor.shape),
                        "inference": state.inference_result,
                        "message": "Governed pipeline completed successfully",
                    }
                )
            else:
                payload["message"] = "Analysis stopped due to policy or distribution checks"

            return self._finalize_and_store(state, payload)

        except ValueError as exc:
            self.logger.error(
                {
                    "event": "validation_failed",
                    "analysis_id": state.analysis_id,
                    "filename": state.filename,
                    "stage": state.current_stage,
                    "message": str(exc),
                }
            )
            return self._finalize_and_store(
                state,
                ErrorResponseBuilder.build_validation_error(
                    analysis_id=state.analysis_id,
                    stage=state.current_stage,
                    message=str(exc),
                ),
            )
        except Exception as exc:
            self.logger.error(
                {
                    "event": "processing_failed",
                    "analysis_id": state.analysis_id,
                    "filename": state.filename,
                    "stage": state.current_stage,
                    "message": str(exc),
                }
            )
            return self._finalize_and_store(
                state,
                ErrorResponseBuilder.build_processing_error(
                    analysis_id=state.analysis_id,
                    stage=state.current_stage,
                    message=str(exc),
                ),
            )
        finally:
            self.cleanup_coordinator.cleanup_temp_file(state.temp_path)