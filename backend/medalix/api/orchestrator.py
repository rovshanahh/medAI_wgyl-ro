from pathlib import Path

from medalix.audit.audit_trace_builder import AuditTraceBuilder
from medalix.audit.logger import Logger
from medalix.detection.route_detector import RouteDetector
from medalix.explainability.explainability_engine import ExplainabilityEngine
from medalix.ingestion.dicom_converter import DicomConverter
from medalix.ingestion.ingestion_validator import IngestionValidator
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
    ROUTE_TO_PREPROCESSING_MODALITY = {
        "brain_mri": "mri",
        "bone_xray": "xray",
        "chest_xray": "xray",
        "retina_fundus": "fundus",
        "skin_dermoscopy": "dermoscopy",
    }

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

        self.route_detector = RouteDetector()
        self.dicom_converter = DicomConverter()

    def get_result(self, analysis_id: str) -> dict | None:
        result = self.session_store.get_result(analysis_id)

        if result is None:
            self.logger.info({"event": "result_lookup_miss", "analysis_id": analysis_id})
        else:
            self.logger.info({"event": "result_lookup_hit", "analysis_id": analysis_id})

        return result

    def _resolve_preprocessing_modality(self, selected_route: str) -> str:
        return self.ROUTE_TO_PREPROCESSING_MODALITY.get(selected_route, "xray")

    def _finalize_and_store(self, state: PipelineState, payload: dict) -> dict:
        payload = self._with_pipeline_debug(state, payload)
        self.session_store.save_result(state.analysis_id, payload)
        self.logger.info({"event": "result_stored", "analysis_id": state.analysis_id})
        return payload

    def _with_pipeline_debug(self, state: PipelineState, payload: dict) -> dict:
        payload = dict(payload or {})
        payload["pipeline_stages"] = list(state.stage_history)
        payload["stage_events"] = list(state.stage_events)
        payload["timing_summary"] = state.timing_summary()
        return payload

    def _prepare_working_content(self, filename: str, content: bytes) -> tuple[bytes, str, dict]:
        extension = Path(filename).suffix.lower()

        if extension != ".dcm":
            return content, filename, {
                "converted": False,
                "source_format": extension,
                "working_format": extension,
                "message": "No format conversion needed.",
            }

        converted_content = self.dicom_converter.convert_to_png_bytes(content)
        converted_filename = f"{Path(filename).stem}.png"

        return converted_content, converted_filename, {
            "converted": True,
            "source_format": ".dcm",
            "working_format": ".png",
            "message": "DICOM image converted to PNG-compatible bytes for MVP inference pipeline.",
        }

    def _build_input_gate_result(
        self,
        route_result: dict | None = None,
        conversion_result: dict | None = None,
    ) -> dict:
        route_result = route_result or {}
        conversion_result = conversion_result or {}
        supported = bool(route_result.get("supported", False))

        return {
            "accepted_for_analysis": supported,
            "confidence": route_result.get("confidence"),
            "message": (
                "Input accepted by multi-class route detector."
                if supported
                else route_result.get(
                    "reason",
                    "Input not accepted by multi-class route detector.",
                )
            ),
            "selected_route": route_result.get("route_label"),
            "route_scores": route_result.get("probabilities", {}),
            "top_level_gate": {
                "route_detector": route_result,
                "conversion": conversion_result,
            },
        }

    def _build_detection_result(self, route_result: dict | None = None) -> dict:
        route_result = route_result or {}

        return {
            "region": route_result.get("region"),
            "modality": route_result.get("modality"),
            "confidence": route_result.get("confidence"),
            "requires_confirmation": bool(route_result.get("requires_confirmation", True)),
            "supported": bool(route_result.get("supported", False)),
            "reason": route_result.get("reason"),
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
                "routing_candidates",
                accepted_findings_set,
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
            "method": ood_result.get("method"),
            "metrics": ood_result.get("metrics", {}),
        }

    def _build_explainability_result(self, explainability_result: dict | None = None) -> dict:
        explainability_result = explainability_result or {}

        return {
            "method": explainability_result.get("method"),
            "heatmap_path": explainability_result.get("heatmap_path"),
            "warning": explainability_result.get("warning"),
            "target_label": explainability_result.get("target_label"),
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
                "secondary_verification_triggered",
                False,
            ),
            "ensemble_member_count": inference_result.get("ensemble_member_count"),
            "mc_passes": inference_result.get("mc_passes"),
            "model_cache_key": inference_result.get("model_cache_key"),
            "device": inference_result.get("device"),
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
        input_gate = state.input_gate_result or {}
        conversion = (
            input_gate.get("top_level_gate", {}).get("conversion", {})
            if isinstance(input_gate, dict)
            else {}
        )

        return self.trace_builder.build(
            analysis_id=state.analysis_id,
            filename=state.filename,
            content_type=state.content_type,
            size_bytes=state.size_bytes,
            input_gate=state.input_gate_result or {},
            detection=state.detection_result or {},
            routing=state.routing_result or {},
            selected_model=self._selected_model_payload(state.selected_model),
            ood=state.ood_result or {},
            quality=state.quality_result or {},
            inference_summary=state.inference_result or {},
            policy=state.policy_result or {},
            explainability=state.explainability_result or {},
            conversion=conversion,
            pipeline_stages=state.stage_history,
        )

    def _base_payload(self, state: PipelineState, message: str) -> dict:
        return {
            "analysis_id": state.analysis_id,
            "filename": state.filename,
            "content_type": state.content_type,
            "size_bytes": state.size_bytes,
            "input_gate": state.input_gate_result or {},
            "detection": state.detection_result or {},
            "routing": state.routing_result or {},
            "quality": state.quality_result or {},
            "ood": state.ood_result or {},
            "explainability": state.explainability_result or {},
            "selected_model": self._selected_model_payload(state.selected_model),
            "policy": state.policy_result or {},
            "warnings": state.warnings,
            "disclaimer": NON_DIAGNOSTIC_DISCLAIMER,
            "message": message,
        }

    def _stop_before_inference(
        self,
        state: PipelineState,
        reason: str,
        warnings: list[str] | None = None,
    ) -> dict:
        state.quality_result = state.quality_result or {}
        state.ood_result = state.ood_result or {}
        state.explainability_result = state.explainability_result or {}

        state.policy_result = self._build_policy_result(
            action="STOP",
            reason=reason,
            risk_category="HIGH",
            warnings=warnings or ["Input not accepted for analysis"],
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
            self._base_payload(state, "Analysis stopped before inference"),
        )

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

            state.set_stage("format_preparation")
            working_content, working_filename, conversion_result = self._prepare_working_content(
                filename=filename,
                content=content,
            )

            state.set_stage("route_detection")
            route_result = self.route_detector.predict(working_content)

            state.input_gate_result = self._build_input_gate_result(
                route_result=route_result,
                conversion_result=conversion_result,
            )
            state.detection_result = self._build_detection_result(route_result)

            if not route_result.get("supported", False):
                state.routing_result = {
                    "accepted_findings_set": [],
                    "selected_model": None,
                    "set_size": 0,
                    "requires_confirmation": True,
                    "alpha": None,
                    "threshold": None,
                    "top_probability": route_result.get("confidence"),
                    "nonconformity": None,
                    "routing_candidates": [],
                }

                return self._stop_before_inference(
                    state=state,
                    reason=route_result.get(
                        "reason",
                        "Route detector could not safely select a supported route.",
                    ),
                )

            selected_route = route_result["route_label"]
            route_name = f"{route_result['region']}:{route_result['modality']}"

            state.set_stage("preprocessing")
            preprocessor = PreprocessingPipeline()
            modality_for_preprocessing = self._resolve_preprocessing_modality(selected_route)

            state.original_image, state.tensor = preprocessor.run(
                working_content,
                modality=modality_for_preprocessing,
            )

            state.set_stage("quality")
            quality_assessor = QualityAssessor()
            raw_quality_result = quality_assessor.assess(state.tensor)
            state.quality_result = self._build_quality_result(raw_quality_result)

            state.set_stage("routing")
            raw_routing_result = {
                "accepted_findings_set": [route_name],
                "set_size": 1,
                "requires_confirmation": False,
                "alpha": 0.0,
                "threshold": 0.0,
                "top_probability": route_result.get("confidence"),
                "nonconformity": 1.0 - float(route_result.get("confidence") or 0.0),
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
                    self._base_payload(
                        state,
                        "Analysis stopped because the routed model is not available yet",
                    ),
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

                error_payload = ErrorResponseBuilder.build_processing_error(
                    analysis_id=state.analysis_id,
                    stage=state.current_stage,
                    message=str(exc),
                )
                return self._finalize_and_store(state, error_payload)

            state.set_stage("ood")
            ood_detector = OODDetector()
            raw_ood_result = ood_detector.evaluate(
                tensor=state.tensor,
                inference_result=raw_inference_result,
                route_label=selected_route,
            )
            state.ood_result = self._build_ood_result(raw_ood_result)

            state.set_stage("explainability")
            try:
                explainability_engine = ExplainabilityEngine(model.model)
                raw_explainability_result = explainability_engine.generate(
                    original_image=state.original_image,
                    tensor=state.tensor,
                    inference_result=raw_inference_result,
                    filename=working_filename,
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

            payload = self._base_payload(
                state,
                (
                    "Governed pipeline completed successfully"
                    if policy_output.action != "STOP"
                    else "Analysis stopped due to policy or distribution checks"
                ),
            )

            if policy_output.action != "STOP":
                payload.update(
                    {
                        "tensor_shape": list(state.tensor.shape),
                        "inference": state.inference_result,
                    }
                )

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

            error_payload = ErrorResponseBuilder.build_validation_error(
                analysis_id=state.analysis_id,
                stage=state.current_stage,
                message=str(exc),
            )
            return self._finalize_and_store(state, error_payload)

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

            error_payload = ErrorResponseBuilder.build_processing_error(
                analysis_id=state.analysis_id,
                stage=state.current_stage,
                message=str(exc),
            )
            return self._finalize_and_store(state, error_payload)

        finally:
            self.cleanup_coordinator.cleanup_temp_file(state.temp_path)