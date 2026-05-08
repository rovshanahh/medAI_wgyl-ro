from datetime import datetime, timezone


class AnalysisReportBuilder:
    @staticmethod
    def _get(data: dict, *keys, default="—"):
        current = data

        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key)

        return current if current is not None else default

    @staticmethod
    def _percent(value):
        if value is None or value == "—":
            return "—"

        try:
            return f"{float(value) * 100:.1f}%"
        except (TypeError, ValueError):
            return "—"

    @classmethod
    def build(cls, result: dict) -> dict:
        analysis_id = result.get("analysis_id", "—")
        filename = result.get("filename", "—")

        route = cls._get(result, "input_gate", "selected_route")
        region = cls._get(result, "detection", "region")
        modality = cls._get(result, "detection", "modality")

        policy_action = cls._get(result, "policy", "action")
        policy_reason = cls._get(result, "policy", "reason")
        risk_category = cls._get(result, "policy", "risk_category")

        model_id = cls._get(result, "routing", "selected_model")
        output = cls._get(result, "inference", "top_label", default="No inference")
        confidence = cls._percent(cls._get(result, "inference", "top_probability", default=None))

        ood_tier = cls._get(result, "ood", "tier")
        ood_reason = cls._get(result, "ood", "reason")

        quality_status = cls._get(result, "quality", "status")
        quality_reason = cls._get(result, "quality", "reason")

        heatmap_path = cls._get(result, "explainability", "heatmap_path")
        explanation_method = cls._get(result, "explainability", "method")

        disclaimer = result.get(
            "disclaimer",
            "Research-use only. Outputs are non-diagnostic and must not be used for clinical decision-making.",
        )

        generated_at = datetime.now(timezone.utc).isoformat()

        summary = {
            "analysis_id": analysis_id,
            "generated_at": generated_at,
            "filename": filename,
            "route": route,
            "region": region,
            "modality": modality,
            "selected_model": model_id,
            "policy_action": policy_action,
            "risk_category": risk_category,
            "output": output,
            "confidence": confidence,
            "heatmap_path": heatmap_path,
        }

        sections = {
            "decision": {
                "policy_action": policy_action,
                "risk_category": risk_category,
                "reason": policy_reason,
            },
            "routing": {
                "selected_route": route,
                "region": region,
                "modality": modality,
                "selected_model": model_id,
            },
            "inference": {
                "output": output,
                "confidence": confidence,
                "probabilities": cls._get(result, "inference", "probabilities", default={}),
                "reliability_score": cls._get(result, "inference", "reliability_score"),
                "disagreement_score": cls._get(result, "inference", "disagreement_score"),
            },
            "safety_checks": {
                "ood_tier": ood_tier,
                "ood_reason": ood_reason,
                "quality_status": quality_status,
                "quality_reason": quality_reason,
                "warnings": result.get("warnings", []),
            },
            "explainability": {
                "method": explanation_method,
                "heatmap_path": heatmap_path,
                "target_label": cls._get(result, "explainability", "target_label"),
            },
            "pipeline": {
                "stages": result.get("pipeline_stages", []),
                "timing_summary": result.get("timing_summary", []),
            },
        }

        plain_text = cls._build_plain_text(summary, sections, disclaimer)

        return {
            "summary": summary,
            "sections": sections,
            "plain_text": plain_text,
            "disclaimer": disclaimer,
        }

    @staticmethod
    def _build_plain_text(summary: dict, sections: dict, disclaimer: str) -> str:
        lines = [
            "MedAIx Analysis Report",
            "=" * 24,
            "",
            f"Analysis ID: {summary['analysis_id']}",
            f"Generated at: {summary['generated_at']}",
            f"Filename: {summary['filename']}",
            "",
            "Decision",
            "-" * 8,
            f"Policy action: {sections['decision']['policy_action']}",
            f"Risk category: {sections['decision']['risk_category']}",
            f"Reason: {sections['decision']['reason']}",
            "",
            "Routing",
            "-" * 7,
            f"Route: {sections['routing']['selected_route']}",
            f"Region: {sections['routing']['region']}",
            f"Modality: {sections['routing']['modality']}",
            f"Selected model: {sections['routing']['selected_model']}",
            "",
            "Inference",
            "-" * 9,
            f"Output: {sections['inference']['output']}",
            f"Confidence: {sections['inference']['confidence']}",
            f"Reliability score: {sections['inference']['reliability_score']}",
            f"Disagreement score: {sections['inference']['disagreement_score']}",
            "",
            "Safety Checks",
            "-" * 13,
            f"OOD tier: {sections['safety_checks']['ood_tier']}",
            f"OOD reason: {sections['safety_checks']['ood_reason']}",
            f"Quality status: {sections['safety_checks']['quality_status']}",
            f"Quality reason: {sections['safety_checks']['quality_reason']}",
            "",
            "Explainability",
            "-" * 14,
            f"Method: {sections['explainability']['method']}",
            f"Target label: {sections['explainability']['target_label']}",
            f"Heatmap path: {sections['explainability']['heatmap_path']}",
            "",
            "Disclaimer",
            "-" * 10,
            disclaimer,
        ]

        return "\n".join(lines)