from medalix.audit.audit_trace import AuditTrace


class AuditTraceBuilder:
    def build(
        self,
        analysis_id: str,
        filename: str,
        content_type: str | None,
        size_bytes: int | None,
        input_gate: dict,
        detection: dict,
        routing: dict,
        selected_model: dict,
        ood: dict,
        quality: dict,
        inference_summary: dict,
        policy: dict,
        explainability: dict,
        conversion: dict | None = None,
        pipeline_stages: list[str] | None = None,
    ) -> AuditTrace:
        return AuditTrace(
            analysis_id=analysis_id,
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            input_gate=dict(input_gate or {}),
            detection=dict(detection or {}),
            routing=dict(routing or {}),
            selected_model=dict(selected_model or {}),
            ood=dict(ood or {}),
            quality=dict(quality or {}),
            inference_summary=dict(inference_summary or {}),
            policy=dict(policy or {}),
            explainability=dict(explainability or {}),
            conversion=dict(conversion or {}),
            pipeline_stages=list(pipeline_stages or []),
        )