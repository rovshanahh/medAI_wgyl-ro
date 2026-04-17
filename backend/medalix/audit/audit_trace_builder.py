from medalix.audit.audit_trace import AuditTrace


class AuditTraceBuilder:
    def build(
        self,
        filename: str,
        input_gate: dict,
        detection: dict,
        routing: dict,
        selected_model: dict,
        ood: dict,
        quality: dict,
        inference_summary: dict,
        policy: dict,
        explainability: dict,
        pipeline_stages: list[str] | None = None,
    ) -> AuditTrace:
        return AuditTrace(
            filename=filename,
            input_gate=input_gate or {},
            detection=detection or {},
            routing=routing or {},
            selected_model=selected_model or {},
            ood=ood or {},
            quality=quality or {},
            inference_summary=inference_summary or {},
            policy=policy or {},
            explainability=explainability or {},
            pipeline_stages=pipeline_stages or [],
        )