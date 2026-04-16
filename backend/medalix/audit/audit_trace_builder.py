from medalix.audit.audit_trace import AuditTrace


class AuditTraceBuilder:
    def build(
        self,
        filename: str,
        input_gate: dict,
        quality: dict,
        ood: dict,
        routing: dict,
        policy: dict,
        inference_summary: dict,
    ) -> AuditTrace:
        return AuditTrace(
            filename=filename,
            input_gate=input_gate,
            quality=quality,
            ood=ood,
            routing=routing,
            policy=policy,
            inference_summary=inference_summary,
        )