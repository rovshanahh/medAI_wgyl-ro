"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import SiteHeader from "../_components/SiteHeader";
import SiteFooter from "../_components/SiteFooter";
import { motion } from "framer-motion";
import {
  ArrowRight,
  CircleHelp,
  Home,
  LayoutDashboard,
  Mail,
  Play,
  ShieldCheck,
} from "lucide-react";

type DemoCase = {
  id: string;
  title: string;
  description: string;
  expected: string;
  route_override?: string;
};

type AnalysisResponse = {
  analysis_id?: string;
  filename?: string;
  input_gate?: {
    accepted_for_analysis?: boolean;
    selected_route?: string;
    manual_override?: boolean;
    top_level_gate?: {
      route_detector?: {
        raw_route_label?: string;
        manual_override?: boolean;
      };
    };
  };
  detection?: {
    region?: string | null;
    modality?: string | null;
    manual_override?: boolean;
  };
  routing?: {
    selected_model?: string | null;
  };
  inference?: {
    top_label?: string;
    top_probability?: number;
    probabilities?: Record<string, number>;
    uncertainty_method?: string;
    deep_ensemble_enabled?: boolean;
    mc_passes?: number;
  };
  explainability?: {
    method?: string | null;
    heatmap_path?: string | null;
    target_label?: string;
    warning?: string;
  };
  quality?: {
    status?: string | null;
    reason?: string;
    warnings?: string[];
  };
  ood?: {
    tier?: string;
    method?: string;
  };
  policy?: {
    action?: string;
    reason?: string;
    risk_category?: string;
    warnings?: string[];
  };
  warnings?: string[];
  message?: string;
};

type DemoRunResponse = {
  case_id: string;
  title: string;
  description: string;
  expected: string;
  result: AnalysisResponse;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

const fadeUp = {
  hidden: { opacity: 0, y: 14 },
  show: { opacity: 1, y: 0 },
};

function formatLabel(value?: string | null) {
  if (!value) return "—";
  return value.replaceAll("_", " ");
}

function formatPercent(value?: number | null) {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function MiniLogo() {
  return (
    <div className="relative h-5 w-8">
      <span className="absolute left-0 top-1/2 h-1.5 w-1.5 -translate-y-1/2 rounded-full bg-slate-950" />
      <span className="absolute right-0 top-1/2 h-1.5 w-1.5 -translate-y-1/2 rounded-full bg-slate-950" />
      <span className="absolute left-1/2 top-1/2 h-px w-4 -translate-x-1/2 -translate-y-1/2 rounded-full bg-slate-400" />
    </div>
  );
}

export default function DemoPage() {
  const [cases, setCases] = useState<DemoCase[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState("");
  const [demoResult, setDemoResult] = useState<DemoRunResponse | null>(null);
  const [loadingCases, setLoadingCases] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [heatmapOpacity, setHeatmapOpacity] = useState(65);

  useEffect(() => {
    const loadCases = async () => {
      try {
        setLoadingCases(true);
        setError("");

        const response = await fetch(`${API_BASE_URL}/demo/cases`);
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data?.detail || "Could not load demo cases.");
        }

        setCases(data.cases || []);

        if (data.cases?.length) {
          setSelectedCaseId(data.cases[0].id);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Something went wrong.");
      } finally {
        setLoadingCases(false);
      }
    };

    loadCases();
  }, []);

  const selectedCase = useMemo(
    () => cases.find((item) => item.id === selectedCaseId),
    [cases, selectedCaseId]
  );

  const result = demoResult?.result;

  const selectedRoute =
    result?.input_gate?.selected_route ||
    result?.input_gate?.top_level_gate?.route_detector?.raw_route_label ||
    "—";

  const heatmapRaw = result?.explainability?.heatmap_path || "";
  const heatmapUrl = heatmapRaw
    ? `${
        heatmapRaw.startsWith("http://") || heatmapRaw.startsWith("https://")
          ? heatmapRaw
          : `${API_BASE_URL}${heatmapRaw}`
      }?v=${result?.analysis_id || Date.now()}`
    : "";

  const inputImageUrl =
    demoResult && !result?.filename?.toLowerCase().endsWith(".dcm")
      ? `${API_BASE_URL}/demo/image/${demoResult.case_id}`
      : "";

  const manualOverrideUsed =
    Boolean(result?.input_gate?.manual_override) ||
    Boolean(result?.detection?.manual_override) ||
    Boolean(result?.input_gate?.top_level_gate?.route_detector?.manual_override);

  const warnings = [
    ...(result?.policy?.warnings ?? []),
    ...(result?.quality?.warnings ?? []),
    ...(result?.warnings ?? []),
    ...(result?.explainability?.warning ? [result.explainability.warning] : []),
  ].filter(Boolean);

  const handleRunDemo = async () => {
    if (!selectedCaseId) {
      setError("Please select a demo case first.");
      return;
    }

    try {
      setRunning(true);
      setError("");
      setDemoResult(null);

      const response = await fetch(`${API_BASE_URL}/demo/run/${selectedCaseId}`, {
        method: "POST",
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data?.detail || "Demo case failed.");
      }

      setDemoResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setRunning(false);
    }
  };

  return (
    <main
      className="relative min-h-screen overflow-hidden bg-[#fffaf3] text-slate-950"
      style={{ fontFamily: '"Aptos","Aptos Body","Segoe UI",Arial,sans-serif' }}
    >
      <div className="pointer-events-none motion-float absolute -right-40 -top-40 h-[520px] w-[520px] rounded-full bg-red-100/70 blur-3xl" />
      <div className="pointer-events-none absolute -left-44 top-[520px] h-[520px] w-[520px] rounded-full bg-sky-100/70 blur-3xl" />

      <div className="relative mx-auto max-w-7xl px-6 py-6">
        <SiteHeader active="demo" />

        <motion.section
          className="mb-10"
          initial="hidden"
          animate="show"
          variants={fadeUp}
          transition={{ duration: 0.5, ease: "easeOut" }}
        >
          <p className="text-xs uppercase tracking-[0.28em] text-red-500">
            Presentation-ready scenarios
          </p>

          <h2 className="mt-5 max-w-3xl text-4xl font-semibold leading-tight tracking-tight">
            Run a guided demo through the same review pipeline.
          </h2>

          <p className="mt-6 max-w-2xl text-sm italic leading-7 text-slate-500">
            Choose a prepared case instead of uploading a file. The system will still
            show routing, model output, OOD status, policy decision, and visual evidence.
          </p>
        </motion.section>

        <section className="grid gap-12 xl:grid-cols-[360px_minmax(0,1fr)]">
          <aside>
            <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
              Choose demo case
            </p>

            <p className="mt-3 text-sm leading-7 text-slate-500">
              Pick one prepared case and run it through the same governed review flow.
            </p>

            <div className="mt-7 max-h-[430px] space-y-2 overflow-y-auto pr-2">
              {loadingCases ? (
                <p className="text-sm text-slate-500">Loading demo cases...</p>
              ) : (
                cases.map((item) => (
                  <button
                    type="button"
                    key={item.id}
                    onClick={() => {
                      setSelectedCaseId(item.id);
                      setDemoResult(null);
                      setError("");
                    }}
                    className={`group relative w-full py-4 pl-5 text-left transition hover:text-red-500 ${
                      selectedCaseId === item.id ? "text-red-600" : "text-slate-800"
                    }`}
                  >
                    <span
                      className={`absolute left-0 top-5 h-2 w-2 rounded-full transition ${
                        selectedCaseId === item.id
                          ? "bg-red-500"
                          : "bg-slate-300 group-hover:bg-red-300"
                      }`}
                    />

                    <p className="text-sm font-semibold">{item.title}</p>
                    <p className="mt-2 text-xs leading-5 text-slate-500">
                      {item.description}
                    </p>
                  </button>
                ))
              )}
            </div>

            {selectedCase ? (
              <div className="mt-7">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
                  Expected behavior
                </p>
                <p className="mt-3 text-sm leading-7 text-slate-500">
                  {selectedCase.expected}
                </p>
              </div>
            ) : null}

            <button
              type="button"
              onClick={handleRunDemo}
              disabled={running || loadingCases}
              className="mt-7 group flex w-full items-center justify-between rounded-2xl border border-red-200 bg-red-50 px-5 py-4 text-sm font-semibold text-red-700 transition motion-hover-lift hover:bg-red-500 hover:text-white disabled:cursor-not-allowed disabled:opacity-60"
            >
              <span>{running ? "Running demo..." : "Run selected demo"}</span>
              <ArrowRight size={17} className="transition group-hover:translate-x-1" />
            </button>

            {error ? <p className="mt-4 text-sm text-red-600">{error}</p> : null}
          </aside>

          <section>
            <p className="text-sm text-slate-500">Assistant result</p>
            <h2
              className={`mt-3 text-3xl font-semibold tracking-tight ${getDecisionColor(
                result?.policy?.action
              )}`}
            >
              {getAssistantDecision(result?.policy?.action)}
            </h2>
            <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-600">
              {getAssistantSummary(result, selectedRoute)}
            </p>

            {!demoResult || !result ? (
              <div className="mt-8 grid max-w-2xl gap-5 sm:grid-cols-3 motion-fade-up-slow">
                <InfoLine label="Step one" value="Choose case" />
                <InfoLine label="Step two" value="Run pipeline" />
                <InfoLine label="Step three" value="Review evidence" />
              </div>
            ) : (
              <div className="motion-fade-up-slow">
                <div className="mt-8">
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-500">
                    Main output
                  </p>
                  <p className="mt-2 text-4xl font-semibold tracking-tight">
                    {result.inference?.top_label || "No inference"}
                  </p>
                  <p className="mt-2 text-sm text-slate-500">
                    {result.inference?.top_label
                      ? `Confidence: ${formatPercent(result.inference?.top_probability)}`
                      : result.policy?.action === "STOP"
                        ? "The system stopped before producing a model output."
                        : result.policy?.action === "REFUSE"
                          ? "The model ran, but the governed policy withheld the output because reliability was too low."
                          : "The system stopped before producing a model output."}
                  </p>
                </div>

                <div className="mt-8 grid gap-5 md:grid-cols-3">
                  <InfoLine
                    label="Route"
                    value={formatLabel(selectedRoute)}
                    text={`Accepted: ${
                      result.input_gate?.accepted_for_analysis ? "Yes" : "No"
                    }`}
                  />
                  <InfoLine
                    label="Safety"
                    value={result.policy?.risk_category || "—"}
                    text={`OOD: ${result.ood?.tier || "—"}`}
                  />
                  <InfoLine
                    label="Model"
                    value={result.routing?.selected_model || "—"}
                    text={`${result.detection?.region || "—"} / ${
                      result.detection?.modality || "—"
                    }`}
                  />
                </div>

                <div className="mt-8 border-y border-slate-300 py-5">
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-500">
                    Pipeline status
                  </p>

                  <div className="mt-4 grid gap-3 md:grid-cols-2">
                    <PipelineStep label="Input validated" done={Boolean(result.input_gate)} />
                    <PipelineStep label="Route selected" done={Boolean(selectedRoute && selectedRoute !== "—")} />
                    <PipelineStep label="Model inference completed" done={Boolean(result.inference?.top_label)} />
                    <PipelineStep label="Quality checked" done={Boolean(result.quality?.status)} />
                    <PipelineStep label="OOD screened" done={Boolean(result.ood?.tier)} />
                    <PipelineStep label="Focus map generated" done={Boolean(result.explainability?.heatmap_path)} />
                    <PipelineStep label="Policy decision made" done={Boolean(result.policy?.action)} />
                  </div>
                </div>

                <DecisionLegend />

                <div className="mt-10">
                  <div className="mb-4 flex items-end justify-between gap-4">
                    <div>
                      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">
                        Visual evidence
                      </p>
                      <p className="mt-1 text-sm text-slate-500">
                        Original image and Grad-CAM++ focus map shown together.
                      </p>
                    </div>
                  </div>

                  <div className="grid gap-5 lg:grid-cols-2">
                    <div>
                      <p className="mb-3 text-sm font-semibold">Original image</p>
                      <div className="flex min-h-[280px] items-center justify-center border-y border-slate-300 py-5">
                        {inputImageUrl ? (
                          <img
                            src={inputImageUrl}
                            alt="Demo input"
                            className="max-h-[340px] w-full object-contain"
                          />
                        ) : (
                          <p className="text-sm text-slate-400">
                            DICOM preview unavailable in browser.
                          </p>
                        )}
                      </div>
                    </div>

                    <div>
                      <p className="mb-3 text-sm font-semibold">
                        Where the model focused
                      </p>
                      <div className="min-h-[280px] border-y border-slate-300 py-5">
                        {heatmapUrl ? (
                          <div className="space-y-4">
                            <div className="relative flex min-h-[280px] items-center justify-center overflow-hidden border-y border-slate-300 bg-white">
                              {inputImageUrl ? (
                                <img
                                  src={inputImageUrl}
                                  alt="Original image under heatmap"
                                  className="max-h-[340px] w-full object-contain"
                                />
                              ) : null}

                              <img
                                src={heatmapUrl}
                                alt="Explainability heatmap overlay"
                                className="absolute inset-0 h-full w-full object-contain"
                                style={{ opacity: heatmapOpacity / 100 }}
                              />
                            </div>

                            <div className="border-t border-slate-200 pt-4">
                              <div className="mb-3 flex items-center justify-between text-xs">
                                <span className="uppercase tracking-[0.18em] text-slate-500">
                                  Heatmap opacity
                                </span>
                                <span className="font-semibold text-red-600">
                                  {heatmapOpacity}%
                                </span>
                              </div>

                              <input
                                type="range"
                                min="0"
                                max="100"
                                value={heatmapOpacity}
                                onChange={(event) => setHeatmapOpacity(Number(event.target.value))}
                                className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-red-100 accent-red-500"
                                aria-label="Heatmap opacity"
                              />

                              <div className="mt-2 flex justify-between text-[11px] text-slate-400">
                                <span>Original</span>
                                <span>Focus map</span>
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className="flex min-h-[280px] items-center justify-center">
                            <p className="text-sm text-slate-400">
                              Focus map unavailable.
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                <details className="mt-8 border-y border-slate-300 py-5">
                  <summary className="flex cursor-pointer list-none items-center justify-between gap-4 text-sm font-semibold text-slate-900">
                    <span>
                      Technical audit
                      <span className="ml-2 text-xs font-normal text-slate-500">
                        explained step by step
                      </span>
                    </span>
                    <span className="text-xs font-medium text-red-500">
                      Open details
                    </span>
                  </summary>

                  <div className="mt-5 border-t border-slate-200 pt-5">
                    <p className="mb-6 max-w-2xl text-sm leading-6 text-slate-600">
                      This audit explains what happened behind the assistant result.
                      It is useful for researchers, reviewers, and demo evaluators.
                    </p>

                    <div className="grid gap-6 md:grid-cols-2">
                      <InfoLine
                        label="Policy reason"
                        value={result.policy?.action || "—"}
                        text={result.policy?.reason || result.message || "—"}
                      />
                      <InfoLine
                        label="Quality"
                        value={result.quality?.status || "—"}
                        text={result.quality?.reason || "—"}
                      />
                      <InfoLine
                        label="Uncertainty"
                        value={formatLabel(result.inference?.uncertainty_method)}
                        text={`MC passes: ${result.inference?.mc_passes ?? "—"} · Ensemble: ${
                          result.inference?.deep_ensemble_enabled ? "Yes" : "No"
                        }`}
                      />
                      <InfoLine
                        label="Explainability"
                        value={result.explainability?.method || "—"}
                        text={`Target: ${
                          result.explainability?.target_label ||
                          result.inference?.top_label ||
                          "—"
                        }`}
                      />
                    </div>

                    {result.inference?.probabilities ? (
                      <div className="mt-7">
                        <p className="mb-3 text-xs uppercase tracking-[0.18em] text-slate-500">
                          Model probabilities
                        </p>
                        <div className="space-y-2 text-sm text-slate-600">
                          {Object.entries(result.inference.probabilities).map(
                            ([label, probability]) => (
                              <div
                                key={label}
                                className="flex items-center justify-between gap-4"
                              >
                                <span>{label}</span>
                                <span className="font-semibold text-slate-900">
                                  {formatPercent(probability)}
                                </span>
                              </div>
                            )
                          )}
                        </div>
                      </div>
                    ) : null}

                    {warnings.length > 0 ? (
                      <div className="mt-7">
                        <p className="mb-3 text-xs uppercase tracking-[0.18em] text-slate-500">
                          Warnings
                        </p>
                        <div className="flex flex-wrap gap-3">
                          {warnings.map((warning, index) => (
                            <span
                              key={`${warning}-${index}`}
                              className="text-xs font-medium text-red-600"
                            >
                              {warning}
                            </span>
                          ))}
                        </div>
                      </div>
                    ) : null}

                    {manualOverrideUsed ? (
                      <div className="mt-7">
                        <p className="text-sm font-semibold text-amber-800">
                          Manual route confirmation was used.
                        </p>
                        <p className="mt-2 text-sm leading-7 text-amber-700">
                          The original route detector result was preserved in the audit metadata.
                        </p>
                      </div>
                    ) : null}
                  </div>
                </details>
              </div>
            )}
          </section>
        </section>

        <SiteFooter />
      </div>
    </main>
  );
}

function getAssistantDecision(action?: string) {
  switch (action) {
    case "ANSWER":
      return "I can show the model output.";
    case "ESCALATE":
      return "This should be reviewed by a human expert.";
    case "REQUEST_EVIDENCE":
      return "I need better evidence before showing a result.";
    case "REFUSE":
      return "Output withheld due to low reliability.";
    case "STOP":
      return "I stopped the review for safety.";
    default:
      return "Ready for demo review.";
  }
}

function getDecisionColor(action?: string) {
  switch (action) {
    case "ANSWER":
      return "text-slate-950";
    case "ESCALATE":
      return "text-amber-700";
    case "REQUEST_EVIDENCE":
      return "text-sky-700";
    case "REFUSE":
    case "STOP":
      return "text-red-700";
    default:
      return "text-slate-950";
  }
}

function getAssistantSummary(result?: AnalysisResponse, selectedRoute?: string) {
  if (!result) {
    return "Run one scenario to see the same guided summary style used on the main page.";
  }

  const action = result.policy?.action || "—";
  const route = formatLabel(selectedRoute);
  const risk = result.policy?.risk_category || "—";

  return `Demo completed with policy action ${action}. Route: ${route}. Risk: ${risk}.`;
}

function InfoLine({
  label,
  value,
  text,
}: {
  label: string;
  value: string;
  text?: string;
}) {
  return (
    <div>
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">
        {label}
      </p>
      <p className="mt-2 text-base font-semibold text-slate-900">{value}</p>
      {text ? <p className="mt-1 text-sm leading-6 text-slate-500">{text}</p> : null}
    </div>
  );
}

function PipelineStep({ label, done }: { label: string; done: boolean }) {
  return (
    <div className="flex items-center gap-3 text-sm text-slate-600">
      <span
        className={`h-2.5 w-2.5 rounded-full ${
          done ? "bg-red-500" : "bg-slate-300"
        }`}
      />
      {label}
    </div>
  );
}

function DecisionLegend() {
  return (
    <div className="mt-8 grid gap-4 text-sm text-slate-500 md:grid-cols-4">
      <p><span className="font-semibold text-slate-900">Answer</span> → output can be shown</p>
      <p><span className="font-semibold text-amber-700">Escalate</span> → expert review needed</p>
      <p><span className="font-semibold text-sky-700">Request evidence</span> → better input needed</p>
      <p><span className="font-semibold text-red-700">Stop</span> → unsafe to continue</p>
    </div>
  );
}

function InfoMini({
  label,
  value,
  helper,
}: {
  label: string;
  value: string;
  helper?: string;
}) {
  return (
    <div>
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">
        {label}
      </p>
      <p className="mt-2 text-base font-semibold text-slate-900">{value}</p>
      {helper ? <p className="mt-1 text-sm leading-6 text-slate-500">{helper}</p> : null}
    </div>
  );
}
