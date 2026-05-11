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
            {!demoResult || !result ? (
              <div className="flex min-h-[430px] flex-col justify-center motion-fade-up-slow">
                <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
                  Review summary
                </p>

                <h3 className="mt-3 text-3xl font-semibold tracking-tight">
                  Ready for demo review.
                </h3>

                <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-500">
                  The result will appear here in the same guided style as the main review page.
                </p>

                <div className="mt-9 grid max-w-2xl gap-7 sm:grid-cols-3">
                  <InfoMini label="Step one" value="Choose case" />
                  <InfoMini label="Step two" value="Run pipeline" />
                  <InfoMini label="Step three" value="Review evidence" />
                </div>
              </div>
            ) : (
              <div className="motion-fade-up-slow">
                <section>
                  <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
                    Guided review result
                  </p>

                  <h3 className="mt-3 text-3xl font-semibold tracking-tight">
                    {result.inference?.top_label || "No inference"}
                  </h3>

                  <p className="mt-3 max-w-2xl text-sm leading-7 text-slate-500">
                    {demoResult.title} · {demoResult.description}
                  </p>

                  <div className="mt-8 grid gap-7 md:grid-cols-4">
                    <InfoMini label="Policy" value={result.policy?.action || "—"} />
                    <InfoMini label="Risk" value={result.policy?.risk_category || "—"} />
                    <InfoMini label="Route" value={formatLabel(selectedRoute)} />
                    <InfoMini
                      label="Accepted"
                      value={result.input_gate?.accepted_for_analysis ? "Yes" : "No"}
                    />
                  </div>
                </section>

                <section className="mt-10 grid gap-8 md:grid-cols-2">
                  <InfoMini
                    label="Model output"
                    value={result.inference?.top_label || "No inference"}
                    helper={`Confidence: ${formatPercent(result.inference?.top_probability)}`}
                  />
                  <InfoMini
                    label="Selected model"
                    value={result.routing?.selected_model || "—"}
                    helper={`Region: ${result.detection?.region || "—"} · Modality: ${
                      result.detection?.modality || "—"
                    }`}
                  />
                  <InfoMini
                    label="Uncertainty"
                    value={formatLabel(result.inference?.uncertainty_method)}
                    helper={`MC passes: ${result.inference?.mc_passes ?? "—"} · Ensemble: ${
                      result.inference?.deep_ensemble_enabled ? "Yes" : "No"
                    }`}
                  />
                  <InfoMini
                    label="OOD check"
                    value={result.ood?.tier || "—"}
                    helper={result.ood?.method || "—"}
                  />
                  <InfoMini
                    label="Quality check"
                    value={result.quality?.status || "—"}
                    helper={result.quality?.reason || "—"}
                  />
                  <InfoMini
                    label="Explanation target"
                    value={
                      result.explainability?.target_label ||
                      result.inference?.top_label ||
                      "—"
                    }
                    helper={`Method: ${result.explainability?.method || "—"}`}
                  />
                </section>

                <section className="mt-10">
                  <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
                    Policy reason
                  </p>
                  <p className="mt-3 text-sm leading-7 text-slate-600">
                    {result.policy?.reason || result.message || "—"}
                  </p>
                </section>

                {manualOverrideUsed ? (
                  <section className="mt-8">
                    <p className="text-sm font-semibold text-amber-800">
                      Manual route confirmation was used.
                    </p>
                    <p className="mt-2 text-sm leading-7 text-amber-700">
                      The original route detector result was preserved in the audit metadata.
                    </p>
                  </section>
                ) : null}

                {warnings.length > 0 ? (
                  <section className="mt-8">
                    <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
                      Warnings
                    </p>
                    <div className="mt-3 flex flex-wrap gap-3">
                      {warnings.map((warning, index) => (
                        <span
                          key={`${warning}-${index}`}
                          className="text-xs font-medium text-red-600"
                        >
                          {warning}
                        </span>
                      ))}
                    </div>
                  </section>
                ) : null}

                <section className="mt-12">
                  <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
                    Visual evidence
                  </p>

                  <div className="mt-5 grid gap-8 lg:grid-cols-2">
                    <div>
                      <p className="mb-3 text-sm font-semibold text-slate-900">
                        Original image
                      </p>
                      <div className="flex min-h-[260px] items-center justify-center py-4">
                        {inputImageUrl ? (
                          <img
                            src={inputImageUrl}
                            alt="Demo input"
                            className="max-h-[320px] w-full object-contain"
                          />
                        ) : (
                          <p className="text-center text-sm text-slate-400">
                            DICOM input preview is unavailable in browser.
                          </p>
                        )}
                      </div>
                    </div>

                    <div>
                      <p className="mb-3 text-sm font-semibold text-slate-900">
                        Where the model focused
                      </p>
                      <div className="flex min-h-[260px] items-center justify-center py-4">
                        {heatmapUrl ? (
                          <img
                            src={heatmapUrl}
                            alt="Demo heatmap"
                            className="max-h-[320px] w-full object-contain"
                          />
                        ) : (
                          <p className="text-center text-sm text-slate-400">
                            Focus map unavailable because inference did not run.
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                </section>

                <details className="mt-10">
                  <summary className="cursor-pointer text-sm font-semibold text-red-600">
                    View technical audit
                  </summary>

                  <div className="mt-5 text-sm leading-7 text-slate-500">
                    {result.inference?.probabilities ? (
                      <div className="space-y-2">
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
                    ) : (
                      <p>No probability table was returned for this demo.</p>
                    )}
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
