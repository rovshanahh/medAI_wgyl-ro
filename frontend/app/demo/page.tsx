"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  CircleHelp,
  House,
  LayoutDashboard,
  Play,
  ShieldPlus,
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
    confidence?: number;
    selected_route?: string;
    manual_override?: boolean;
    top_level_gate?: {
      route_detector?: {
        raw_route_label?: string;
        manual_override?: boolean;
      };
      conversion?: {
        converted?: boolean;
        source_format?: string;
        working_format?: string;
        message?: string;
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
    requires_confirmation?: boolean;
  };
  inference?: {
    top_label?: string;
    top_probability?: number;
    probabilities?: Record<string, number>;
    uncertainty_method?: string;
    deep_ensemble_enabled?: boolean;
    uncertainty_note?: string;
    ensemble_member_count?: number;
    mc_passes?: number;
    calibration?: {
      enabled?: boolean;
      method?: string;
      temperature?: number;
    };
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
    reason?: string;
  };
  policy?: {
    action?: string;
    reason?: string;
    risk_category?: string;
    warnings?: string[];
  };
  warnings?: string[];
  disclaimer?: string;
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

function formatLabel(value?: string | null) {
  if (!value) return "—";
  return value.replaceAll("_", " ");
}

function formatPercent(value?: number | null) {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function getRouteTone(route?: string | null) {
  switch (route) {
    case "abdomen_ct":
      return "bg-cyan-50 text-cyan-700 ring-1 ring-cyan-100";
    case "brain_mri":
      return "bg-indigo-50 text-indigo-700 ring-1 ring-indigo-100";
    case "bone_xray":
      return "bg-sky-50 text-sky-700 ring-1 ring-sky-100";
    case "breast_mammography":
      return "bg-rose-50 text-rose-700 ring-1 ring-rose-100";
    case "chest_xray":
      return "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-100";
    case "retina_fundus":
      return "bg-violet-50 text-violet-700 ring-1 ring-violet-100";
    case "skin_dermoscopy":
      return "bg-orange-50 text-orange-700 ring-1 ring-orange-100";
    case "unknown":
      return "bg-red-50 text-red-700 ring-1 ring-red-100";
    default:
      return "bg-zinc-100 text-zinc-700 ring-1 ring-zinc-200";
  }
}

function getRiskTone(risk?: string) {
  switch (risk?.toLowerCase()) {
    case "low":
      return "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-100";
    case "moderate":
      return "bg-amber-50 text-amber-700 ring-1 ring-amber-100";
    case "high":
      return "bg-red-50 text-red-700 ring-1 ring-red-100";
    default:
      return "bg-zinc-100 text-zinc-700 ring-1 ring-zinc-200";
  }
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
      className="min-h-screen bg-[#FBF8F3] text-zinc-900"
      style={{ fontFamily: '"Aptos","Aptos Body","Segoe UI",Arial,sans-serif' }}
    >
      <div className="mx-auto max-w-7xl px-6 py-6">
        <nav className="mb-10 flex flex-col gap-5 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-white shadow-[0_8px_24px_rgba(0,0,0,0.04)] ring-1 ring-black/5">
              <ShieldPlus size={22} className="text-red-500" />
            </div>

            <div>
              <p className="text-[11px] uppercase tracking-[0.28em] text-zinc-400">
                Demo mode
              </p>
              <h1 className="mt-1 text-3xl font-semibold tracking-tight">
                Governed Medical Image Analysis
              </h1>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-5 text-sm text-zinc-600">
            <Link href="/" className="inline-flex items-center gap-2 hover:text-zinc-900">
              <House size={16} />
              Home
            </Link>

            <Link
              href="/demo"
              className="inline-flex items-center gap-2 font-medium text-zinc-900"
            >
              <Play size={16} />
              Demo
            </Link>

            <Link
              href="/workspace"
              className="inline-flex items-center gap-2 hover:text-zinc-900"
            >
              <LayoutDashboard size={16} />
              Workspace
            </Link>

            <Link
              href="/about"
              className="inline-flex items-center gap-2 hover:text-zinc-900"
            >
              <CircleHelp size={16} />
              About
            </Link>
          </div>
        </nav>

        <section className="mb-10">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full bg-white px-4 py-2 text-xs text-zinc-600 shadow-[0_6px_20px_rgba(0,0,0,0.03)]">
            <Play size={15} className="text-red-500" />
            Presentation-ready scenarios
          </div>

          <h2 className="max-w-3xl text-3xl font-semibold tracking-tight sm:text-4xl">
            Run predefined safety and routing demos without manual file upload.
          </h2>

          <p className="mt-5 max-w-2xl text-base leading-8 text-zinc-600">
            Select a demo case, run it through the same backend pipeline, and show
            route detection, model inference, OOD status, policy decision, and
            Grad-CAM++ focus maps.
          </p>
        </section>

        <section className="grid gap-8 lg:grid-cols-[360px_minmax(0,1fr)]">
          <aside className="rounded-[24px] border border-zinc-200 bg-white p-5 shadow-[0_12px_30px_rgba(0,0,0,0.04)]">
            <h3 className="text-base font-semibold text-zinc-900">
              Demo scenarios
            </h3>

            <p className="mt-2 text-sm leading-6 text-zinc-500">
              These cases use local files from backend/test_samples.
            </p>

            <div className="mt-5 space-y-3">
              {loadingCases ? (
                <p className="text-sm text-zinc-500">Loading demo cases...</p>
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
                    className={`w-full rounded-[18px] border px-4 py-3 text-left transition ${
                      selectedCaseId === item.id
                        ? "border-red-200 bg-red-50/60"
                        : "border-zinc-200 bg-white hover:border-red-100 hover:bg-red-50/20"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <p className="text-sm font-semibold text-zinc-900">
                        {item.title}
                      </p>

                      {item.route_override ? (
                        <span className="rounded-md bg-amber-50 px-2 py-1 text-[10px] font-medium text-amber-700 ring-1 ring-amber-100">
                          Override
                        </span>
                      ) : null}
                    </div>

                    <p className="mt-2 text-xs leading-5 text-zinc-500">
                      {item.description}
                    </p>
                  </button>
                ))
              )}
            </div>

            {selectedCase ? (
              <div className="mt-5 rounded-[18px] bg-zinc-50 p-4 text-sm">
                <p className="font-medium text-zinc-900">Expected behavior</p>
                <p className="mt-2 leading-6 text-zinc-600">
                  {selectedCase.expected}
                </p>
              </div>
            ) : null}

            <button
              type="button"
              onClick={handleRunDemo}
              disabled={running || loadingCases}
              className="mt-5 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-red-500 px-5 py-3 text-sm font-medium text-white shadow-[0_12px_28px_rgba(239,68,68,0.22)] transition hover:bg-red-600 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {running ? "Running demo..." : "Run selected demo"}
              <ArrowRight size={16} />
            </button>

            {error ? (
              <div className="mt-4 rounded-[18px] bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-100">
                {error}
              </div>
            ) : null}
          </aside>

          <section>
            {!demoResult || !result ? (
              <div className="flex min-h-[420px] items-center justify-center rounded-[24px] border border-zinc-200 bg-white text-zinc-400">
                Run a demo case to see the governed review summary
              </div>
            ) : (
              <div className="grid gap-8 xl:grid-cols-[minmax(0,1fr)_320px]">
                <div className="rounded-[24px] border border-zinc-200 bg-white p-6 shadow-[0_12px_30px_rgba(0,0,0,0.04)]">
                  <div className="border-b border-zinc-200 pb-5">
                    <p className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                      Demo result
                    </p>
                    <h3 className="mt-2 text-3xl font-semibold tracking-tight">
                      {demoResult.title}
                    </h3>
                    <p className="mt-2 text-sm leading-6 text-zinc-500">
                      {demoResult.description}
                    </p>
                  </div>

                  <div className="mt-6 grid gap-5 md:grid-cols-4">
                    <div>
                      <p className="text-xs uppercase tracking-[0.16em] text-zinc-400">
                        Policy
                      </p>
                      <p className="mt-2 font-semibold text-zinc-900">
                        {result.policy?.action || "—"}
                      </p>
                    </div>

                    <div>
                      <p className="text-xs uppercase tracking-[0.16em] text-zinc-400">
                        Risk
                      </p>
                      <span
                        className={`mt-2 inline-flex rounded-md px-2.5 py-1 text-xs font-medium ${getRiskTone(
                          result.policy?.risk_category
                        )}`}
                      >
                        {result.policy?.risk_category || "—"}
                      </span>
                    </div>

                    <div>
                      <p className="text-xs uppercase tracking-[0.16em] text-zinc-400">
                        Route
                      </p>
                      <span
                        className={`mt-2 inline-flex rounded-md px-2.5 py-1 text-xs font-medium ${getRouteTone(
                          selectedRoute
                        )}`}
                      >
                        {formatLabel(selectedRoute)}
                      </span>
                    </div>

                    <div>
                      <p className="text-xs uppercase tracking-[0.16em] text-zinc-400">
                        Accepted
                      </p>
                      <p className="mt-2 font-semibold text-zinc-900">
                        {result.input_gate?.accepted_for_analysis ? "Yes" : "No"}
                      </p>
                    </div>
                  </div>

                  {manualOverrideUsed ? (
                    <div className="mt-6 rounded-[18px] bg-amber-50 px-4 py-4 text-sm text-amber-800 ring-1 ring-amber-100">
                      <p className="font-medium">Manual route confirmation was used</p>
                      <p className="mt-2 leading-6">
                        The original route detector result was preserved in the audit metadata.
                      </p>
                    </div>
                  ) : null}

                  <div className="mt-8 grid gap-6 md:grid-cols-2">
                    <div>
                      <p className="text-xs uppercase tracking-[0.16em] text-zinc-400">
                        Model output
                      </p>
                      <p className="mt-2 text-xl font-semibold text-zinc-900">
                        {result.inference?.top_label || "No inference"}
                      </p>
                      <p className="mt-1 text-sm text-zinc-500">
                        Confidence: {formatPercent(result.inference?.top_probability)}
                      </p>
                    </div>

                    <div>
                      <p className="text-xs uppercase tracking-[0.16em] text-zinc-400">
                        Selected model
                      </p>
                      <p className="mt-2 text-sm font-medium text-zinc-900">
                        {result.routing?.selected_model || "—"}
                      </p>
                      <p className="mt-1 text-sm text-zinc-500">
                        Region: {result.detection?.region || "—"} · Modality:{" "}
                        {result.detection?.modality || "—"}
                      </p>
                    </div>

                    <div>
                      <p className="text-xs uppercase tracking-[0.16em] text-zinc-400">
                        Uncertainty
                      </p>
                      <p className="mt-2 text-sm font-medium text-zinc-900">
                        {formatLabel(result.inference?.uncertainty_method)}
                      </p>
                      <p className="mt-1 text-sm text-zinc-500">
                        MC passes: {result.inference?.mc_passes ?? "—"} · Ensemble:{" "}
                        {result.inference?.deep_ensemble_enabled ? "Yes" : "No"}
                      </p>
                    </div>

                    <div>
                      <p className="text-xs uppercase tracking-[0.16em] text-zinc-400">
                        Calibration
                      </p>
                      <p className="mt-2 text-sm font-medium text-zinc-900">
                        {result.inference?.calibration?.enabled ? "Enabled" : "Not applied"}
                      </p>
                      <p className="mt-1 text-sm text-zinc-500">
                        Temperature: {result.inference?.calibration?.temperature ?? "—"}
                      </p>
                    </div>

                    <div>
                      <p className="text-xs uppercase tracking-[0.16em] text-zinc-400">
                        Quality check
                      </p>
                      <p className="mt-2 text-sm font-medium text-zinc-900">
                        {result.quality?.status || "—"}
                      </p>
                      <p className="mt-1 text-sm leading-6 text-zinc-500">
                        {result.quality?.reason || "—"}
                      </p>
                    </div>

                    <div>
                      <p className="text-xs uppercase tracking-[0.16em] text-zinc-400">
                        OOD check
                      </p>
                      <p className="mt-2 text-sm font-medium text-zinc-900">
                        {result.ood?.tier || "—"}
                      </p>
                      <p className="mt-1 text-sm leading-6 text-zinc-500">
                        {result.ood?.method || "—"}
                      </p>
                    </div>
                  </div>

                  <div className="mt-8 border-t border-zinc-200 pt-5">
                    <p className="text-xs uppercase tracking-[0.16em] text-zinc-400">
                      Policy reason
                    </p>
                    <p className="mt-3 text-sm leading-7 text-zinc-700">
                      {result.policy?.reason || result.message || "—"}
                    </p>
                  </div>

                  {warnings.length > 0 ? (
                    <div className="mt-6">
                      <p className="text-xs uppercase tracking-[0.16em] text-zinc-400">
                        Warnings
                      </p>
                      <div className="mt-3 flex flex-wrap gap-2">
                        {warnings.map((warning, index) => (
                          <span
                            key={`${warning}-${index}`}
                            className="rounded-md bg-red-50 px-3 py-1.5 text-xs text-red-700 ring-1 ring-red-100"
                          >
                            {warning}
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : null}

                  {result.inference?.probabilities ? (
                    <div className="mt-8 border-t border-zinc-200 pt-5">
                      <p className="text-xs uppercase tracking-[0.16em] text-zinc-400">
                        Model probabilities
                      </p>

                      <div className="mt-4 divide-y divide-zinc-200 border-y border-zinc-200">
                        {Object.entries(result.inference.probabilities).map(
                          ([label, probability]) => (
                            <div
                              key={label}
                              className="flex items-center justify-between gap-4 py-3 text-sm"
                            >
                              <span className="text-zinc-700">{label}</span>
                              <span className="font-medium text-zinc-900">
                                {formatPercent(probability)}
                              </span>
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  ) : null}
                </div>

                <aside className="space-y-6">
                  <div>
                    <p className="mb-4 text-xs uppercase tracking-[0.18em] text-zinc-400">
                      Demo input image
                    </p>

                    {inputImageUrl ? (
                      <div className="overflow-hidden rounded-[18px] border border-zinc-200 bg-white shadow-[0_12px_30px_rgba(0,0,0,0.04)]">
                        <img
                          src={inputImageUrl}
                          alt="Demo input"
                          className="h-auto w-full object-cover"
                        />
                      </div>
                    ) : (
                      <div className="flex min-h-[220px] items-center justify-center rounded-[24px] border border-zinc-200 bg-white px-5 text-center text-sm text-zinc-400">
                        DICOM input preview is unavailable in browser.
                      </div>
                    )}
                  </div>

                  <div>
                    <p className="mb-4 text-xs uppercase tracking-[0.18em] text-zinc-400">
                      Model focus map
                    </p>

                    {heatmapUrl ? (
                      <div className="overflow-hidden rounded-[18px] border border-zinc-200 bg-white shadow-[0_12px_30px_rgba(0,0,0,0.04)]">
                        <img
                          src={heatmapUrl}
                          alt="Demo heatmap"
                          className="h-auto w-full object-cover"
                        />
                      </div>
                    ) : (
                      <div className="flex min-h-[220px] items-center justify-center rounded-[24px] border border-zinc-200 bg-white px-5 text-center text-sm text-zinc-400">
                        Focus map unavailable because inference did not run.
                      </div>
                    )}
                  </div>

                  <div className="rounded-[18px] border border-zinc-200 bg-white p-4 text-sm text-zinc-600">
                    <div className="flex justify-between gap-4 border-b border-zinc-200 py-2">
                      <span>Method</span>
                      <span className="font-medium text-zinc-900">
                        {result.explainability?.method || "—"}
                      </span>
                    </div>
                    <div className="flex justify-between gap-4 py-2">
                      <span>Target</span>
                      <span className="font-medium text-zinc-900">
                        {result.explainability?.target_label ||
                          result.inference?.top_label ||
                          "—"}
                      </span>
                    </div>
                  </div>
                </aside>
              </div>
            )}
          </section>
        </section>
      </div>
    </main>
  );
}
