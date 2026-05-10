"use client";

import { ChangeEvent, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  CircleHelp,
  House,
  LayoutDashboard,
  Mail,
  MapPin,
  Phone,
  ShieldPlus,
} from "lucide-react";
import Link from "next/link";

type AppConfig = {
  supported_uploads?: string[];
  active_routes?: {
    route: string;
    region: string;
    modality: string;
    model: string;
    description: string;
    status: string;
  }[];
  safety_routes?: {
    route: string;
    description: string;
    status: string;
  }[];
  inactive_placeholders?: {
    route: string;
    region?: string;
    modality?: string;
    description?: string;
    status?: string;
  }[];
  max_batch_size?: number;
  disclaimer?: string;
};

type AnalysisResponse = {
  analysis_id?: string;
  filename?: string;
  input_gate?: {
    accepted_for_analysis?: boolean;
    confidence?: number;
    message?: string;
    selected_route?: string;
    route_scores?: Record<string, number>;
    manual_override?: boolean;
    top_level_gate?: {
      route_detector?: {
        route_label?: string;
        raw_route_label?: string;
        confidence?: number;
        margin?: number;
        supported?: boolean;
        requires_confirmation?: boolean;
        probabilities?: Record<string, number>;
        reason?: string;
        manual_override?: boolean;
        override_metadata?: {
          requested_route?: string;
          original_route_result?: Record<string, unknown>;
        };
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
    confidence?: number;
    requires_confirmation?: boolean;
    supported?: boolean;
    reason?: string;
    manual_override?: boolean;
  };
  inference?: {
    top_label?: string;
    top_probability?: number;
    positive_findings?: string[];
    probabilities?: Record<string, number>;
    reliability_score?: number;
    disagreement_score?: number;
  };
  explainability?: {
    method?: string | null;
    heatmap_path?: string | null;
    warning?: string;
    target_label?: string;
  };
  quality?: {
    status?: string | null;
    warnings?: string[];
    requires_reupload?: boolean | null;
    blocking?: boolean;
    reason?: string;
    metrics?: Record<string, number>;
  };
  routing?: {
    accepted_findings_set?: string[];
    selected_model?: string | null;
    set_size?: number;
    requires_confirmation?: boolean;
    top_probability?: number;
    routing_candidates?: string[];
    alpha?: number | null;
    threshold?: number | null;
    nonconformity?: number | null;
    method?: string;
    reason?: string;
    routing_candidate_details?: {
      route: string;
      probability: number;
      nonconformity: number;
    }[];
  };
  ood?: {
    tier?: string;
    score?: number | null;
    reason?: string;
    method?: string;
    metrics?: Record<string, number | boolean>;
    is_hard_ood?: boolean;
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

type RecentReview = {
  id: string;
  filename: string;
  route: string;
  policy: string;
  output: string;
  confidence?: number | null;
  result: AnalysisResponse;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

const HEATMAP_BASE_URL = API_BASE_URL;

const FALLBACK_ROUTES = [
  {
    route: "brain_mri",
    region: "brain",
    modality: "mri",
    model: "brain_mri_resnet18",
    description:
      "Reviews brain MRI images and returns the most likely tumor-related class.",
    status: "ACTIVE",
  },
  {
    route: "bone_xray",
    region: "bone",
    modality: "xray",
    model: "bone_xray_standard",
    description:
      "Reviews bone X-ray images and separates normal from abnormal cases.",
    status: "ACTIVE",
  },
  {
    route: "breast_mammography",
    region: "breast",
    modality: "mammography",
    model: "breast_mammography_resnet18",
    description:
      "Reviews mammography images and separates benign from malignant cases.",
    status: "ACTIVE",
  },
  {
    route: "chest_xray",
    region: "chest",
    modality: "xray",
    model: "chest_xray_mvp",
    description:
      "Reviews chest X-ray images and highlights possible visible findings.",
    status: "ACTIVE",
  },
  {
    route: "retina_fundus",
    region: "retina",
    modality: "fundus",
    model: "retina_fundus_resnet18",
    description:
      "Reviews eye fundus images and estimates diabetic retinopathy severity.",
    status: "ACTIVE",
  },
  {
    route: "skin_dermoscopy",
    region: "skin",
    modality: "dermoscopy",
    model: "skin_dermoscopy_resnet18",
    description:
      "Reviews skin dermoscopy images and returns the most likely lesion class.",
    status: "ACTIVE",
  },
];

function formatPercent(value?: number | null) {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value?: number | null) {
  if (value == null) return "—";
  return value.toFixed(3);
}

function formatLabel(value?: string | null) {
  if (!value) return "—";
  return value.replaceAll("_", " ");
}

function getRouteTitle(route?: string | null) {
  switch (route) {
    case "brain_mri":
      return "Brain MRI review";
    case "bone_xray":
      return "Bone X-ray review";
    case "breast_mammography":
      return "Breast mammography review";
    case "chest_xray":
      return "Chest X-ray review";
    case "retina_fundus":
      return "Retina fundus review";
    case "skin_dermoscopy":
      return "Skin dermoscopy review";
    case "abdomen_ct":
      return "Abdomen CT review";
    case "unknown":
      return "Unsupported or uncertain input";
    default:
      return "Medical image review";
  }
}

function getRouteCardTitle(route?: string | null) {
  return getRouteTitle(route).replace(" review", "");
}

function getOutputLabel(route?: string | null) {
  switch (route) {
    case "brain_mri":
      return "Model output";
    case "bone_xray":
    case "breast_mammography":
    case "abdomen_ct":
      return "Classification output";
    case "chest_xray":
      return "Primary finding";
    case "retina_fundus":
      return "Severity output";
    case "skin_dermoscopy":
      return "Lesion class output";
    default:
      return "Model output";
  }
}

function getPolicyDisplay(action?: string) {
  switch (action) {
    case "ANSWER":
      return {
        title: "Answer allowed",
        description: "The system found enough support to show the model output.",
      };
    case "ESCALATE":
      return {
        title: "Human review recommended",
        description:
          "The model produced an output, but confidence, uncertainty, or routing ambiguity suggests expert review.",
      };
    case "REQUEST_EVIDENCE":
      return {
        title: "More evidence needed",
        description:
          "The uploaded image may not provide enough reliable information for automatic review.",
      };
    case "REFUSE":
      return {
        title: "Review refused",
        description:
          "The system refused to provide an output because safety checks were not satisfied.",
      };
    case "STOP":
      return {
        title: "Stopped before inference",
        description:
          "The system stopped the request before model inference because the input was unsupported or uncertain.",
      };
    default:
      return {
        title: "Review pending",
        description: "No final policy decision is available yet.",
      };
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

function getInputTone(accepted?: boolean) {
  if (accepted === true) {
    return "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-100";
  }

  if (accepted === false) {
    return "bg-red-50 text-red-700 ring-1 ring-red-100";
  }

  return "bg-zinc-100 text-zinc-700 ring-1 ring-zinc-200";
}

function getRouteTone(route?: string | null) {
  switch (route) {
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
    case "abdomen_ct":
      return "bg-cyan-50 text-cyan-700 ring-1 ring-cyan-100";
    case "unknown":
      return "bg-red-50 text-red-700 ring-1 ring-red-100";
    default:
      return "bg-zinc-100 text-zinc-700 ring-1 ring-zinc-200";
  }
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [recentReviews, setRecentReviews] = useState<RecentReview[]>([]);
  const [appConfig, setAppConfig] = useState<AppConfig | null>(null);
  const [confirmedRoute, setConfirmedRoute] = useState("");
  const [loading, setLoading] = useState(false);
  const [overrideLoading, setOverrideLoading] = useState(false);
  const [error, setError] = useState("");

  const isDicomFile = file?.name.toLowerCase().endsWith(".dcm") ?? false;

  const previewUrl = useMemo(() => {
    if (!file) return "";
    return URL.createObjectURL(file);
  }, [file]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    const loadConfig = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/config`);
        if (!response.ok) return;

        const data = await response.json();
        setAppConfig(data);
      } catch {
        setAppConfig(null);
      }
    };

    loadConfig();
  }, []);

  const heatmapUrl = useMemo(() => {
    const rawPath = result?.explainability?.heatmap_path;

    if (!rawPath) return "";

    if (rawPath.startsWith("http://") || rawPath.startsWith("https://")) {
      return rawPath;
    }

    return `${HEATMAP_BASE_URL}${rawPath}`;
  }, [result]);

  const activeRoutes = appConfig?.active_routes?.length
    ? appConfig.active_routes
    : FALLBACK_ROUTES;

  const supportedUploads = appConfig?.supported_uploads?.length
    ? appConfig.supported_uploads.join(", ").replaceAll(".", "").toUpperCase()
    : "PNG, JPG, TIFF, DICOM";

  const routeDetector = result?.input_gate?.top_level_gate?.route_detector;
  const conversion = result?.input_gate?.top_level_gate?.conversion;
  const selectedRoute =
    result?.input_gate?.selected_route || routeDetector?.route_label;
  const routeProbabilities =
    routeDetector?.probabilities || result?.input_gate?.route_scores || {};
  const inferenceRan = Boolean(result?.inference?.top_label);
  const policyDisplay = getPolicyDisplay(result?.policy?.action);

  const manualOverrideUsed =
    Boolean(result?.input_gate?.manual_override) ||
    Boolean(result?.detection?.manual_override) ||
    Boolean(routeDetector?.manual_override);

  const needsManualConfirmation =
    Boolean(result) &&
    !manualOverrideUsed &&
    (result?.policy?.action === "STOP" ||
      result?.policy?.action === "ESCALATE" ||
      result?.routing?.requires_confirmation === true ||
      result?.detection?.requires_confirmation === true ||
      result?.input_gate?.accepted_for_analysis === false);

  const warnings = [
    ...(result?.policy?.warnings ?? []),
    ...(result?.quality?.warnings ?? []),
    ...(result?.warnings ?? []),
    ...(result?.explainability?.warning ? [result.explainability.warning] : []),
  ].filter(Boolean);

  const addRecentReview = (data: AnalysisResponse, sourceFile: File) => {
    const route =
      data?.input_gate?.selected_route ||
      data?.input_gate?.top_level_gate?.route_detector?.route_label ||
      "unknown";

    const review: RecentReview = {
      id: `${Date.now()}-${sourceFile.name}`,
      filename: sourceFile.name,
      route,
      policy: data?.policy?.action || "—",
      output: data?.inference?.top_label || "No inference",
      confidence: data?.inference?.top_probability,
      result: data,
    };

    setRecentReviews((previous) => [review, ...previous].slice(0, 5));
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0] ?? null;
    setFile(selected);
    setResult(null);
    setConfirmedRoute("");
    setError("");
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError("Please choose an image first.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      setResult(null);
      setConfirmedRoute("");

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      const data: AnalysisResponse & { detail?: string } = await response.json();

      if (!response.ok) {
        throw new Error(data?.detail || "Request failed.");
      }

      setResult(data);
      addRecentReview(data, file);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const handleRunWithConfirmedRoute = async () => {
    if (!file) {
      setError("Please choose an image first.");
      return;
    }
  
    if (!confirmedRoute) {
      setError("Please select a confirmed route first.");
      return;
    }
  
    try {
      setOverrideLoading(true);
      setError("");
  
      console.log("Running override with route:", confirmedRoute);
  
      const formData = new FormData();
      formData.append("file", file);
      formData.append("route_override", confirmedRoute);
  
      const response = await fetch(`${API_BASE_URL}/analyze/override`, {
        method: "POST",
        body: formData,
      });
  
      const text = await response.text();
      console.log("Override raw response:", text);
  
      let data: AnalysisResponse & { detail?: string };
  
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error(text || "Override request failed.");
      }
  
      if (!response.ok) {
        throw new Error(data?.detail || "Override request failed.");
      }
  
      setResult(data);
      addRecentReview(data, file);
      setConfirmedRoute("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setOverrideLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setConfirmedRoute("");
    setError("");
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
              <div className="relative h-4 w-7">
                <span className="absolute left-0 top-1/2 h-2 w-2 -translate-y-1/2 rounded-full bg-red-500" />
                <span className="absolute right-0 top-1/2 h-2 w-2 -translate-y-1/2 rounded-full bg-red-500" />
                <span className="absolute left-1/2 top-1/2 h-[2px] w-4 -translate-x-1/2 -translate-y-1/2 bg-red-400" />
              </div>
            </div>

            <div>
              <p className="text-[11px] uppercase tracking-[0.28em] text-zinc-400">
                Research-use assistant
              </p>
              <h1 className="mt-1 text-3xl font-semibold tracking-tight">
                MedAIx
              </h1>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-5 text-sm text-zinc-600">
            <Link
              href="/"
              className="inline-flex items-center gap-2 transition hover:text-zinc-900"
            >
              <House size={16} />
              Home
            </Link>

            <Link
              href="/workspace"
              className="inline-flex items-center gap-2 transition hover:text-zinc-900"
            >
              <LayoutDashboard size={16} />
              Workspace
            </Link>

            <Link
              href="/about"
              className="inline-flex items-center gap-2 transition hover:text-zinc-900"
            >
              <CircleHelp size={16} />
              About
            </Link>

            <Link
              href="/contact"
              className="inline-flex items-center gap-2 transition hover:text-zinc-900"
            >
              <Mail size={16} />
              Contact
            </Link>
          </div>
        </nav>

        <section className="mb-14">
          <div>
            <div className="mb-4 inline-flex items-center gap-2 rounded-full bg-white px-4 py-2 text-xs text-zinc-600 shadow-[0_6px_20px_rgba(0,0,0,0.03)]">
              <ShieldPlus size={15} className="text-red-500" />
              Research-use image review assistant
            </div>

            <h2 className="max-w-3xl text-3xl font-semibold tracking-tight sm:text-4xl">
              A careful assistant for reviewing medical images.
            </h2>

            <p className="mt-5 max-w-2xl text-base leading-8 text-zinc-600">
              Upload a supported medical image to receive a structured summary,
              visual explanation, and a carefully framed next-step recommendation
              for research use.
            </p>

            <div className="mt-10 grid gap-4 lg:grid-cols-2 2xl:grid-cols-3">
              {activeRoutes.map((routeInfo) => (
                <div
                  key={routeInfo.route}
                  className="group relative overflow-hidden rounded-[24px] border border-zinc-200/80 bg-white px-5 py-4 shadow-[0_12px_30px_rgba(0,0,0,0.04)] transition hover:-translate-y-0.5 hover:border-red-100 hover:shadow-[0_18px_42px_rgba(239,68,68,0.09)]"
                >
                  <div className="absolute -right-14 -top-16 h-32 w-32 rounded-full bg-red-50/70 blur-md transition group-hover:scale-125" />
                  <div className="absolute -bottom-16 -left-16 h-32 w-32 rounded-full bg-zinc-50 blur-md" />

                  <div className="relative flex items-center justify-between gap-6">
                    <div className="min-w-0">
                      <p className="text-sm font-semibold tracking-tight text-zinc-900">
                        {getRouteCardTitle(routeInfo.route)}
                      </p>

                      <p className="mt-2 text-xs leading-5 text-zinc-500">
                        {routeInfo.description}
                      </p>
                    </div>

                    <div className="hidden shrink-0 text-right sm:block">
                      <p className="text-[10px] uppercase tracking-[0.18em] text-zinc-400">
                        Route
                      </p>
                      <p className="mt-1 text-xs font-medium text-zinc-700">
                        {formatLabel(routeInfo.route)}
                      </p>
                    </div>
                  </div>

                  <div className="relative mt-4 h-px w-full bg-zinc-100">
                    <div className="h-px w-16 bg-gradient-to-r from-red-200 via-red-300 to-transparent transition-all group-hover:w-28" />
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-8 flex gap-3">
              <button
                onClick={handleAnalyze}
                disabled={loading}
                className="inline-flex items-center gap-2 rounded-xl bg-red-500 px-5 py-3 text-sm font-medium text-white shadow-[0_12px_28px_rgba(239,68,68,0.22)] transition hover:bg-red-600 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? "Reviewing..." : "Start review"}
                <ArrowRight size={16} />
              </button>

              <button
                onClick={handleReset}
                className="rounded-xl bg-white px-5 py-3 text-sm text-zinc-700 shadow-[0_8px_24px_rgba(0,0,0,0.04)] transition hover:bg-zinc-50"
              >
                Reset
              </button>
            </div>
          </div>
        </section>

        <section
          id="workspace"
          className="grid gap-12 xl:grid-cols-[300px_minmax(0,1fr)]"
        >
          <aside>
            <h3 className="mb-4 text-base font-medium">Upload image</h3>

            <label className="flex min-h-[210px] cursor-pointer flex-col items-center justify-center rounded-[24px] border border-dashed border-zinc-300 bg-white p-6 text-center transition hover:border-red-300 hover:bg-red-50/20">
              <div className="mb-3 flex h-14 w-14 items-center justify-center rounded-full bg-red-50 ring-1 ring-red-100">
                <div className="h-5 w-5 rounded-full bg-red-500" />
              </div>

              <span className="text-sm text-zinc-700">
                Select a medical image to begin review
              </span>

              <span className="mt-2 text-xs text-zinc-500">
                Supported: {supportedUploads}
              </span>

              <input
                type="file"
                accept=".png,.jpg,.jpeg,.tif,.tiff,.dcm"
                className="hidden"
                onChange={handleFileChange}
              />
            </label>

            <div className="mt-4 text-sm text-zinc-600">
              <div className="flex items-center justify-between border-b border-zinc-200 py-3">
                <span>File</span>
                <span className="max-w-[150px] truncate text-right text-zinc-800">
                  {file?.name || "No file selected"}
                </span>
              </div>
            </div>

            {previewUrl && !isDicomFile ? (
              <div className="mt-4 overflow-hidden rounded-[14px] border border-zinc-200 bg-white">
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="h-auto w-full object-cover"
                />
              </div>
            ) : isDicomFile ? (
              <div className="mt-4 flex h-64 flex-col items-center justify-center rounded-[24px] border border-zinc-200 bg-white px-6 text-center text-sm text-zinc-500">
                <p className="font-medium text-zinc-700">DICOM file selected</p>
                <p className="mt-2">
                  Browser preview is unavailable for .dcm files. The backend will
                  convert it for analysis.
                </p>
              </div>
            ) : (
              <div className="mt-4 flex h-64 items-center justify-center rounded-[24px] border border-zinc-200 bg-white text-sm text-zinc-400">
                Image preview
              </div>
            )}

            {needsManualConfirmation ? (
              <div className="mt-5 rounded-[20px] border border-amber-200 bg-amber-50 p-4">
                <p className="text-sm font-semibold text-amber-900">
                  Manual route confirmation
                </p>

                <p className="mt-2 text-xs leading-5 text-amber-800">
                  The system did not safely continue with automatic routing.
                  Select the correct route only if a human evaluator knows the
                  intended image type.
                </p>

                <select
                  value={confirmedRoute}
                  onChange={(event) => setConfirmedRoute(event.target.value)}
                  className="mt-4 w-full rounded-xl border border-amber-200 bg-white px-3 py-2 text-sm text-zinc-700 outline-none transition focus:border-amber-400"
                >
                  <option value="">Select confirmed route</option>
                  {activeRoutes.map((routeInfo) => (
                    <option key={routeInfo.route} value={routeInfo.route}>
                      {formatLabel(routeInfo.route)}
                    </option>
                  ))}
                </select>

                <button
                  type="button"
                  onClick={handleRunWithConfirmedRoute}
                  disabled={overrideLoading}
                  className="mt-4 w-full rounded-xl bg-amber-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-amber-700 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {overrideLoading
                    ? "Running confirmed route..."
                    : "Run again with confirmed route"}
                </button>
              </div>
            ) : null}

            {error ? (
              <div className="mt-4 rounded-[18px] bg-red-50 px-4 py-3 text-sm text-red-700 ring-1 ring-red-100">
                {error}
              </div>
            ) : null}

            {recentReviews.length > 0 ? (
              <div className="mt-6">
                <h3 className="mb-3 text-base font-medium">Recent reviews</h3>

                <div className="space-y-3">
                  {recentReviews.map((review) => (
                    <button
                      key={review.id}
                      onClick={() => {
                        setResult(review.result);
                        setConfirmedRoute("");
                      }}
                      className="w-full rounded-[18px] border border-zinc-200 bg-white px-4 py-3 text-left text-sm transition hover:border-red-100 hover:bg-red-50/20"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span className="max-w-[150px] truncate font-medium text-zinc-900">
                          {review.filename}
                        </span>
                        <span
                          className={`shrink-0 rounded-md px-2 py-1 text-[11px] font-medium ${getRouteTone(
                            review.route
                          )}`}
                        >
                          {formatLabel(review.route)}
                        </span>
                      </div>

                      <div className="mt-2 flex items-center justify-between gap-3 text-xs text-zinc-500">
                        <span>{review.policy}</span>
                        <span className="max-w-[130px] truncate">
                          {review.output}
                        </span>
                      </div>

                      <div className="mt-1 text-xs text-zinc-400">
                        Confidence: {formatPercent(review.confidence)}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
          </aside>

          <section>
            <div className="mb-8 border-b border-zinc-200 pb-6">
              <h3 className="text-3xl font-semibold tracking-tight">
                {result ? getRouteTitle(selectedRoute) : "Review summary"}
              </h3>
              <p className="mt-2 text-sm text-zinc-500">
                A calm, structured summary for careful human review
              </p>
            </div>

            {!result ? (
              <div className="flex min-h-[280px] items-center justify-center rounded-[24px] border border-zinc-200 bg-white text-zinc-400">
                Begin review to see the guided summary
              </div>
            ) : (
              <div className="grid gap-10 xl:grid-cols-[minmax(0,1fr)_300px]">
                <div>
                  <div className="grid gap-6 border-b border-zinc-200 pb-6 md:grid-cols-[1fr_auto_auto_auto] md:items-end">
                    <div>
                      <p className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                        Recommended next step
                      </p>
                      <p className="mt-3 text-[1.25rem] font-semibold leading-snug text-zinc-900">
                        {policyDisplay.title}
                      </p>
                      <p className="mt-2 max-w-xs text-sm leading-6 text-zinc-500">
                        {policyDisplay.description}
                      </p>
                    </div>

                    <div>
                      <p className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                        Risk level
                      </p>
                      <div className="mt-3">
                        <span
                          className={`inline-flex rounded-md px-3 py-1.5 text-xs font-medium ${getRiskTone(
                            result.policy?.risk_category
                          )}`}
                        >
                          {result.policy?.risk_category || "—"}
                        </span>
                      </div>
                    </div>

                    <div>
                      <p className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                        Image check
                      </p>
                      <div className="mt-3">
                        <span
                          className={`inline-flex rounded-md px-3 py-1.5 text-xs font-medium ${getInputTone(
                            result.input_gate?.accepted_for_analysis
                          )}`}
                        >
                          {result.input_gate?.accepted_for_analysis === true
                            ? "Accepted"
                            : result.input_gate?.accepted_for_analysis === false
                              ? "Rejected"
                              : "—"}
                        </span>
                      </div>
                    </div>

                    <div>
                      <p className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                        Route
                      </p>
                      <div className="mt-3">
                        <span
                          className={`inline-flex rounded-md px-3 py-1.5 text-xs font-medium capitalize ${getRouteTone(
                            selectedRoute
                          )}`}
                        >
                          {formatLabel(selectedRoute)}
                        </span>
                      </div>
                    </div>
                  </div>

                  {manualOverrideUsed ? (
                    <div className="mt-8 rounded-[18px] bg-amber-50 px-4 py-4 text-sm text-amber-800 ring-1 ring-amber-100">
                      <p className="font-medium">Manual override was used</p>
                      <p className="mt-2 leading-6">
                        The automatic detector result was overridden by a human
                        selected route. This should be reviewed carefully.
                      </p>
                    </div>
                  ) : null}

                  <div className="mt-10 border-t border-zinc-200 pt-6">
                    <p className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                      Why this was suggested
                    </p>
                    <p className="mt-4 text-base leading-7 text-zinc-700">
                      {result.policy?.reason ||
                        policyDisplay.description ||
                        result.input_gate?.message ||
                        "—"}
                    </p>
                  </div>

                  {conversion?.converted ? (
                    <div className="mt-10 rounded-[18px] bg-white px-4 py-4 text-sm text-zinc-700 ring-1 ring-zinc-200">
                      <p className="font-medium text-zinc-900">
                        Format conversion
                      </p>
                      <p className="mt-2">
                        {conversion.message || "DICOM was converted for analysis."}
                      </p>
                      <p className="mt-2 text-zinc-500">
                        {conversion.source_format} → {conversion.working_format}
                      </p>
                    </div>
                  ) : null}

                  <div className="mt-10 grid gap-8 md:grid-cols-2">
                    <div>
                      <p className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                        {getOutputLabel(selectedRoute)}
                      </p>
                      <p className="mt-3 break-words text-[1.15rem] font-semibold leading-snug text-zinc-900">
                        {inferenceRan
                          ? result.inference?.top_label
                          : "No inference was run"}
                      </p>
                    </div>

                    <div>
                      <p className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                        Output confidence
                      </p>
                      <p className="mt-3 text-[1.15rem] font-semibold leading-snug text-zinc-900">
                        {inferenceRan
                          ? formatPercent(result.inference?.top_probability)
                          : "—"}
                      </p>
                    </div>
                  </div>

                  <div className="mt-10">
                    <p className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                      Findings considered
                    </p>

                    <div className="mt-4 divide-y divide-zinc-200 border-b border-t border-zinc-200">
                      {result.routing?.accepted_findings_set?.length ? (
                        result.routing.accepted_findings_set.map((item: string) => (
                          <div key={item} className="py-3 text-sm text-zinc-700">
                            {formatLabel(item)}
                          </div>
                        ))
                      ) : (
                        <div className="py-3 text-sm text-zinc-500">
                          No findings listed for this review
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="mt-10">
                    <p className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                      Route detector probabilities
                    </p>

                    <div className="mt-4 divide-y divide-zinc-200 border-b border-t border-zinc-200">
                      {Object.keys(routeProbabilities).length ? (
                        Object.entries(routeProbabilities).map(
                          ([label, probability]) => (
                            <div
                              key={label}
                              className="flex items-center justify-between gap-4 py-3 text-sm text-zinc-700"
                            >
                              <span className="capitalize">
                                {formatLabel(label)}
                              </span>
                              <span className="font-medium text-zinc-900">
                                {formatPercent(probability)}
                              </span>
                            </div>
                          )
                        )
                      ) : (
                        <div className="py-3 text-sm text-zinc-500">
                          No route probabilities available
                        </div>
                      )}
                    </div>
                  </div>

                  {result.routing?.routing_candidate_details?.length ? (
                    <div className="mt-10">
                      <p className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                        Conformal routing candidates
                      </p>

                      <div className="mt-4 divide-y divide-zinc-200 border-b border-t border-zinc-200">
                        {result.routing.routing_candidate_details.map((candidate) => (
                          <div
                            key={candidate.route}
                            className="grid gap-2 py-3 text-sm text-zinc-700 sm:grid-cols-3"
                          >
                            <span className="font-medium text-zinc-900">
                              {formatLabel(candidate.route)}
                            </span>
                            <span>
                              Probability: {formatPercent(candidate.probability)}
                            </span>
                            <span>
                              Nonconformity:{" "}
                              {formatNumber(candidate.nonconformity)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null}

                  {inferenceRan && result.inference?.probabilities ? (
                    <div className="mt-10">
                      <p className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                        Model output probabilities
                      </p>

                      <div className="mt-4 divide-y divide-zinc-200 border-b border-t border-zinc-200">
                        {Object.entries(result.inference.probabilities).map(
                          ([label, probability]) => (
                            <div
                              key={label}
                              className="flex items-center justify-between gap-4 py-3 text-sm text-zinc-700"
                            >
                              <span>{label}</span>
                              <span className="font-medium text-zinc-900">
                                {formatPercent(probability)}
                              </span>
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  ) : null}

                  {warnings.length > 0 ? (
                    <div className="mt-10 border-t border-zinc-200 pt-6">
                      <p className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                        Things to review carefully
                      </p>

                      <div className="mt-4 flex flex-wrap gap-2">
                        {warnings.map((warning, index) => (
                          <span
                            key={`${warning}-${index}`}
                            className="inline-flex items-center gap-2 rounded-md bg-red-50 px-3 py-1.5 text-xs text-red-700 ring-1 ring-red-100"
                          >
                            <AlertTriangle size={12} />
                            {warning}
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : null}

                  <div className="mt-10 border-t border-zinc-200 pt-6">
                    <p className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                      Technical review details
                    </p>

                    <div className="mt-4 grid gap-6 text-sm leading-7 text-zinc-700 md:grid-cols-2">
                      <div>
                        <p>
                          Accepted for review:{" "}
                          {result.input_gate?.accepted_for_analysis === true
                            ? "Yes"
                            : result.input_gate?.accepted_for_analysis === false
                              ? "No"
                              : "—"}
                        </p>
                        <p>
                          Input confidence:{" "}
                          {formatPercent(result.input_gate?.confidence)}
                        </p>
                      </div>

                      <div>
                        <p>Policy action: {result.policy?.action || "—"}</p>
                        <p>Risk category: {result.policy?.risk_category || "—"}</p>
                      </div>

                      <div>
                        <p>Selected route: {formatLabel(selectedRoute)}</p>
                        <p>Raw route: {formatLabel(routeDetector?.raw_route_label)}</p>
                      </div>

                      <div>
                        <p>
                          Manual override:{" "}
                          {manualOverrideUsed ? "Yes" : "No"}
                        </p>
                        <p>
                          Requested override:{" "}
                          {formatLabel(
                            routeDetector?.override_metadata?.requested_route
                          )}
                        </p>
                      </div>

                      <div>
                        <p>Detected region: {result.detection?.region || "—"}</p>
                        <p>Detected modality: {result.detection?.modality || "—"}</p>
                      </div>

                      <div>
                        <p>Routing method: {result.routing?.method || "—"}</p>
                        <p>
                          Conformal set size:{" "}
                          {result.routing?.set_size ?? "—"}
                        </p>
                      </div>

                      <div>
                        <p>Alpha: {formatNumber(result.routing?.alpha)}</p>
                        <p>Threshold: {formatNumber(result.routing?.threshold)}</p>
                      </div>

                      <div>
                        <p>
                          Nonconformity:{" "}
                          {formatNumber(result.routing?.nonconformity)}
                        </p>
                        <p>
                          Requires confirmation:{" "}
                          {result.routing?.requires_confirmation ? "Yes" : "No"}
                        </p>
                      </div>

                      <div>
                        <p>Selected model: {result.routing?.selected_model || "—"}</p>
                        <p>Review set size: {result.routing?.set_size ?? "—"}</p>
                      </div>

                      <div>
                        <p>Distribution status: {result.ood?.tier || "—"}</p>
                        <p>OOD method: {result.ood?.method || "—"}</p>
                      </div>

                      <div>
                        <p>Screening score: {formatNumber(result.ood?.score)}</p>
                        <p>OOD reason: {result.ood?.reason || "—"}</p>
                      </div>

                      <div>
                        <p>Quality note: {result.quality?.reason || "—"}</p>
                        <p>
                          Re-upload suggested:{" "}
                          {result.quality?.requires_reupload ? "Yes" : "No"}
                        </p>
                      </div>

                      <div>
                        <p>
                          Reliability score:{" "}
                          {formatNumber(result.inference?.reliability_score)}
                        </p>
                        <p>
                          Disagreement score:{" "}
                          {formatNumber(result.inference?.disagreement_score)}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="mt-10 border-t border-zinc-200 pt-6">
                    <p className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                      Disclaimer
                    </p>

                    <p className="mt-4 text-sm leading-7 text-zinc-600">
                      {result.disclaimer ||
                        appConfig?.disclaimer ||
                        "This platform is intended solely for research and educational use. Outputs are non-diagnostic and must not be used for clinical decision-making."}
                    </p>
                  </div>
                </div>

                <div>
                  <p className="mb-4 text-xs uppercase tracking-[0.18em] text-zinc-400">
                    Model focus map
                  </p>

                  {heatmapUrl ? (
                    <div className="overflow-hidden rounded-[14px] border border-zinc-200 bg-white">
                      <img
                        src={heatmapUrl}
                        alt="Heatmap"
                        className="h-auto w-full object-cover"
                      />
                    </div>
                  ) : (
                    <div className="flex min-h-[240px] items-center justify-center rounded-[24px] border border-zinc-200 bg-white text-zinc-400">
                      Focus map unavailable
                    </div>
                  )}

                  <div className="mt-4 text-sm text-zinc-600">
                    <div className="flex items-center justify-between border-b border-zinc-200 py-3">
                      <span>Explanation method</span>
                      <span className="font-medium text-zinc-900">
                        {result.explainability?.method || "—"}
                      </span>
                    </div>

                    <div className="flex items-center justify-between border-b border-zinc-200 py-3">
                      <span>Explanation target</span>
                      <span className="font-medium text-zinc-900">
                        {result.explainability?.target_label ||
                          result.inference?.top_label ||
                          "—"}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </section>
        </section>
      </div>

      <footer id="contact" className="mt-16 border-t border-zinc-200 bg-[#F7F7F4]">
        <div className="mx-auto grid max-w-7xl gap-10 px-6 py-10 md:grid-cols-3">
          <div>
            <div className="mb-3 inline-flex items-center gap-2 text-zinc-900">
              <ShieldPlus size={18} className="text-red-500" />
              <span className="font-semibold">MedAIx</span>
            </div>
            <p className="text-sm leading-7 text-zinc-600">
              A research-use assistant for careful, non-diagnostic medical image
              review.
            </p>
          </div>

          <div>
            <h4 className="mb-3 text-sm font-semibold text-zinc-900">
              Quick links
            </h4>
            <div className="space-y-2 text-sm text-zinc-600">
              <Link
                href="/"
                className="inline-flex items-center gap-2 hover:text-zinc-900"
              >
                <House size={16} />
                Home
              </Link>
              <br />
              <Link
                href="/workspace"
                className="inline-flex items-center gap-2 hover:text-zinc-900"
              >
                <LayoutDashboard size={16} />
                Workspace
              </Link>
              <br />
              <Link
                href="/about"
                className="inline-flex items-center gap-2 hover:text-zinc-900"
              >
                <CircleHelp size={16} />
                About
              </Link>
            </div>
          </div>

          <div>
            <h4 className="mb-3 text-sm font-semibold text-zinc-900">Contact</h4>
            <div className="space-y-2 text-sm text-zinc-600">
              <div className="inline-flex items-center gap-2">
                <Mail size={16} />
                research@medaix.ai
              </div>
              <br />
              <div className="inline-flex items-center gap-2">
                <Phone size={16} />
                +00 000 000 0000
              </div>
              <br />
              <div className="inline-flex items-center gap-2">
                <MapPin size={16} />
                Ankara, Türkiye
              </div>
            </div>
          </div>
        </div>

        <div className="border-t border-zinc-200 bg-white/70">
          <div className="mx-auto max-w-7xl px-6 py-4">
            <p className="text-xs leading-6 tracking-[0.01em] text-zinc-500">
              <span className="font-semibold text-zinc-700">
                Important notice.
              </span>{" "}
              This platform is intended only for research and educational use.
              Outputs are non-diagnostic and must not be used for clinical
              decision-making.
            </p>
          </div>
        </div>
      </footer>
    </main>
  );
}