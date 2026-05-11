"use client";

import { ChangeEvent, useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  CircleHelp,
  Download,
  FileText,
  Home,
  LayoutDashboard,
  Mail,
  Play,
  RefreshCcw,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

type AnalysisResponse = {
  analysis_id?: string;
  filename?: string;
  input_gate?: {
    accepted_for_analysis?: boolean;
    confidence?: number;
    message?: string;
    selected_route?: string;
    manual_override?: boolean;
    route_scores?: Record<string, number>;
    top_level_gate?: {
      route_detector?: {
        route_label?: string;
        raw_route_label?: string;
        confidence?: number;
        probabilities?: Record<string, number>;
        reason?: string;
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
    requires_confirmation?: boolean;
    manual_override?: boolean;
  };
  routing?: {
    selected_model?: string | null;
    requires_confirmation?: boolean;
    method?: string;
    set_size?: number;
  };
  inference?: {
    top_label?: string;
    top_probability?: number;
    probabilities?: Record<string, number>;
    reliability_score?: number;
    disagreement_score?: number;
    uncertainty_method?: string;
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
    warning?: string;
    target_label?: string;
  };
  quality?: {
    status?: string | null;
    warnings?: string[];
    requires_reupload?: boolean | null;
    blocking?: boolean;
    reason?: string;
  };
  ood?: {
    tier?: string;
    score?: number | null;
    reason?: string;
    method?: string;
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

type BatchResultItem = {
  index: number;
  filename: string;
  status: string;
  result?: AnalysisResponse;
  error?: string;
};

type BatchResponse = {
  batch_size: number;
  max_batch_size: number;
  completed: number;
  failed: number;
  results: BatchResultItem[];
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

const ACTIVE_ROUTES = [
  "abdomen_ct",
  "brain_mri",
  "bone_xray",
  "breast_mammography",
  "chest_xray",
  "retina_fundus",
  "skin_dermoscopy",
];

function formatLabel(value?: string | null) {
  if (!value) return "—";
  return value.replaceAll("_", " ");
}

function formatPercent(value?: number | null) {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value?: number | null) {
  if (value == null) return "—";
  return value.toFixed(3);
}

function getAssistantDecision(action?: string) {
  switch (action) {
    case "ANSWER":
      return "Review output is available.";
    case "ESCALATE":
      return "Human review is recommended.";
    case "REQUEST_EVIDENCE":
      return "Better image evidence is needed.";
    case "REFUSE":
      return "Output cannot be provided safely.";
    case "STOP":
      return "Review stopped before inference.";
    default:
      return "Ready for image review.";
  }
}

function getDecisionColor(action?: string) {
  switch (action) {
    case "ANSWER":
      return "text-emerald-700";
    case "ESCALATE":
    case "REQUEST_EVIDENCE":
      return "text-amber-700";
    case "STOP":
    case "REFUSE":
      return "text-red-700";
    default:
      return "text-slate-950";
  }
}

function getAssistantSummary(result: AnalysisResponse | null, selectedRoute: string) {
  if (!result) return "Choose an image and I’ll walk through the review step by step.";

  const action = result.policy?.action;
  const output = result.inference?.top_label;
  const confidence = formatPercent(result.inference?.top_probability);
  const route = formatLabel(selectedRoute);

  if (action === "ANSWER") {
    return `I selected the ${route} route and found ${output || "an output"} with ${confidence} confidence. The safety checks allowed the result to be shown for research use.`;
  }

  if (action === "ESCALATE") {
    return "The model produced an output, but the confidence or uncertainty suggests that a human reviewer should check it.";
  }

  if (action === "STOP") {
    return "I did not run inference because one of the safety checks was not satisfied.";
  }

  return result.policy?.reason || result.message || "The review is complete.";
}

function InfoLine({
  label,
  value,
  text,
}: {
  label: string;
  value: string;
  text: string;
}) {
  return (
    <div className="border-l border-slate-300 pl-4">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className="mt-2 text-lg font-semibold">{value}</p>
      <p className="mt-1 text-sm text-slate-500">{text}</p>
    </div>
  );
}


function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Could not read image file."));
    reader.readAsDataURL(file);
  });
}

async function fetchImageAsDataUrl(url: string): Promise<string | null> {
  try {
    const response = await fetch(url);
    const blob = await response.blob();

    return await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(new Error("Could not read heatmap image."));
      reader.readAsDataURL(blob);
    });
  } catch {
    return null;
  }
}


async function imageUrlToDataUrl(url: string): Promise<string | null> {
  try {
    const response = await fetch(url);
    if (!response.ok) return null;

    const blob = await response.blob();

    return await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(String(reader.result));
      reader.onerror = () => resolve(null);
      reader.readAsDataURL(blob);
    });
  } catch {
    return null;
  }
}

function guessImageFormat(dataUrl: string) {
  if (dataUrl.startsWith("data:image/png")) return "PNG";
  if (dataUrl.startsWith("data:image/jpeg") || dataUrl.startsWith("data:image/jpg")) return "JPEG";
  return "PNG";
}

export default function ReviewPage() {
  const [file, setFile] = useState<File | null>(null);
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [batchResult, setBatchResult] = useState<BatchResponse | null>(null);
  const [confirmedRoute, setConfirmedRoute] = useState("");
  const [loading, setLoading] = useState(false);
  const [batchLoading, setBatchLoading] = useState(false);
  const [overrideLoading, setOverrideLoading] = useState(false);
  const [error, setError] = useState("");

  const isDicomFile = file?.name.toLowerCase().endsWith(".dcm") ?? false;

  const previewUrl = useMemo(() => {
    if (!file) return "";
    return URL.createObjectURL(file);
  }, [file]);

  const routeDetector = result?.input_gate?.top_level_gate?.route_detector;
  const selectedRoute =
    result?.input_gate?.selected_route || routeDetector?.route_label || "—";
  const inferenceRan = Boolean(result?.inference?.top_label);

  const heatmapUrl = useMemo(() => {
    const rawPath = result?.explainability?.heatmap_path;
    if (!rawPath) return "";
    if (rawPath.startsWith("http://") || rawPath.startsWith("https://")) {
      return `${rawPath}?v=${result?.analysis_id || Date.now()}`;
    }
    return `${API_BASE_URL}${rawPath}?v=${result?.analysis_id || Date.now()}`;
  }, [result]);

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

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0] ?? null;
    setFile(selected);
    setResult(null);
    setConfirmedRoute("");
    setError("");
  };

  const handleBatchFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(event.target.files ?? []);
    setBatchFiles(selected);
    setBatchResult(null);
    setError("");
  };

  const handleAnalyzeBatch = async () => {
    if (!batchFiles.length) {
      setError("Please choose at least one image for batch review.");
      return;
    }

    try {
      setBatchLoading(true);
      setError("");
      setBatchResult(null);

      const formData = new FormData();
      batchFiles.forEach((selectedFile) => {
        formData.append("files", selectedFile);
      });

      const response = await fetch(`${API_BASE_URL}/analyze/batch`, {
        method: "POST",
        body: formData,
      });

      const text = await response.text();
      let data: BatchResponse & { detail?: string };

      try {
        data = JSON.parse(text);
      } catch {
        throw new Error(text || "Batch request failed.");
      }

      if (!response.ok) throw new Error(data?.detail || "Batch request failed.");

      setBatchResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setBatchLoading(false);
    }
  };

  const runAnalyze = async (routeOverride?: string) => {
    if (!file) {
      setError("Please choose an image first.");
      return;
    }

    try {
      routeOverride ? setOverrideLoading(true) : setLoading(true);
      setError("");

      const formData = new FormData();
      formData.append("file", file);
      if (routeOverride) formData.append("route_override", routeOverride);

      const endpoint = routeOverride ? "/analyze/override" : "/analyze";

      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: "POST",
        body: formData,
      });

      const text = await response.text();
      let data: AnalysisResponse & { detail?: string };

      try {
        data = JSON.parse(text);
      } catch {
        throw new Error(text || "Request failed.");
      }

      if (!response.ok) throw new Error(data?.detail || "Request failed.");

      setResult(data);
      setConfirmedRoute("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
      setOverrideLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setBatchFiles([]);
    setBatchResult(null);
    setResult(null);
    setConfirmedRoute("");
    setError("");
  };

  const handleDownloadJson = () => {
    if (!result) {
      setError("No review result available.");
      return;
    }

    const report = {
      generated_at: new Date().toISOString(),
      project: "Governed Medical Image Analysis",
      notice:
        "Research-use only. Outputs are non-diagnostic and must not be used for clinical decision-making.",
      result,
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: "application/json",
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = `governed_medical_image_analysis_report_${
      result.analysis_id || "review"
    }.json`;

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  };

  const handleDownloadPdf = async () => {
    if (!result) {
      setError("No review result available.");
      return;
    }

    const doc = new jsPDF();
    const generatedAt = new Date().toLocaleString();
    const fileName = result.filename || file?.name || "—";

    const addFooter = () => {
      const pageCount = doc.getNumberOfPages();

      for (let i = 1; i <= pageCount; i++) {
        doc.setPage(i);
        doc.setFont("helvetica", "normal");
        doc.setFontSize(8);
        doc.setTextColor(120);
        doc.text(
          "Research-use only. Non-diagnostic output. Not for clinical decision-making.",
          14,
          286
        );
        doc.text(`Page ${i} of ${pageCount}`, 180, 286);
      }
    };

    doc.setTextColor(15, 23, 42);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(18);
    doc.text("Governed Medical Image Analysis Report", 14, 18);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(80);
    doc.text(`Generated: ${generatedAt}`, 14, 26);
    doc.text(`File: ${fileName}`, 14, 32);

    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.setTextColor(15, 23, 42);
    doc.text("Summary", 14, 45);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(70);
    doc.text(
      doc.splitTextToSize(getAssistantSummary(result, selectedRoute), 180),
      14,
      52
    );

    autoTable(doc, {
      startY: 68,
      head: [["Field", "Value"]],
      body: [
        ["Analysis ID", result.analysis_id || "—"],
        ["Policy action", result.policy?.action || "—"],
        ["Policy reason", result.policy?.reason || "—"],
        ["Risk category", result.policy?.risk_category || "—"],
        ["Selected route", formatLabel(selectedRoute)],
        ["Route confidence", formatPercent(result.input_gate?.confidence)],
        ["Raw route", formatLabel(routeDetector?.raw_route_label)],
        ["Accepted for analysis", result.input_gate?.accepted_for_analysis ? "Yes" : "No"],
        ["Manual override used", manualOverrideUsed ? "Yes" : "No"],
        ["Region / Modality", `${result.detection?.region || "—"} / ${result.detection?.modality || "—"}`],
        ["Selected model", result.routing?.selected_model || "—"],
        ["Model output", result.inference?.top_label || "No inference"],
        ["Output confidence", formatPercent(result.inference?.top_probability)],
        ["Quality status", result.quality?.status || "—"],
        ["Quality reason", result.quality?.reason || "—"],
        ["Re-upload suggested", result.quality?.requires_reupload ? "Yes" : "No"],
        ["OOD tier", result.ood?.tier || "—"],
        ["OOD score", formatNumber(result.ood?.score)],
        ["OOD method", result.ood?.method || "—"],
        ["Uncertainty method", formatLabel(result.inference?.uncertainty_method)],
        ["MC dropout passes", String(result.inference?.mc_passes ?? "—")],
        ["Reliability score", formatNumber(result.inference?.reliability_score)],
        ["Disagreement score", formatNumber(result.inference?.disagreement_score)],
        ["Calibration", result.inference?.calibration?.enabled ? "Enabled" : "Not applied"],
        ["Temperature", String(result.inference?.calibration?.temperature ?? "—")],
        ["Explainability method", result.explainability?.method || "—"],
        ["Explanation target", result.explainability?.target_label || result.inference?.top_label || "—"],
        ["Heatmap path", result.explainability?.heatmap_path || "Unavailable"],
      ],
      styles: { fontSize: 8.5, cellPadding: 3, valign: "top" },
      headStyles: { fillColor: [15, 23, 42], textColor: [255, 255, 255] },
      columnStyles: {
        0: { fontStyle: "bold", cellWidth: 52 },
        1: { cellWidth: 128 },
      },
    });

    const routeRows = routeDetector?.probabilities
      ? Object.entries(routeDetector.probabilities).map(([label, probability]) => [
          formatLabel(label),
          formatPercent(probability),
        ])
      : [];

    if (routeRows.length) {
      autoTable(doc, {
        startY: (doc as any).lastAutoTable.finalY + 10,
        head: [["Route probability", "Value"]],
        body: routeRows,
        styles: { fontSize: 8.5, cellPadding: 3 },
        headStyles: { fillColor: [80, 80, 90] },
      });
    }

    const probabilityRows = result.inference?.probabilities
      ? Object.entries(result.inference.probabilities).map(([label, probability]) => [
          label,
          formatPercent(probability),
        ])
      : [];

    if (probabilityRows.length) {
      autoTable(doc, {
        startY: (doc as any).lastAutoTable.finalY + 10,
        head: [["Model probability", "Value"]],
        body: probabilityRows,
        styles: { fontSize: 8.5, cellPadding: 3 },
        headStyles: { fillColor: [80, 80, 90] },
      });
    }

    const warnings = [
      ...(result.policy?.warnings ?? []),
      ...(result.quality?.warnings ?? []),
      ...(result.warnings ?? []),
      ...(result.explainability?.warning ? [result.explainability.warning] : []),
    ];

    autoTable(doc, {
      startY: (doc as any).lastAutoTable.finalY + 10,
      head: [["Warnings / Notes"]],
      body: warnings.length ? warnings.map((item) => [item]) : [["No warnings reported."]],
      styles: { fontSize: 8.5, cellPadding: 3 },
      headStyles: { fillColor: [185, 28, 28] },
    });

    doc.addPage();

    doc.setTextColor(15, 23, 42);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(14);
    doc.text("Visual Evidence", 14, 18);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(80);
    doc.text(
      "Original uploaded image and generated model focus map, when available.",
      14,
      26
    );

    const inputImage = previewUrl && !isDicomFile ? await imageUrlToDataUrl(previewUrl) : null;
    const heatmapImage = heatmapUrl ? await imageUrlToDataUrl(heatmapUrl) : null;

    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.setTextColor(15, 23, 42);
    doc.text("Original image", 14, 40);

    if (inputImage) {
      doc.addImage(inputImage, guessImageFormat(inputImage), 14, 46, 82, 82);
    } else {
      doc.setFont("helvetica", "normal");
      doc.setFontSize(9);
      doc.setTextColor(120);
      doc.text("Original image preview unavailable.", 14, 52);
    }

    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.setTextColor(15, 23, 42);
    doc.text("Model focus map", 112, 40);

    if (heatmapImage) {
      doc.addImage(heatmapImage, guessImageFormat(heatmapImage), 112, 46, 82, 82);
    } else {
      doc.setFont("helvetica", "normal");
      doc.setFontSize(9);
      doc.setTextColor(120);
      doc.text("Focus map unavailable.", 112, 52);
    }

    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.setTextColor(15, 23, 42);
    doc.text("Technical interpretation", 14, 148);

    autoTable(doc, {
      startY: 156,
      head: [["Check", "Meaning"]],
      body: [
        [
          "Policy decision",
          "The final governance decision controlling whether the system can answer, escalate, request evidence, refuse, or stop.",
        ],
        [
          "Route detection",
          "The predicted image type used to select the specialist model before inference.",
        ],
        [
          "Quality check",
          "A usability check for blur, contrast, readability, artifacts, and whether re-upload is needed.",
        ],
        [
          "OOD screening",
          "Checks whether the input appears too far from expected medical-image patterns.",
        ],
        [
          "Uncertainty",
          "Repeated stochastic passes estimate how stable or unstable the prediction is.",
        ],
        [
          "Calibration",
          "Temperature scaling helps reduce overconfident probability values.",
        ],
      ],
      styles: { fontSize: 8.5, cellPadding: 3, valign: "top" },
      headStyles: { fillColor: [15, 23, 42] },
      columnStyles: {
        0: { fontStyle: "bold", cellWidth: 45 },
        1: { cellWidth: 135 },
      },
    });

    addFooter();

    doc.save(
      `governed_medical_image_analysis_report_${result.analysis_id || "review"}.pdf`
    );
  };

  return (
    <main
      className="relative min-h-screen overflow-hidden bg-[#fffaf3] text-slate-950"
      style={{ fontFamily: '"Aptos","Aptos Body","Segoe UI",Arial,sans-serif' }}
    >
      <div className="pointer-events-none absolute right-0 top-0 h-80 w-80 rounded-full bg-pink-300/35 blur-3xl" />
      <div className="pointer-events-none absolute bottom-10 left-8 h-80 w-80 rounded-full bg-sky-300/35 blur-3xl" />

      <div className="relative mx-auto max-w-7xl px-6 py-6">
        <header className="flex items-center justify-between border-b border-slate-300 pb-5">
          <Link href="/" className="flex items-center gap-3">
            <div className="relative h-10 w-14">
              <span className="absolute left-1 top-1/2 h-2.5 w-2.5 -translate-y-1/2 rounded-full bg-slate-950" />
              <span className="absolute right-1 top-1/2 h-2.5 w-2.5 -translate-y-1/2 rounded-full bg-slate-950" />
              <span className="absolute left-1/2 top-1/2 h-[2px] w-7 -translate-x-1/2 -translate-y-1/2 rounded-full bg-slate-400" />
            </div>
            <div>
              <p className="text-[11px] uppercase tracking-[0.25em] text-slate-500">
                Gentle research-use review
              </p>
              <h1 className="text-xl font-semibold">
                Governed Medical Image Analysis
              </h1>
            </div>
          </Link>

          <nav className="flex flex-wrap items-center gap-5 text-sm text-slate-600">
            <Link href="/" className="inline-flex items-center gap-2 transition hover:text-red-500">
              <Home size={16} />
              Home
            </Link>
            <Link href="/demo" className="inline-flex items-center gap-2 transition hover:text-red-500">
              <Play size={16} />
              Demo
            </Link>
            <Link href="/workspace" className="inline-flex items-center gap-2 transition hover:text-red-500">
              <LayoutDashboard size={16} />
              Workspace
            </Link>
            <Link href="/about" className="inline-flex items-center gap-2 transition hover:text-red-500">
              <CircleHelp size={16} />
              About
            </Link>
            <Link href="/contact" className="inline-flex items-center gap-2 transition hover:text-red-500">
              <Mail size={16} />
              Contact
            </Link>
          </nav>
        </header>

        <section className="grid gap-10 py-8 lg:grid-cols-[320px_minmax(0,1fr)]">
          <aside>
            <label className="group block cursor-pointer border-y border-slate-300 py-6 transition hover:border-red-400">
              <div className="flex items-start justify-between gap-5">
                <div className="min-w-0">
                  <div className="mb-3 inline-flex items-center gap-2 text-xs font-medium uppercase tracking-[0.18em] text-red-500">
                    <ShieldCheck size={13} />
                    Safe upload
                  </div>

                  <p className="text-lg font-semibold tracking-tight text-slate-950">
                    {file ? "Selected image" : "Choose an image"}
                  </p>

                  <p className="mt-2 max-w-[270px] text-sm leading-6 text-slate-500">
                    {file
                      ? "Ready for review. I’ll check whether it is safe before showing a result."
                      : "Select a scan and I’ll check whether it is safe to review before showing a result."}
                  </p>

                  {file ? (
                    <p className="mt-3 max-w-[270px] truncate text-xs text-slate-400">
                      File: {file.name}
                    </p>
                  ) : null}

                  <p className="mt-4 text-xs font-medium uppercase tracking-[0.16em] text-slate-400">
                    PNG · JPG · TIFF · DICOM
                  </p>
                </div>

                <span className="mt-1 text-sm font-semibold text-slate-500 transition group-hover:text-red-500">
                  Browse
                </span>
              </div>

              <input
                type="file"
                accept=".png,.jpg,.jpeg,.tif,.tiff,.dcm"
                className="hidden"
                onChange={handleFileChange}
              />
            </label>

            <div className="mt-5 grid gap-3">
              <button
                type="button"
                onClick={() => runAnalyze()}
                disabled={loading}
                className="group flex w-full items-center justify-between rounded-2xl border border-red-200 bg-red-50 px-5 py-4 text-sm font-semibold text-red-700 transition hover:bg-red-500 hover:text-white disabled:opacity-60"
              >
                <span>{loading ? "Reviewing carefully..." : "Start guided review"}</span>
                <ArrowRight size={17} className="transition group-hover:translate-x-1" />
              </button>

              <button
                type="button"
                onClick={handleReset}
                className="group flex w-full items-center justify-between rounded-2xl border border-red-200 bg-red-50 px-5 py-4 text-sm font-semibold text-red-700 transition hover:bg-red-500 hover:text-white"
              >
                <span>Clear and start over</span>
                <RefreshCcw size={17} className="transition group-hover:rotate-[-20deg]" />
              </button>
            </div>

            {error ? <p className="mt-4 text-sm text-red-700">{error}</p> : null}

            <div className="mt-8 border-t border-slate-300 pt-5">
              <div className="mb-4 flex items-center gap-2">
                <Sparkles size={15} className="text-red-500" />
                <p className="text-sm font-semibold text-slate-950">
                  Batch review
                </p>
              </div>

              <label className="block cursor-pointer border-y border-slate-200 py-4 transition hover:border-red-400">
                <p className="text-sm font-semibold">
                  {batchFiles.length
                    ? `${batchFiles.length} image(s) selected`
                    : "Choose multiple images"}
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  Run several images through the same governed pipeline.
                </p>

                <input
                  type="file"
                  multiple
                  accept=".png,.jpg,.jpeg,.tif,.tiff,.dcm"
                  className="hidden"
                  onChange={handleBatchFileChange}
                />
              </label>

              <button
                type="button"
                onClick={handleAnalyzeBatch}
                disabled={batchLoading}
                className="mt-4 group flex w-full items-center justify-between rounded-2xl border border-red-200 bg-red-50 px-5 py-4 text-sm font-semibold text-red-700 transition hover:bg-red-500 hover:text-white disabled:opacity-60"
              >
                <span>{batchLoading ? "Running batch..." : "Run batch review"}</span>
                <ArrowRight size={17} className="transition group-hover:translate-x-1" />
              </button>

              {batchResult ? (
                <div className="mt-5 space-y-3 text-sm">
                  <div className="grid grid-cols-3 gap-3 border-y border-slate-200 py-3 text-center">
                    <div>
                      <p className="font-semibold text-slate-950">{batchResult.batch_size}</p>
                      <p className="text-xs text-slate-500">Total</p>
                    </div>
                    <div>
                      <p className="font-semibold text-emerald-700">{batchResult.completed}</p>
                      <p className="text-xs text-slate-500">Done</p>
                    </div>
                    <div>
                      <p className="font-semibold text-red-700">{batchResult.failed}</p>
                      <p className="text-xs text-slate-500">Failed</p>
                    </div>
                  </div>

                  <div className="max-h-48 divide-y divide-slate-200 overflow-y-auto border-b border-slate-200">
                    {batchResult.results.map((item) => {
                      const itemRoute =
                        item.result?.input_gate?.selected_route ||
                        item.result?.input_gate?.top_level_gate?.route_detector?.route_label ||
                        "—";
                      const itemOutput = item.result?.inference?.top_label || "No inference";
                      const itemPolicy = item.result?.policy?.action || "—";

                      return (
                        <button
                          type="button"
                          key={`${item.index}-${item.filename}`}
                          onClick={() => {
                            if (item.result) {
                              const matchedFile = batchFiles.find(
                                (selectedFile) => selectedFile.name === item.filename
                              );
                              if (matchedFile) setFile(matchedFile);
                              setResult(item.result);
                            }
                          }}
                          className="w-full py-3 text-left transition hover:text-red-500"
                        >
                          <p className="truncate font-medium">{item.filename}</p>
                          <p className="mt-1 text-xs text-slate-500">
                            {formatLabel(itemRoute)} · {itemOutput} · {itemPolicy}
                          </p>
                        </button>
                      );
                    })}
                  </div>
                </div>
              ) : null}
            </div>


            {needsManualConfirmation ? (
              <div className="mt-7 border-t border-red-200 pt-5">
                <div className="mb-4 flex items-start gap-3">
                  <div className="relative mt-1 h-6 w-10 shrink-0">
                    <span className="absolute left-1 top-1/2 h-2 w-2 -translate-y-1/2 rounded-full bg-slate-950" />
                    <span className="absolute right-1 top-1/2 h-2 w-2 -translate-y-1/2 rounded-full bg-slate-950" />
                    <span className="absolute left-1/2 top-1/2 h-[2px] w-5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-slate-400" />
                  </div>

                  <div>
                    <p className="text-sm font-semibold text-slate-950">
                      Confirm the image route
                    </p>
                    <p className="mt-1 text-sm leading-6 text-slate-500">
                      Use only when a human evaluator knows the intended image type.
                    </p>
                  </div>
                </div>

                <div className="flex flex-col gap-3">
                  <select
                    value={confirmedRoute}
                    onChange={(event) => setConfirmedRoute(event.target.value)}
                    className="border-0 border-b border-slate-300 bg-transparent px-0 py-3 text-sm text-slate-800 outline-none transition focus:border-red-400"
                  >
                    <option value="">Select confirmed route</option>
                    {ACTIVE_ROUTES.map((route) => (
                      <option key={route} value={route}>
                        {formatLabel(route)}
                      </option>
                    ))}
                  </select>

                  <button
                    type="button"
                    onClick={() => runAnalyze(confirmedRoute)}
                    disabled={overrideLoading}
                    className="w-fit rounded-2xl border border-red-200 bg-red-50 px-5 py-3 text-sm font-semibold text-red-700 transition hover:bg-red-500 hover:text-white disabled:opacity-60"
                  >
                    {overrideLoading ? "Running..." : "Run confirmed route"}
                  </button>
                </div>
              </div>
            ) : null}
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

            {result ? (
              <>
                <div className="mt-8">
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-500">
                    Main output
                  </p>
                  <p className="mt-2 text-4xl font-semibold tracking-tight">
                    {inferenceRan ? result.inference?.top_label : "No inference"}
                  </p>
                  <p className="mt-2 text-sm text-slate-500">
                    {inferenceRan
                      ? `Confidence: ${formatPercent(result.inference?.top_probability)}`
                      : "The system stopped before producing a model output."}
                  </p>
                </div>

                <div className="mt-8 grid gap-5 md:grid-cols-3">
                  <InfoLine
                    label="Route"
                    value={formatLabel(selectedRoute)}
                    text={`Confidence: ${formatPercent(result.input_gate?.confidence)}`}
                  />
                  <InfoLine
                    label="Safety"
                    value={result.policy?.risk_category || "—"}
                    text={`OOD: ${result.ood?.tier || "—"}`}
                  />
                  <InfoLine
                    label="Model"
                    value={result.routing?.selected_model || "—"}
                    text={`${result.detection?.region || "—"} / ${result.detection?.modality || "—"}`}
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
                        {previewUrl && !isDicomFile ? (
                          <img
                            src={previewUrl}
                            alt="Input preview"
                            className="max-h-[340px] w-full object-contain"
                          />
                        ) : isDicomFile ? (
                          <p className="text-sm text-slate-500">
                            DICOM preview unavailable in browser.
                          </p>
                        ) : (
                          <p className="text-sm text-slate-400">No input selected.</p>
                        )}
                      </div>
                    </div>

                    <div>
                      <p className="mb-3 text-sm font-semibold">
                        Where the model focused
                      </p>
                      <div className="flex min-h-[280px] items-center justify-center border-y border-slate-300 py-5">
                        {heatmapUrl ? (
                          <img
                            src={heatmapUrl}
                            alt="Heatmap"
                            className="max-h-[340px] w-full object-contain"
                          />
                        ) : (
                          <p className="text-sm text-slate-400">
                            Focus map unavailable.
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mt-8 border-t border-slate-300 pt-5">
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-500">
                    Save review
                  </p>

                  <p className="mt-2 text-sm leading-6 text-slate-600">
                    Save a readable report for presentation, or export the raw backend audit.
                  </p>

                  <div className="mt-4 flex flex-wrap gap-x-6 gap-y-3 text-sm">
                    <button
                      type="button"
                      onClick={handleDownloadPdf}
                      className="relative inline-flex items-center gap-2 border border-slate-300 px-4 py-2.5 font-semibold text-slate-950 transition hover:border-red-400 hover:text-red-500 before:absolute before:left-0 before:top-0 before:h-2.5 before:w-2.5 before:border-l-2 before:border-t-2 before:border-red-500 after:absolute after:bottom-0 after:right-0 after:h-2.5 after:w-2.5 after:border-b-2 after:border-r-2 after:border-red-500"
                    >
                      <Download size={15} />
                      Save readable PDF
                    </button>

                    <button
                      type="button"
                      onClick={handleDownloadJson}
                      className="relative inline-flex items-center gap-2 border border-slate-300 px-4 py-2.5 font-semibold text-slate-700 transition hover:border-red-400 hover:text-red-500 before:absolute before:left-0 before:top-0 before:h-2.5 before:w-2.5 before:border-l-2 before:border-t-2 before:border-red-400 after:absolute after:bottom-0 after:right-0 after:h-2.5 after:w-2.5 after:border-b-2 after:border-r-2 after:border-red-400"
                    >
                      Export technical JSON
                      <ArrowRight size={15} />
                    </button>
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

                    <div className="mb-7 grid gap-5 md:grid-cols-3">
                      <AuditMetric
                        label="Route confidence"
                        value={result.input_gate?.confidence}
                        helper="How strongly the router selected this image type."
                      />
                      <AuditMetric
                        label="Output confidence"
                        value={result.inference?.top_probability}
                        helper="How confident the selected model was in its top output."
                      />
                      <AuditMetric
                        label="Reliability"
                        value={result.inference?.reliability_score}
                        helper="Higher means the repeated uncertainty passes were more stable."
                      />
                    </div>

                    <div className="space-y-6">
                      <AuditBlock title="Input and route decision" tag="routing">
                        <p>
                          The system first checked the uploaded file and selected the{" "}
                          <span className="font-semibold text-slate-950">
                            {formatLabel(selectedRoute)}
                          </span>{" "}
                          route.
                        </p>
                        <p>
                          This means the image was handled by the specialist pathway for{" "}
                          <span className="font-semibold text-slate-950">
                            {result.detection?.region || "—"} / {result.detection?.modality || "—"}
                          </span>.
                        </p>
                        <p className="text-xs text-slate-500">
                          Raw route: {formatLabel(routeDetector?.raw_route_label)} ·
                          Accepted for analysis:{" "}
                          {result.input_gate?.accepted_for_analysis ? "Yes" : "No"}
                        </p>
                      </AuditBlock>

                      <AuditBlock title="Model output" tag="inference">
                        <p>
                          The selected model was{" "}
                          <span className="font-semibold text-slate-950">
                            {result.routing?.selected_model || "—"}
                          </span>.
                        </p>
                        <p>
                          It produced{" "}
                          <span className="font-semibold text-slate-950">
                            {result.inference?.top_label || "No inference"}
                          </span>{" "}
                          with {formatPercent(result.inference?.top_probability)} confidence.
                        </p>
                        <p className="border-l-2 border-red-300 bg-red-50/60 px-3 py-2 text-xs font-medium text-red-700">
                          This confidence is model evidence only. It is not a clinical diagnosis.
                        </p>
                      </AuditBlock>

                      <AuditBlock title="Image quality check" tag="quality">
                        <p>
                          Status:{" "}
                          <span className="font-semibold text-slate-950">
                            {result.quality?.status || "—"}
                          </span>
                        </p>
                        <p>{result.quality?.reason || "No quality note returned."}</p>
                        <p className="text-xs text-slate-500">
                          Re-upload suggested: {result.quality?.requires_reupload ? "Yes" : "No"}
                        </p>
                      </AuditBlock>

                      <AuditBlock title="Distribution check" tag="OOD">
                        <p>
                          OOD tier:{" "}
                          <span className="font-semibold text-slate-950">
                            {result.ood?.tier || "—"}
                          </span>
                        </p>
                        <p>
                          This checks whether the image looks too different from the type of
                          medical images the system expects for this route.
                        </p>
                        <p className="text-xs text-slate-500">
                          Method: {result.ood?.method || "—"} · Score:{" "}
                          {formatNumber(result.ood?.score)}
                        </p>
                      </AuditBlock>

                      <AuditBlock title="Uncertainty and calibration" tag="confidence">
                        <p>
                          Uncertainty method:{" "}
                          <span className="font-semibold text-slate-950">
                            {formatLabel(result.inference?.uncertainty_method)}
                          </span>.
                        </p>
                        <p>
                          The system used {result.inference?.mc_passes ?? "—"} MC passes to
                          estimate prediction stability.
                        </p>
                        <p className="text-xs text-slate-500">
                          Disagreement: {formatNumber(result.inference?.disagreement_score)} ·
                          Calibration:{" "}
                          {result.inference?.calibration?.enabled ? "Enabled" : "Not applied"} ·
                          Temperature: {result.inference?.calibration?.temperature ?? "—"}
                        </p>
                      </AuditBlock>

                      <AuditBlock title="Focus map interpretation" tag="Grad-CAM++">
                        <p>
                          Method:{" "}
                          <span className="font-semibold text-slate-950">
                            {result.explainability?.method || "—"}
                          </span>.
                        </p>
                        <p>
                          The focus map shows which visual areas influenced the output most.
                          It should be used to check whether the model looked at a reasonable
                          anatomical area, not as proof of disease.
                        </p>
                        <p className="text-xs text-slate-500">
                          Target: {result.explainability?.target_label || result.inference?.top_label || "—"}
                        </p>
                      </AuditBlock>

                      <AuditBlock title="Final governance decision" tag="policy">
                        <p>
                          Final action:{" "}
                          <span className="font-semibold text-slate-950">
                            {result.policy?.action || "—"}
                          </span>.
                        </p>
                        <p>{result.policy?.reason || "No policy reason returned."}</p>
                        <div className="space-y-2 text-xs text-slate-600">
                          <p>
                            <span className="font-semibold text-emerald-700">ANSWER</span>
                            {" "}→ output may be shown.
                          </p>
                          <p>
                            <span className="font-semibold text-amber-700">ESCALATE</span>
                            {" "}→ output exists, but human review is recommended.
                          </p>
                          <p>
                            <span className="font-semibold text-amber-700">REQUEST_EVIDENCE</span>
                            {" "}→ a clearer or more suitable image is needed.
                          </p>
                          <p>
                            <span className="font-semibold text-red-700">REFUSE</span>
                            {" "}→ the system refuses to provide an output because safety checks were not satisfied.
                          </p>
                          <p>
                            <span className="font-semibold text-red-700">STOP</span>
                            {" "}→ the system stopped before inference or before showing an unsafe output.
                          </p>
                        </div>
                      </AuditBlock>
                    </div>
                  </div>
                </details>
              </>
            ) : null}
          </section>
        </section>

        <footer className="mt-14 border-t border-slate-300 py-8">
          <div className="grid gap-8 md:grid-cols-3">
            <div>
              <div className="mb-3 flex items-center gap-3">
                <div className="relative h-8 w-12">
                  <span className="absolute left-1 top-1/2 h-2 w-2 -translate-y-1/2 rounded-full bg-slate-950" />
                  <span className="absolute right-1 top-1/2 h-2 w-2 -translate-y-1/2 rounded-full bg-slate-950" />
                  <span className="absolute left-1/2 top-1/2 h-[2px] w-6 -translate-x-1/2 -translate-y-1/2 rounded-full bg-slate-400" />
                </div>
                <p className="font-semibold text-slate-950">
                  Governed Medical Image Analysis
                </p>
              </div>
              <p className="max-w-md text-sm leading-7 text-slate-500">
                A gentle research-use assistant for careful, non-diagnostic medical image review.
              </p>
            </div>

            <div>
              <p className="mb-3 text-sm font-semibold text-slate-950">Quick links</p>
              <div className="space-y-2 text-sm text-slate-600">
                <Link href="/" className="flex items-center gap-2 hover:text-red-500">
                  <Home size={15} />
                  Home
                </Link>
                <Link href="/demo" className="flex items-center gap-2 hover:text-red-500">
                  <Play size={15} />
                  Demo
                </Link>
                <Link href="/workspace" className="flex items-center gap-2 hover:text-red-500">
                  <LayoutDashboard size={15} />
                  Workspace
                </Link>
                <Link href="/about" className="flex items-center gap-2 hover:text-red-500">
                  <CircleHelp size={15} />
                  About
                </Link>
              </div>
            </div>

            <div>
              <div className="mb-3 flex items-center gap-2">
                <ShieldCheck size={16} className="text-red-500" />
                <p className="text-sm font-semibold text-slate-950">Notice</p>
              </div>
              <p className="text-xs leading-6 text-slate-500">
                <span className="font-semibold text-slate-700">Research-use only.</span>{" "}
                Outputs are non-diagnostic and must not be used for clinical decision-making.
              </p>
            </div>
          </div>
        </footer>
      </div>
    </main>
  );
}



function PipelineStep({
  label,
  done,
}: {
  label: string;
  done: boolean;
}) {
  return (
    <div className="flex items-center gap-2 text-sm">
      <span
        className={`h-2.5 w-2.5 rounded-full ${
          done ? "bg-red-500" : "bg-slate-300"
        }`}
      />
      <span className={done ? "text-slate-950" : "text-slate-400"}>
        {label}
      </span>
    </div>
  );
}

function DecisionLegend() {
  return (
    <div className="mt-5 border-y border-slate-300 py-4">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">
        Governance decisions
      </p>

      <div className="mt-3 grid gap-2 text-xs leading-5 md:grid-cols-2">
        <p><span className="font-semibold text-emerald-700">ANSWER</span> → output may be shown</p>
        <p><span className="font-semibold text-amber-700">ESCALATE</span> → human review recommended</p>
        <p><span className="font-semibold text-amber-700">REQUEST_EVIDENCE</span> → better image needed</p>
        <p><span className="font-semibold text-red-700">REFUSE</span> → output refused</p>
        <p><span className="font-semibold text-red-700">STOP</span> → pipeline stopped safely</p>
      </div>
    </div>
  );
}

function ReportPreview() {
  return (
    <div className="mt-6 border-y border-slate-300 py-5">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">
        Report includes
      </p>

      <div className="mt-3 grid gap-2 text-sm text-slate-600 md:grid-cols-2">
        <p>✓ Output summary</p>
        <p>✓ Route decision</p>
        <p>✓ Quality check</p>
        <p>✓ OOD screening</p>
        <p>✓ Uncertainty values</p>
        <p>✓ Original image</p>
        <p>✓ Grad-CAM++ focus map</p>
        <p>✓ Governance decision</p>
      </div>
    </div>
  );
}

function AuditMetric({
  label,
  value,
  helper,
}: {
  label: string;
  value?: number | null;
  helper: string;
}) {
  const percent = value == null ? 0 : Math.max(0, Math.min(100, value * 100));

  return (
    <div className="border-b border-slate-200 pb-4">
      <div className="mb-2 flex items-center justify-between gap-4">
        <div>
          <p className="text-sm font-semibold text-slate-950">{label}</p>
          <p className="mt-1 text-xs leading-5 text-slate-500">{helper}</p>
        </div>
        <span className="text-sm font-semibold text-slate-800">
          {value == null ? "—" : formatPercent(value)}
        </span>
      </div>

      <div className="h-2 overflow-hidden rounded-full bg-slate-200">
        <div
          className="h-2 rounded-full bg-red-500"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}

function AuditBlock({
  title,
  tag,
  children,
}: {
  title: string;
  tag: string;
  children: React.ReactNode;
}) {
  return (
    <section className="border-b border-slate-200 pb-5">
      <div className="mb-4 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <span className="h-2.5 w-2.5 rounded-full bg-red-500" />
          <h4 className="text-base font-semibold tracking-tight text-slate-950">
            {title}
          </h4>
        </div>

        <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-red-500">
          {tag}
        </span>
      </div>

      <div className="border-l-2 border-slate-200 pl-4">
        <div className="space-y-3 text-sm leading-7 text-slate-600">
          {children}
        </div>
      </div>
    </section>
  );
}

function AuditItem({
  title,
  value,
  text,
}: {
  title: string;
  value: string;
  text: string;
}) {
  return (
    <div className="grid gap-2 border-b border-slate-200 pb-4 md:grid-cols-[180px_minmax(0,1fr)]">
      <div>
        <p className="font-semibold text-slate-950">{title}</p>
      </div>
      <div>
        <p className="font-semibold text-slate-950">{value}</p>
        <p className="mt-1 text-slate-600">{text}</p>
      </div>
    </div>
  );
}
