"use client";

import { useEffect, useState } from "react";
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
  RefreshCcw,
  ShieldCheck,
} from "lucide-react";

type RouteInfo = {
  route: string;
  region: string;
  modality: string;
  model: string;
  description?: string;
  status?: string;
};

type AppConfig = {
  active_routes?: RouteInfo[];
  supported_uploads?: string[];
  max_batch_size?: number;
};

type CacheInfo = {
  cached_models?: string[];
  count?: number;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

const fadeUp = {
  hidden: { opacity: 0, y: 14 },
  show: { opacity: 1, y: 0 },
};

const fallbackRoutes: RouteInfo[] = [
  {
    route: "abdomen_ct",
    region: "abdomen",
    modality: "ct",
    model: "abdomen_ct_resnet18",
    description: "Kidney CT classification.",
    status: "ACTIVE",
  },
  {
    route: "brain_mri",
    region: "brain",
    modality: "mri",
    model: "brain_mri_resnet18",
    description: "Brain MRI tumor-related classification.",
    status: "ACTIVE",
  },
  {
    route: "bone_xray",
    region: "bone",
    modality: "xray",
    model: "bone_xray_standard",
    description: "Normal / abnormal bone X-ray review.",
    status: "ACTIVE",
  },
  {
    route: "chest_xray",
    region: "chest",
    modality: "xray",
    model: "chest_xray_mvp",
    description: "Chest X-ray multilabel findings.",
    status: "ACTIVE",
  },
  {
    route: "retina_fundus",
    region: "retina",
    modality: "fundus",
    model: "retina_fundus_resnet18",
    description: "Diabetic retinopathy severity review.",
    status: "ACTIVE",
  },
];

const checks = [
  ["Route", "specialist pathway"],
  ["Quality", "image usability"],
  ["OOD", "distribution safety"],
  ["Uncertainty", "prediction stability"],
  ["Explain", "Grad-CAM++ evidence"],
  ["Policy", "final governed action"],
];

function formatLabel(value?: string | null) {
  if (!value) return "—";
  return value.replaceAll("_", " ");
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

export default function WorkspacePage() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [cache, setCache] = useState<CacheInfo | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(false);

  const activeRoutes = config?.active_routes?.length
    ? config.active_routes
    : fallbackRoutes;

  const loadWorkspaceData = async () => {
    try {
      setLoading(true);

      const healthResponse = await fetch(`${API_BASE_URL}/health`);
      setBackendOnline(healthResponse.ok);

      const configResponse = await fetch(`${API_BASE_URL}/config`);
      if (configResponse.ok) setConfig(await configResponse.json());

      const cacheResponse = await fetch(`${API_BASE_URL}/model-cache`);
      if (cacheResponse.ok) setCache(await cacheResponse.json());
    } catch {
      setBackendOnline(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadWorkspaceData();
  }, []);

  const statusText =
    backendOnline === null ? "Checking" : backendOnline ? "Online" : "Offline";

  return (
    <main
      className="relative min-h-screen overflow-hidden bg-[#fffaf3] text-slate-950"
      style={{ fontFamily: '"Aptos","Aptos Body","Segoe UI",Arial,sans-serif' }}
    >
      <div className="pointer-events-none absolute -right-40 -top-40 h-[520px] w-[520px] rounded-full bg-red-100/70 blur-3xl" />
      <div className="pointer-events-none absolute -left-44 top-[520px] h-[520px] w-[520px] rounded-full bg-sky-100/70 blur-3xl" />

      <div className="relative mx-auto max-w-7xl px-6 py-6">
        <SiteHeader active="workspace" />

        <motion.section
          className="grid min-h-[430px] gap-12 lg:grid-cols-[minmax(0,1fr)_460px] lg:items-center"
          initial="hidden"
          animate="show"
          transition={{ staggerChildren: 0.12 }}
        >
          <motion.div variants={fadeUp} transition={{ duration: 0.5 }}>
            <p className="text-xs uppercase tracking-[0.28em] text-red-500">
              Research workspace
            </p>

            <h2 className="mt-5 max-w-3xl text-4xl font-semibold leading-tight tracking-tight">
              A live map of the system behind the review.
            </h2>

            <p className="mt-6 max-w-2xl text-sm italic leading-7 text-slate-500">
              Use this page to inspect backend state, active routes, model cache,
              upload limits, and the governance checks that sit around model output.
            </p>

            <div className="mt-8 flex flex-wrap gap-3">
              <Link
                href="/"
                className="group inline-flex items-center gap-2 rounded-2xl border border-red-200 bg-red-50 px-5 py-4 text-sm font-semibold text-red-700 transition hover:bg-red-500 hover:text-white"
              >
                Open main review
                <ArrowRight size={17} className="transition group-hover:translate-x-1" />
              </Link>

              <button
                type="button"
                onClick={loadWorkspaceData}
                className="inline-flex items-center gap-2 rounded-2xl border border-slate-300 px-5 py-4 text-sm font-semibold text-slate-700 transition hover:border-red-300 hover:text-red-500"
              >
                <RefreshCcw size={16} className={loading ? "animate-spin" : ""} />
                {loading ? "Refreshing" : "Refresh"}
              </button>
            </div>
          </motion.div>

          <motion.aside
            variants={fadeUp}
            transition={{ duration: 0.55 }}
            className="relative min-h-[360px]"
          >
            <div className="absolute left-0 top-0">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
                Backend
              </p>
              <p className="mt-2 text-5xl font-semibold tracking-tight">
                {statusText}
              </p>
            </div>

            <div className="absolute left-0 top-32 grid grid-cols-3 gap-8">
              <div>
                <p className="text-xs uppercase tracking-[0.18em] text-red-400">
                  Routes
                </p>
                <p className="mt-2 text-3xl font-semibold">{activeRoutes.length}</p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.18em] text-sky-500">
                  Cached
                </p>
                <p className="mt-2 text-3xl font-semibold">{cache?.count ?? 0}</p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.18em] text-emerald-500">
                  Batch
                </p>
                <p className="mt-2 text-3xl font-semibold">
                  {config?.max_batch_size ?? "—"}
                </p>
              </div>
            </div>

            <div className="absolute bottom-0 left-0 right-0">
              <div className="mb-4 flex items-center gap-3">
                <span className="h-2 w-2 rounded-full bg-red-500" />
                <div className="h-px flex-1 bg-gradient-to-r from-red-300 via-slate-200 to-transparent" />
              </div>

              <div className="grid grid-cols-3 gap-3 text-xs text-slate-500">
                <p>health check</p>
                <p>route registry</p>
                <p>model cache</p>
              </div>
            </div>

            <motion.div
              animate={{ opacity: [0.4, 1, 0.4] }}
              transition={{ duration: 2.6, repeat: Infinity }}
              className="absolute right-12 top-24 h-2 w-2 rounded-full bg-red-500"
            />

            <motion.div
              animate={{ opacity: [1, 0.35, 1] }}
              transition={{ duration: 3.1, repeat: Infinity }}
              className="absolute right-28 top-44 h-2 w-2 rounded-full bg-sky-500"
            />

            <div className="absolute right-4 top-28 h-40 w-40 rounded-full border border-slate-200/80" />
            <div className="absolute right-16 top-40 h-16 w-16 rounded-full border border-red-200/80" />
          </motion.aside>
        </motion.section>

        <section className="mt-10">
          <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
            System path
          </p>

          <div className="mt-6 grid gap-4 md:grid-cols-6">
            {checks.map(([title, detail], index) => (
              <motion.div
                key={title}
                variants={fadeUp}
                initial="hidden"
                whileInView="show"
                viewport={{ once: true, amount: 0.2 }}
                transition={{ duration: 0.35, delay: index * 0.04 }}
              >
                <div className="mb-3 h-1 w-9 rounded-full bg-red-400" />
                <p className="text-sm font-semibold text-slate-900">{title}</p>
                <p className="mt-2 text-xs leading-5 text-slate-500">{detail}</p>
              </motion.div>
            ))}
          </div>
        </section>

        <section className="mt-16 grid gap-10 lg:grid-cols-[280px_minmax(0,1fr)]">
          <div>
            <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
              Active routes
            </p>
            <p className="mt-4 text-sm leading-7 text-slate-500">
              Each route is a specialist pathway connected to a model and expected image type.
            </p>
          </div>

          <div className="divide-y divide-slate-200/70">
            {activeRoutes.map((routeInfo, index) => (
              <motion.div
                key={routeInfo.route}
                variants={fadeUp}
                initial="hidden"
                whileInView="show"
                viewport={{ once: true, amount: 0.2 }}
                transition={{ duration: 0.35, delay: index * 0.03 }}
                className="grid gap-3 py-5 md:grid-cols-[190px_minmax(0,1fr)_120px]"
              >
                <div>
                  <p className="font-semibold text-slate-900">
                    {formatLabel(routeInfo.route)}
                  </p>
                  <p className="mt-1 text-xs uppercase tracking-[0.16em] text-red-400">
                    {routeInfo.region} / {routeInfo.modality}
                  </p>
                </div>

                <div>
                  <p className="text-sm leading-6 text-slate-500">
                    {routeInfo.description || "Specialist model route."}
                  </p>
                  <p className="mt-1 truncate text-xs text-slate-400">
                    {routeInfo.model}
                  </p>
                </div>

                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-emerald-600">
                  {routeInfo.status || "ACTIVE"}
                </p>
              </motion.div>
            ))}
          </div>
        </section>

        <section className="mt-16 grid gap-10 lg:grid-cols-[280px_minmax(0,1fr)]">
          <div>
            <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
              Runtime details
            </p>
            <p className="mt-4 text-sm leading-7 text-slate-500">
              Compact values that are useful while testing.
            </p>
          </div>

          <div className="grid gap-8 md:grid-cols-3">
            <div>
              <p className="text-sm font-semibold text-slate-900">Supported uploads</p>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                {config?.supported_uploads?.join(", ") || "PNG, JPG, TIFF, DICOM"}
              </p>
            </div>

            <div>
              <p className="text-sm font-semibold text-slate-900">Cached model keys</p>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                {cache?.cached_models?.length
                  ? `${cache.cached_models.length} key(s) available`
                  : "No model keys reported yet."}
              </p>
            </div>

            <div>
              <p className="text-sm font-semibold text-slate-900">Purpose</p>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                Inspect system state, not duplicate the upload screen.
              </p>
            </div>
          </div>
        </section>

        <SiteFooter />
      </div>
    </main>
  );
}
