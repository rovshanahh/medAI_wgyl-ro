"use client";

import Link from "next/link";
import SiteHeader from "../_components/SiteHeader";
import SiteFooter from "../_components/SiteFooter";
import { motion } from "framer-motion";
import {
  ArrowRight,
  CircleHelp,
  FileText,
  Home,
  LayoutDashboard,
  Mail,
  Play,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

const fadeUp = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0 },
};

const softPop = {
  hidden: { opacity: 0, scale: 0.96 },
  show: { opacity: 1, scale: 1 },
};

const ideas = [
  {
    title: "Question the input first.",
    text: "The system checks format, readability, quality, and supported route before inference.",
  },
  {
    title: "Route to the right model.",
    text: "Different image types are sent to their own specialist pathway.",
  },
  {
    title: "Do not trust confidence alone.",
    text: "Confidence is reviewed together with uncertainty, OOD, quality, and routing signals.",
  },
  {
    title: "Use safe final actions.",
    text: "The policy can answer, escalate, request evidence, refuse, or stop.",
  },
];

const flow = [
  ["Upload", "User provides a supported image or DICOM file."],
  ["Validate", "The file is checked for format, readability, and basic safety."],
  ["Route", "The system selects the correct medical image pathway."],
  ["Infer", "The selected specialist model produces the output."],
  ["Check risk", "Quality, OOD, confidence, and uncertainty are reviewed."],
  ["Explain", "Grad-CAM++ shows the image regions that influenced the output."],
  ["Decide", "The policy layer decides whether to answer, escalate, request evidence, refuse, or stop."],
  ["Report", "The result is saved with images, heatmap, signals, and audit details."],
];

export default function AboutPage() {
  return (
    <main
      className="relative min-h-screen overflow-hidden bg-[#fffaf3] text-slate-950"
      style={{ fontFamily: '"Aptos","Aptos Body","Segoe UI",Arial,sans-serif' }}
    >
      <div className="pointer-events-none absolute right-[-160px] top-[-160px] h-[460px] w-[460px] rounded-full bg-red-100 blur-3xl" />
      <div className="pointer-events-none absolute left-[-160px] top-[520px] h-[460px] w-[460px] rounded-full bg-sky-100 blur-3xl" />

      <div className="relative mx-auto max-w-7xl px-6 py-6">
        <SiteHeader active="about" />

        <motion.section
          className="grid min-h-[520px] gap-12 lg:grid-cols-[minmax(0,1fr)_420px] lg:items-center"
          initial="hidden"
          animate="show"
          transition={{ staggerChildren: 0.12 }}
        >
          <motion.div variants={fadeUp} transition={{ duration: 0.55, ease: "easeOut" }}>
            <p className="text-xs uppercase tracking-[0.28em] text-red-500">
              Safety layer for medical AI
            </p>

            <h2 className="mt-5 max-w-4xl text-4xl font-semibold leading-tight tracking-tight">
              The goal is not only to predict. The goal is to know when prediction should be trusted.
            </h2>

            <p className="mt-6 max-w-3xl text-base italic leading-8 text-slate-600">
              This research-use platform wraps medical image AI with routing,
              uncertainty, out-of-distribution screening, visual explanation, and a
              governed policy decision before showing output.
            </p>

            <div className="mt-8 flex flex-wrap gap-3">
              <Link
                href="/"
                className="group inline-flex items-center gap-2 rounded-2xl border border-red-200 bg-red-50 px-5 py-4 text-sm font-semibold text-red-700 transition hover:bg-red-500 hover:text-white"
              >
                Try main review
                <ArrowRight size={17} className="transition group-hover:translate-x-1" />
              </Link>

              <Link
                href="/demo"
                className="inline-flex items-center gap-2 rounded-2xl border border-slate-300 px-5 py-4 text-sm font-semibold text-slate-700 transition hover:border-red-300 hover:text-red-500"
              >
                Run demo
                <Play size={17} />
              </Link>
            </div>
          </motion.div>

          <motion.aside
            variants={softPop}
            transition={{ duration: 0.65, ease: "easeOut" }}
            className="relative min-h-[360px]"
          >
            <div className="absolute left-8 top-8 h-44 w-44 rounded-full bg-white shadow-[0_28px_70px_rgba(15,23,42,0.08)]" />
            <div className="absolute right-8 top-20 h-36 w-36 rounded-full bg-red-50 shadow-[0_18px_50px_rgba(239,68,68,0.10)]" />
            <div className="absolute bottom-8 left-24 h-32 w-32 rounded-full bg-sky-50 shadow-[0_18px_50px_rgba(14,165,233,0.10)]" />

            <div className="absolute left-20 top-24 flex h-28 w-48 items-center justify-center rounded-[42px] bg-white shadow-[0_24px_80px_rgba(15,23,42,0.08)]">
              <div className="relative h-12 w-28">
                <span className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 rounded-full bg-slate-950" />
                <span className="absolute right-2 top-1/2 h-4 w-4 -translate-y-1/2 rounded-full bg-slate-950" />
                <span className="absolute left-1/2 top-1/2 h-[3px] w-16 -translate-x-1/2 -translate-y-1/2 rounded-full bg-slate-400" />
              </div>
            </div>

            <div className="absolute right-2 top-4 rounded-full bg-red-50 px-4 py-2 text-xs font-semibold text-red-600 ring-1 ring-red-100">
              stop if unsafe
            </div>

            <div className="absolute bottom-2 left-4 rounded-full bg-sky-50 px-4 py-2 text-xs font-semibold text-sky-700 ring-1 ring-sky-100">
              explain before trust
            </div>

            <div className="absolute bottom-28 right-0 rounded-full bg-white px-4 py-2 text-xs font-semibold text-slate-600 shadow-sm">
              audit every decision
            </div>
          </motion.aside>
        </motion.section>

        <section className="mt-10 grid gap-12 lg:grid-cols-[300px_minmax(0,1fr)]">
          <div>
            <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
              Why it matters
            </p>
            <p className="mt-4 text-sm leading-7 text-slate-500">
              Medical AI can look confident even when the image is unclear,
              unsupported, or unfamiliar. This project focuses on that reliability gap.
            </p>
          </div>

          <div className="grid gap-x-12 gap-y-7 md:grid-cols-2">
            {ideas.map((item) => (
              <motion.div
                key={item.title}
                variants={fadeUp}
                initial="hidden"
                whileInView="show"
                viewport={{ once: true, amount: 0.2 }}
                transition={{ duration: 0.45, ease: "easeOut" }}
              >
                <h3 className="text-lg font-semibold tracking-tight text-slate-900">
                  {item.title}
                </h3>
                <p className="mt-3 text-sm leading-7 text-slate-500">
                  {item.text}
                </p>
              </motion.div>
            ))}
          </div>
        </section>

        <section className="mt-16">
          <div className="mb-8 flex items-end justify-between gap-6">
            <div>
              <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
                How the review moves
              </p>
              <h3 className="mt-3 max-w-2xl text-3xl font-semibold tracking-tight">
                From uploaded image to governed decision.
              </h3>
            </div>

            <p className="hidden max-w-sm text-sm italic leading-7 text-slate-500 md:block">
              The system moves like a careful assistant: first checking, then thinking,
              then explaining, and only then deciding.
            </p>
          </div>

          <div className="relative">
            <div className="absolute left-[15px] top-0 hidden h-full w-px bg-gradient-to-b from-red-200 via-slate-200 to-sky-200 md:block" />

            <div className="grid gap-7 md:grid-cols-2 lg:grid-cols-4">
              {flow.map(([step, detail], index) => (
                <motion.div
                key={step}
                variants={fadeUp}
                initial="hidden"
                whileInView="show"
                viewport={{ once: true, amount: 0.2 }}
                transition={{ duration: 0.45, ease: "easeOut" }}
                className="relative"
              >
                  <div className="mb-4 flex items-center gap-3">
                    <span className="flex h-8 w-8 items-center justify-center rounded-full bg-white text-xs font-semibold text-red-500 shadow-[0_10px_30px_rgba(15,23,42,0.08)]">
                      {index + 1}
                    </span>
                    <div className="h-px flex-1 bg-gradient-to-r from-red-200 to-transparent" />
                  </div>

                  <h3 className="text-lg font-semibold tracking-tight text-slate-900">
                    {step}
                  </h3>
                  <p className="mt-2 text-sm leading-7 text-slate-500">
                    {detail}
                  </p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        <section className="mt-14">
          <div className="mb-6">
            <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
              What the system produces
            </p>
          </div>

          <div className="grid gap-6 lg:grid-cols-3">
            <motion.div
              whileHover={{ y: -4 }}
              transition={{ duration: 0.2 }}
              className="relative pl-9"
            >
              <ShieldCheck size={20} className="absolute left-0 top-1 text-red-500" />
              <h3 className="text-base font-semibold text-slate-900">Governed decision</h3>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                Clear policy action: ANSWER, ESCALATE, REQUEST_EVIDENCE, REFUSE, or STOP.
              </p>
            </motion.div>

            <motion.div
              whileHover={{ y: -4 }}
              transition={{ duration: 0.2 }}
              className="relative pl-9"
            >
              <Sparkles size={20} className="absolute left-0 top-1 text-red-500" />
              <h3 className="text-base font-semibold text-slate-900">Visual explanation</h3>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                Grad-CAM++ shows where the model focused and whether the area looks reasonable.
              </p>
            </motion.div>

            <motion.div
              whileHover={{ y: -4 }}
              transition={{ duration: 0.2 }}
              className="relative pl-9"
            >
              <FileText size={20} className="absolute left-0 top-1 text-red-500" />
              <h3 className="text-base font-semibold text-slate-900">Audit-ready report</h3>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                Route, output, quality, OOD, uncertainty, heatmap, and policy reason.
              </p>
            </motion.div>
          </div>
        </section>

        <SiteFooter />
      </div>
    </main>
  );
}
