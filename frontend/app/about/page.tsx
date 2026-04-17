"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import {
  CircleHelp,
  House,
  LayoutDashboard,
  Mail,
  MapPin,
  Phone,
  ShieldPlus,
} from "lucide-react";

const fadeUp = {
  hidden: { opacity: 0, y: 10 },
  show: { opacity: 1, y: 0 },
};

export default function AboutPage() {
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
                Research-use platform
              </p>
              <h1 className="mt-1 text-3xl font-semibold tracking-tight">MedAIx</h1>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-5 text-sm text-zinc-600">
            <Link href="/" className="inline-flex items-center gap-2 transition hover:text-zinc-900">
              <House size={16} />
              Home
            </Link>

            <Link href="/workspace" className="inline-flex items-center gap-2 transition hover:text-zinc-900">
              <LayoutDashboard size={16} />
              Workspace
            </Link>

            <Link href="/about" className="inline-flex items-center gap-2 transition hover:text-zinc-900">
              <CircleHelp size={16} />
              About
            </Link>

            <Link href="/contact" className="inline-flex items-center gap-2 transition hover:text-zinc-900">
              <Mail size={16} />
              Contact
            </Link>
          </div>
        </nav>

        <motion.section
          className="mb-14 grid gap-10 lg:grid-cols-[1.15fr_0.85fr] lg:items-end"
          initial="hidden"
          animate="show"
          transition={{ staggerChildren: 0.1 }}
        >
          <div>
            <motion.div
              variants={fadeUp}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="mb-4 inline-flex items-center gap-2 rounded-full bg-white px-4 py-2 text-xs text-zinc-600 shadow-[0_6px_20px_rgba(0,0,0,0.03)]"
            >
              <ShieldPlus size={15} className="text-red-500" />
              About MedAIx
            </motion.div>

            <motion.h2
              variants={fadeUp}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="text-3xl font-semibold tracking-tight sm:text-4xl"
            >
              A governed research interface for AI-based medical image analysis.
            </motion.h2>

            <motion.p
              variants={fadeUp}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="mt-5 max-w-2xl text-base leading-8 text-zinc-600"
            >
              MedAIx is designed to present model predictions, uncertainty-related
              signals, and visual explanation outputs within a controlled review
              workflow. The system is intended for research and educational use and
              is not designed for clinical decision-making.
            </motion.p>
          </div>

          <motion.div
            variants={fadeUp}
            transition={{ duration: 0.4, ease: "easeOut" }}
            className="text-sm leading-7 text-zinc-600"
          >
          </motion.div>
        </motion.section>

        <section className="mb-14 border-t border-zinc-200 pt-6">
          <div className="grid gap-8 md:grid-cols-3">
            <div>
              <p className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                Purpose
              </p>
              <p className="mt-3 text-sm leading-7 text-zinc-700">
                To provide a structured environment for examining model outputs,
                supporting evidence, and system behavior in a research setting.
              </p>
            </div>

            <div>
              <p className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                Scope
              </p>
              <p className="mt-3 text-sm leading-7 text-zinc-700">
                The current prototype focuses on image validation, model inference,
                result presentation, and visual explanation artifacts.
              </p>
            </div>

            <div>
              <p className="text-xs uppercase tracking-[0.18em] text-zinc-400">
                Limitation
              </p>
              <p className="mt-3 text-sm leading-7 text-zinc-700">
                Outputs are experimental and must not be interpreted as diagnosis,
                treatment advice, or clinical recommendation.
              </p>
            </div>
          </div>
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
              Research-use assistant for careful, non-diagnostic review of medical images.
            </p>
          </div>

          <div>
            <h4 className="mb-3 text-sm font-semibold text-zinc-900">Quick links</h4>
            <div className="space-y-2 text-sm text-zinc-600">
              <Link href="/" className="inline-flex items-center gap-2 hover:text-zinc-900">
                <House size={16} />
                Home
              </Link>
              <br />
              <Link href="/workspace" className="inline-flex items-center gap-2 hover:text-zinc-900">
                <LayoutDashboard size={16} />
                Workspace
              </Link>
              <br />
              <Link href="/about" className="inline-flex items-center gap-2 hover:text-zinc-900">
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
              <span className="font-semibold text-zinc-700">Important notice.</span>{" "}
              This platform is intended solely for research and educational use. The
              information presented here is non-diagnostic and must not be used for
              clinical decision-making.
            </p>
          </div>
        </div>
      </footer>
    </main>
  );
}