"use client";

import Link from "next/link";
import SiteHeader from "../_components/SiteHeader";
import SiteFooter from "../_components/SiteFooter";
import {
  ArrowRight,
  CircleHelp,
  Home,
  LayoutDashboard,
  Mail,
  MapPin,
  Play,
  ShieldCheck,
} from "lucide-react";

export default function ContactPage() {
  return (
    <main
      className="relative min-h-screen overflow-hidden bg-[#fffaf3] text-slate-950"
      style={{ fontFamily: '"Aptos","Aptos Body","Segoe UI",Arial,sans-serif' }}
    >
      <div className="pointer-events-none motion-float absolute -right-40 -top-40 h-[480px] w-[480px] rounded-full bg-red-100/70 blur-3xl" />
      <div className="pointer-events-none absolute -left-44 top-[520px] h-[480px] w-[480px] rounded-full bg-sky-100/70 blur-3xl" />

      <div className="relative mx-auto max-w-7xl px-6 py-6">
        <SiteHeader active="contact" />

        <section className="grid min-h-[430px] gap-12 lg:grid-cols-[minmax(0,1fr)_360px] lg:items-center motion-fade-up-slow">
          <div>
            <p className="text-xs uppercase tracking-[0.28em] text-red-500">
              Project communication
            </p>

            <h2 className="mt-5 max-w-3xl text-4xl font-semibold leading-tight tracking-tight">
              Send questions, feedback, or demo notes.
            </h2>

            <p className="mt-6 max-w-2xl text-sm italic leading-7 text-slate-500">
              Use this page for academic communication about the prototype, UI,
              workflow, safety behavior, or implementation choices.
            </p>

            <div className="mt-8 flex flex-wrap gap-3">
              <a
                href="mailto:research@medaix.ai"
                className="group inline-flex items-center gap-2 rounded-2xl border border-red-200 bg-red-50 px-5 py-4 text-sm font-semibold text-red-700 transition motion-hover-lift hover:bg-red-500 hover:text-white"
              >
                Email project team
                <ArrowRight size={17} className="transition group-hover:translate-x-1" />
              </a>

              <Link
                href="/demo"
                className="inline-flex items-center gap-2 rounded-2xl border border-slate-300 px-5 py-4 text-sm font-semibold text-slate-700 transition motion-hover-lift hover:border-red-300 hover:text-red-500"
              >
                Open demo
                <Play size={17} />
              </Link>
            </div>
          </div>

          <aside className="space-y-10">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
                Email
              </p>
              <a
                href="mailto:research@medaix.ai"
                className="mt-2 block text-2xl font-semibold text-slate-900 transition hover:text-red-500"
              >
                research@medaix.ai
              </a>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                Best for project questions, demo review, and academic feedback.
              </p>
            </div>

            <div className="grid gap-8 sm:grid-cols-2">
              <div>
                <MapPin size={18} className="text-red-500" />
                <p className="mt-3 text-xs uppercase tracking-[0.18em] text-slate-500">
                  Location
                </p>
                <p className="mt-2 text-sm text-slate-600">Ankara, Türkiye</p>
              </div>

              <div>
                <ShieldCheck size={18} className="text-red-500" />
                <p className="mt-3 text-xs uppercase tracking-[0.18em] text-slate-500">
                  Scope
                </p>
                <p className="mt-2 text-sm text-slate-600">Research-use only</p>
              </div>
            </div>
          </aside>
        </section>

        <section className="mt-10 grid gap-10 border-t border-slate-200 pt-8 md:grid-cols-[260px_minmax(0,1fr)]">
          <div>
            <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
              Helpful to include
            </p>
            <p className="mt-4 text-sm leading-7 text-slate-500">
              Clear context makes feedback easier to understand.
            </p>
          </div>

          <div className="grid gap-8 md:grid-cols-3">
            <div className="motion-hover-lift">
              <Mail size={18} className="text-red-500" />
              <h3 className="mt-3 text-base font-semibold text-slate-900">
                Page or feature
              </h3>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                Mention Home, Demo, Workspace, report export, batch review, or audit.
              </p>
            </div>

            <div className="motion-hover-lift">
              <Play size={18} className="text-red-500" />
              <h3 className="mt-3 text-base font-semibold text-slate-900">
                Demo case
              </h3>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                If it is about demo behavior, include which scenario was run.
              </p>
            </div>

            <div className="motion-hover-lift">
              <CircleHelp size={18} className="text-red-500" />
              <h3 className="mt-3 text-base font-semibold text-slate-900">
                Expected result
              </h3>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                Say what felt unclear, incorrect, too dense, or missing.
              </p>
            </div>
          </div>
        </section>

        <SiteFooter />
      </div>
    </main>
  );
}
