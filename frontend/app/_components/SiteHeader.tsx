"use client";

import Link from "next/link";
import { CircleHelp, Home, LayoutDashboard, Mail, Play } from "lucide-react";

type SiteHeaderProps = {
  active?: "home" | "demo" | "workspace" | "about" | "contact";
};

function MiniLogo() {
  return (
    <div className="relative h-5 w-8">
      <span className="absolute left-0 top-1/2 h-1.5 w-1.5 -translate-y-1/2 rounded-full bg-slate-950" />
      <span className="absolute right-0 top-1/2 h-1.5 w-1.5 -translate-y-1/2 rounded-full bg-slate-950" />
      <span className="absolute left-1/2 top-1/2 h-px w-4 -translate-x-1/2 -translate-y-1/2 rounded-full bg-slate-400" />
    </div>
  );
}

function navClass(active: boolean) {
  return active
    ? "inline-flex items-center gap-2 font-medium text-slate-950"
    : "inline-flex items-center gap-2 text-slate-600 transition hover:text-red-500";
}

export default function SiteHeader({ active = "home" }: SiteHeaderProps) {
  return (
    <nav className="mb-12 flex flex-col gap-5 md:flex-row md:items-center md:justify-between motion-fade-up">
      <div className="flex items-center gap-4">
        <MiniLogo />
        <div>
          <p className="text-[11px] uppercase tracking-[0.25em] text-slate-500">
            Research-use platform
          </p>
          <h1 className="text-lg font-semibold text-slate-900">
            Governed Medical Image Analysis
          </h1>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-5 text-sm">
        <Link href="/" className={navClass(active === "home")}><Home size={16} />Home</Link>
        <Link href="/demo" className={navClass(active === "demo")}><Play size={16} />Demo</Link>
        <Link href="/workspace" className={navClass(active === "workspace")}><LayoutDashboard size={16} />Workspace</Link>
        <Link href="/about" className={navClass(active === "about")}><CircleHelp size={16} />About</Link>
        <Link href="/contact" className={navClass(active === "contact")}><Mail size={16} />Contact</Link>
      </div>
    </nav>
  );
}
