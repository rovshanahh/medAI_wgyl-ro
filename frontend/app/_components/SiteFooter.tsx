"use client";

import Link from "next/link";
import { CircleHelp, Home, LayoutDashboard, Play, ShieldCheck } from "lucide-react";

function MiniLogo() {
  return (
    <div className="relative h-5 w-8">
      <span className="absolute left-0 top-1/2 h-1.5 w-1.5 -translate-y-1/2 rounded-full bg-slate-950" />
      <span className="absolute right-0 top-1/2 h-1.5 w-1.5 -translate-y-1/2 rounded-full bg-slate-950" />
      <span className="absolute left-1/2 top-1/2 h-px w-4 -translate-x-1/2 -translate-y-1/2 rounded-full bg-slate-400" />
    </div>
  );
}

export default function SiteFooter() {
  return (
    <footer className="mt-16 border-t border-slate-200 py-8">
      <div className="grid gap-8 md:grid-cols-3">
        <div>
          <div className="mb-3 flex items-center gap-3">
            <MiniLogo />
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
            <Link href="/" className="flex items-center gap-2 hover:text-red-500"><Home size={15} />Home</Link>
            <Link href="/demo" className="flex items-center gap-2 hover:text-red-500"><Play size={15} />Demo</Link>
            <Link href="/workspace" className="flex items-center gap-2 hover:text-red-500"><LayoutDashboard size={15} />Workspace</Link>
            <Link href="/about" className="flex items-center gap-2 hover:text-red-500"><CircleHelp size={15} />About</Link>
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
  );
}
