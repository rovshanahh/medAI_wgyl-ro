"use client";

import Link from "next/link";

export default function ReviewPage() {
  return (
    <main className="min-h-screen bg-[#fbfaf6] px-6 py-6 text-slate-950">
      <div className="mx-auto max-w-5xl">
        <header className="flex items-center justify-between border-b border-slate-300 pb-5">
          <div>
            <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
              Review workspace
            </p>
            <h1 className="mt-1 text-2xl font-semibold">
              Governed Medical Image Analysis
            </h1>
          </div>

          <Link href="/" className="text-sm text-slate-600 hover:text-slate-950">
            Back to Home
          </Link>
        </header>

        <section className="py-16">
          <p className="text-sm text-slate-500">Coming next</p>
          <h2 className="mt-3 text-4xl font-semibold tracking-tight">
            A cleaner review page will be rebuilt here.
          </h2>
          <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-600">
            The previous experimental design was removed. This page is now a safe
            placeholder, so the route exists without breaking the app.
          </p>
        </section>
      </div>
    </main>
  );
}
