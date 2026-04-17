import Link from "next/link";

export default function NotFound() {
  return (
    <main
      className="flex min-h-screen items-center justify-center bg-[#FAFAF7] px-6 text-zinc-900"
      style={{ fontFamily: '"Aptos","Aptos Body","Segoe UI",Arial,sans-serif' }}
    >
      <div className="max-w-lg text-center">
        <p className="text-[11px] uppercase tracking-[0.28em] text-zinc-400">
          404
        </p>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight">
          Page not found
        </h1>
        <p className="mt-4 text-[15px] leading-8 text-zinc-600">
          The page you are looking for does not exist or may have been moved.
        </p>

        <Link
          href="/"
          className="mt-8 inline-block rounded-lg bg-red-500 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-red-600"
        >
          Go home
        </Link>
      </div>
    </main>
  );
}