import Image from "next/image";
import Link from "next/link";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { withBasePath } from "@/lib/base-path";

const quickstart = ["make setup", "make demo", "make build-site"];

export default function HomePage() {
  return (
    <main className="space-y-12">
      <section className="animate-rise rounded-3xl border border-[var(--ads-border)] bg-white/88 p-8 shadow-soft md:p-12">
        <div className="grid gap-8 lg:grid-cols-[1.15fr_0.85fr]">
          <div>
            <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-cyan-700">
              Attribution Density Scanner
            </p>
            <h1 className="mb-4 text-4xl font-bold tracking-tight text-slate-900 md:text-6xl">
              Groundedness from influence geometry
            </h1>
            <p className="max-w-2xl text-base leading-relaxed text-slate-700 md:text-lg">
              ADS converts attribution distributions into calibrated
              groundedness signals. Use FULL mode for local real-time scans and
              STATIC mode for deployable, reproducible demos.
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <Link
                href="/demo"
                className="rounded-xl bg-gradient-to-r from-cyan-700 to-cyan-500 px-4 py-2 text-sm font-semibold text-white shadow-[0_10px_24px_rgba(15,157,184,0.28)]"
              >
                Explore Demo
              </Link>
              <Link
                href="/scan"
                className="rounded-xl border border-slate-300 bg-white/90 px-4 py-2 text-sm font-semibold text-slate-800"
              >
                Try Scan
              </Link>
            </div>
          </div>
          <div className="grid gap-3">
            <div className="rounded-2xl border border-cyan-200 bg-cyan-50/80 p-4">
              <p className="text-xs uppercase text-cyan-800">Signal</p>
              <p className="mt-2 text-2xl font-bold text-cyan-900">
                H@K + concentration
              </p>
            </div>
            <div className="rounded-2xl border border-orange-200 bg-orange-50/80 p-4">
              <p className="text-xs uppercase text-orange-800">Detectors</p>
              <p className="mt-2 text-2xl font-bold text-orange-900">
                threshold + logistic
              </p>
            </div>
            <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
              <p className="text-xs uppercase text-slate-600">Artifacts</p>
              <p className="mt-2 text-2xl font-bold text-slate-900">
                JSON / CSV / PNG / SVG
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="animate-rise animate-delay-1 grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Why</CardTitle>
            <CardDescription>
              Hallucination signals are often hidden in attribution spread.
            </CardDescription>
          </CardHeader>
          <CardContent>
            Faithful answers usually show concentrated influence, while
            hallucinated outputs look diffuse.
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>How</CardTitle>
            <CardDescription>
              Backend-agnostic attribution &rarr; density features &rarr;
              detector.
            </CardDescription>
          </CardHeader>
          <CardContent>
            Features include entropy at top-k, top-share concentration, Gini,
            and abstain safeguards.
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Results</CardTitle>
            <CardDescription>
              Reproducible plots and static report artifacts.
            </CardDescription>
          </CardHeader>
          <CardContent>
            ROC/PR/calibration + per-sample influence tables are exported as
            JSON/CSV/PNG/SVG.
          </CardContent>
        </Card>
      </section>

      <section className="animate-rise animate-delay-2 rounded-2xl border border-[var(--ads-border)] bg-white/88 p-6 shadow-soft">
        <h2 className="mb-3 text-2xl font-semibold">Quickstart</h2>
        <div className="grid gap-2 rounded-xl bg-slate-950 p-4 text-sm text-slate-100 shadow-inner">
          {quickstart.map((command) => (
            <code key={command}>{command}</code>
          ))}
        </div>
      </section>

      <section className="rounded-2xl border border-[var(--ads-border)] bg-white/88 p-6 shadow-soft">
        <h2 className="mb-3 text-2xl font-semibold">Architecture</h2>
        <Image
          src={withBasePath("/ads-architecture.svg")}
          alt="ADS architecture diagram"
          width={920}
          height={260}
          unoptimized
          className="w-full"
        />
      </section>
    </main>
  );
}
