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
      <section className="rounded-3xl border border-slate-200 bg-white/85 p-8 shadow-soft md:p-12">
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.15em] text-teal-700">
          Attribution Density Scanner
        </p>
        <h1 className="mb-4 text-4xl font-bold tracking-tight text-slate-900 md:text-6xl">
          Groundedness detection from influence density
        </h1>
        <p className="max-w-3xl text-base leading-relaxed text-slate-700 md:text-lg">
          ADS turns attribution distributions into actionable groundedness
          scores. It ships both FULL mode (local API + real-time scan) and
          STATIC mode (precomputed demo assets for GitHub Pages).
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <Link
            href="/demo"
            className="rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white"
          >
            Explore Demo
          </Link>
          <Link
            href="/scan"
            className="rounded-xl border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-800"
          >
            Try Scan
          </Link>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
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

      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-soft">
        <h2 className="mb-3 text-2xl font-semibold">Quickstart</h2>
        <div className="grid gap-2 rounded-xl bg-slate-950 p-4 text-sm text-slate-100">
          {quickstart.map((command) => (
            <code key={command}>{command}</code>
          ))}
        </div>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-soft">
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
