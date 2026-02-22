import Image from "next/image";
import Link from "next/link";

import { MetricTile } from "@/components/metric-tile";
import { PageIntro } from "@/components/page-intro";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { withBasePath } from "@/lib/base-path";

const quickstart = ["make setup", "make demo", "make build-site"];

const pillars = [
  {
    title: "Problem",
    body: "Hallucination signals often hide in attribution spread. Dense concentration is typically associated with faithful responses.",
  },
  {
    title: "Method",
    body: "Backend-agnostic top-k attribution is transformed into density features and passed to threshold/logistic detectors.",
  },
  {
    title: "Reproducibility",
    body: "The pipeline exports JSON/CSV/PNG/SVG artifacts and static demo assets to support deterministic portfolio demos.",
  },
];

export default function HomePage() {
  return (
    <main className="space-y-8">
      <PageIntro
        eyebrow="Attribution Density Scanner"
        title="Groundedness from influence geometry"
        description="ADS turns attribution distributions into calibrated groundedness signals. Use FULL mode for local real-time scan workflows and STATIC mode for deployable, reproducible showcase demos."
        aside={
          <div className="rounded-xl border border-cyan-200 bg-cyan-50/80 px-4 py-3 text-right shadow-soft">
            <p className="text-xs uppercase tracking-wide text-cyan-800">Current Demo</p>
            <p className="text-sm font-semibold text-cyan-950">
              v0.1 · 40 samples · seed 42
            </p>
          </div>
        }
      />

      <section className="animate-rise animate-delay-1 rounded-3xl border border-[var(--ads-border)] bg-white/88 p-8 shadow-soft md:p-10">
        <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
          <div>
            <p className="mb-3 text-xs font-semibold uppercase tracking-[0.2em] text-cyan-700">
              Portfolio-ready detector stack
            </p>
            <h2 className="mb-4 text-3xl font-bold tracking-tight text-slate-900 md:text-5xl">
              Influence concentration as a groundedness signal
            </h2>
            <p className="max-w-2xl text-base leading-relaxed text-slate-700 md:text-lg">
              The core idea is simple: faithful answers tend to show peaked
              influence on relevant training evidence, while hallucinated
              outputs look diffuse. ADS quantifies this geometry and calibrates
              a practical detector.
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <Link
                href="/demo"
                className="rounded-xl bg-gradient-to-r from-cyan-700 to-cyan-500 px-4 py-2 text-sm font-semibold text-white shadow-[0_10px_24px_rgba(15,157,184,0.28)]"
              >
                Explore Demo
              </Link>
              <Link
                href="/analysis"
                className="rounded-xl border border-slate-300 bg-white/90 px-4 py-2 text-sm font-semibold text-slate-800"
              >
                View Analysis
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
            <MetricTile
              label="Signal"
              value="H@K + concentration"
              hint="entropy, top1/top5 share, gini, effective_k"
              tone="cyan"
            />
            <MetricTile
              label="Detectors"
              value="threshold + logistic"
              hint="abstain guard for distributed-truth edge cases"
              tone="orange"
            />
            <MetricTile
              label="Artifacts"
              value="JSON / CSV / PNG / SVG"
              hint="static demo assets + reproducibility manifest"
              tone="slate"
            />
          </div>
        </div>
      </section>

      <section className="animate-rise animate-delay-2 grid gap-4 md:grid-cols-3">
        {pillars.map((entry) => (
          <Card key={entry.title}>
            <CardHeader>
              <CardTitle>{entry.title}</CardTitle>
            </CardHeader>
            <CardContent className="text-sm leading-relaxed text-slate-700">
              {entry.body}
            </CardContent>
          </Card>
        ))}
      </section>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricTile label="ROC-AUC" value="1.000" hint="toy controlled set" />
        <MetricTile label="PR-AUC" value="1.000" hint="toy controlled set" />
        <MetricTile label="ECE" value="0.0159" hint="calibration quality" />
        <MetricTile label="Coverage" value="1.000" hint="with default floor" />
      </section>

      <section className="rounded-2xl border border-[var(--ads-border)] bg-white/88 p-6 shadow-soft">
        <h3 className="mb-3 text-xl font-semibold text-slate-900">Quickstart</h3>
        <div className="grid gap-2 rounded-xl bg-slate-950 p-4 text-sm text-slate-100 shadow-inner">
          {quickstart.map((command) => (
            <code key={command}>{command}</code>
          ))}
        </div>
        <p className="mt-3 text-sm text-slate-600">
          Outputs are exported to <code>artifacts/</code>,{" "}
          <code>site/public/demo/</code>, and <code>site/out/</code>.
        </p>
      </section>

      <section className="rounded-2xl border border-[var(--ads-border)] bg-white/88 p-6 shadow-soft">
        <h3 className="mb-3 text-xl font-semibold text-slate-900">Architecture</h3>
        <Image
          src={withBasePath("/ads-architecture.svg")}
          alt="ADS architecture diagram"
          width={920}
          height={260}
          unoptimized
          className="w-full rounded-xl border border-slate-200 bg-white"
        />
      </section>
    </main>
  );
}
