"use client";

import Image from "next/image";
import { useEffect, useState } from "react";

import { AblationChart } from "@/components/ablation-chart";
import { AnalysisChart } from "@/components/analysis-charts";
import { ConfusionMatrix } from "@/components/confusion-matrix";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { withBasePath } from "@/lib/base-path";

type AnalysisPayload = {
  metrics: {
    roc_auc?: number;
    pr_auc?: number;
    brier?: number;
    ece?: number;
    coverage?: number;
    accuracy_when_answered?: number;
    curves?: {
      roc?: Array<{ x: number; y: number }>;
      pr?: Array<{ x: number; y: number }>;
      calibration?: Array<{ x: number; y: number }>;
      abstain?: Array<{ x: number; y: number; threshold: number }>;
    };
    ablation?: Array<{
      feature: string;
      roc_auc?: number | null;
      pr_auc?: number | null;
    }>;
  };
  summary?: {
    num_samples: number;
    num_faithful: number;
    num_hallucinated: number;
    mean_score_faithful: number;
    mean_score_hallucinated: number;
    tp: number;
    tn: number;
    fp: number;
    fn: number;
  };
  plot_refs: Record<string, string>;
};

export default function AnalysisPage() {
  const [payload, setPayload] = useState<AnalysisPayload | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    fetch(withBasePath("/demo/analysis.json"))
      .then((response) => response.json())
      .then((data: AnalysisPayload) => setPayload(data))
      .catch((error: unknown) => {
        const message =
          error instanceof Error
            ? error.message
            : "Failed to load analysis data.";
        setLoadError(message);
      });
  }, []);

  if (loadError) {
    return (
      <main>
        <p className="text-sm text-red-600">{loadError}</p>
      </main>
    );
  }

  if (!payload) {
    return (
      <main>
        <p className="text-sm text-slate-600">Loading analysis...</p>
      </main>
    );
  }

  const metrics = payload.metrics;
  const summary = payload.summary;
  const precision =
    summary && summary.tp + summary.fp > 0
      ? summary.tp / (summary.tp + summary.fp)
      : null;
  const recall =
    summary && summary.tp + summary.fn > 0
      ? summary.tp / (summary.tp + summary.fn)
      : null;
  const f1 =
    precision != null && recall != null && precision + recall > 0
      ? (2 * precision * recall) / (precision + recall)
      : null;

  return (
    <main className="space-y-6">
      <section className="animate-rise rounded-2xl border border-[var(--ads-border)] bg-white/88 p-6 shadow-soft">
        <h1 className="text-3xl font-bold">Analysis</h1>
        <p className="mt-2 text-sm text-slate-600">
          ROC / PR / Calibration and score distributions.
        </p>
      </section>

      <section className="grid gap-4 md:grid-cols-5">
        <Card>
          <CardHeader>
            <CardTitle>ROC-AUC</CardTitle>
          </CardHeader>
          <CardContent>{metrics.roc_auc?.toFixed(4) ?? "N/A"}</CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>PR-AUC</CardTitle>
          </CardHeader>
          <CardContent>{metrics.pr_auc?.toFixed(4) ?? "N/A"}</CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>ECE</CardTitle>
          </CardHeader>
          <CardContent>{metrics.ece?.toFixed(4) ?? "N/A"}</CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Coverage</CardTitle>
          </CardHeader>
          <CardContent>{metrics.coverage?.toFixed(4) ?? "N/A"}</CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Answered Accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            {metrics.accuracy_when_answered?.toFixed(4) ?? "N/A"}
          </CardContent>
        </Card>
      </section>

      {summary && (
        <section className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
          <ConfusionMatrix
            tp={summary.tp}
            tn={summary.tn}
            fp={summary.fp}
            fn={summary.fn}
          />
          <div className="grid gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Dataset</CardTitle>
              </CardHeader>
              <CardContent>
                n={summary.num_samples} | faithful={summary.num_faithful} |
                hallu={summary.num_hallucinated}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Mean Score (Faithful / Hallu)</CardTitle>
              </CardHeader>
              <CardContent>
                {summary.mean_score_faithful.toFixed(4)} /{" "}
                {summary.mean_score_hallucinated.toFixed(4)}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Precision / Recall / F1</CardTitle>
              </CardHeader>
              <CardContent>
                {(precision ?? 0).toFixed(4)} / {(recall ?? 0).toFixed(4)} /{" "}
                {(f1 ?? 0).toFixed(4)}
              </CardContent>
            </Card>
          </div>
        </section>
      )}

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <AnalysisChart
          title="ROC"
          points={metrics.curves?.roc ?? []}
          xLabel="FPR"
          yLabel="TPR"
        />
        <AnalysisChart
          title="Precision-Recall"
          points={metrics.curves?.pr ?? []}
          xLabel="Recall"
          yLabel="Precision"
        />
        <AnalysisChart
          title="Calibration"
          points={metrics.curves?.calibration ?? []}
          xLabel="Predicted"
          yLabel="Observed"
        />
        <AnalysisChart
          title="Abstain Tradeoff"
          points={
            metrics.curves?.abstain?.map((point) => ({
              x: point.x,
              y: point.y,
            })) ?? []
          }
          xLabel="Coverage"
          yLabel="Answered Accuracy"
        />
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <Image
          src={withBasePath(payload.plot_refs.hist_faithful)}
          alt="Faithful histogram"
          width={900}
          height={560}
          unoptimized
          className="rounded-2xl border border-slate-200 bg-white p-2 shadow-soft"
        />
        <Image
          src={withBasePath(payload.plot_refs.hist_hallucinated)}
          alt="Hallucinated histogram"
          width={900}
          height={560}
          unoptimized
          className="rounded-2xl border border-slate-200 bg-white p-2 shadow-soft"
        />
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-soft">
        <h2 className="mb-3 text-xl font-semibold">
          Feature Ablation (Single-Feature AUC)
        </h2>
        <div className="mb-4">
          <AblationChart rows={metrics.ablation ?? []} />
        </div>
        <div className="overflow-auto rounded-xl border border-slate-200">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-100 text-xs uppercase text-slate-500">
              <tr>
                <th className="px-3 py-2">Feature</th>
                <th className="px-3 py-2">ROC-AUC</th>
                <th className="px-3 py-2">PR-AUC</th>
              </tr>
            </thead>
            <tbody>
              {(metrics.ablation ?? []).map((entry) => (
                <tr key={entry.feature} className="border-t border-slate-100">
                  <td className="px-3 py-2 font-mono text-xs">
                    {entry.feature}
                  </td>
                  <td className="px-3 py-2">
                    {entry.roc_auc == null ? "N/A" : entry.roc_auc.toFixed(4)}
                  </td>
                  <td className="px-3 py-2">
                    {entry.pr_auc == null ? "N/A" : entry.pr_auc.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
