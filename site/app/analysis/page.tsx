"use client";

import Image from "next/image";
import { useEffect, useState } from "react";

import { AnalysisChart } from "@/components/analysis-charts";
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
    };
    ablation?: Array<{
      feature: string;
      roc_auc?: number | null;
      pr_auc?: number | null;
    }>;
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

  return (
    <main className="space-y-6">
      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-soft">
        <h1 className="text-3xl font-bold">Analysis</h1>
        <p className="mt-2 text-sm text-slate-600">
          ROC / PR / Calibration and score distributions.
        </p>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
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
      </section>

      <section className="grid gap-4 md:grid-cols-2">
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
