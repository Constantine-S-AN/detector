"use client";

import { useEffect, useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { withBasePath } from "@/lib/base-path";
import type { DemoIndexItem } from "@/types/demo";

type ScanResult = {
  prediction: {
    groundedness_score: number;
    predicted_label: number;
    confidence: number;
    abstain_flag: boolean;
  };
  features: Record<string, number | boolean | string>;
  top_influential: Array<{ train_id: string; score: number; text: string }>;
};

export default function ScanPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_BASE;
  const [mode, setMode] = useState<"static" | "full">(
    apiBase ? "full" : "static",
  );
  const [prompt, setPrompt] = useState(
    "Provide a grounded answer about the Pacific Ocean.",
  );
  const [answer, setAnswer] = useState(
    "According to the provided sources, the Pacific Ocean is the largest ocean on Earth.",
  );
  const [result, setResult] = useState<ScanResult | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [examples, setExamples] = useState<DemoIndexItem[]>([]);
  const [selectedSample, setSelectedSample] = useState<string>("");

  useEffect(() => {
    fetch(withBasePath("/demo/index.json"))
      .then((response) => response.json())
      .then((data: { examples: DemoIndexItem[] }) => {
        setExamples(data.examples);
        if (data.examples.length > 0) {
          setSelectedSample(data.examples[0].sample_id);
        }
      })
      .catch((error: unknown) => {
        const message =
          error instanceof Error
            ? error.message
            : "Failed to load demo examples.";
        setErrorMessage(message);
      });
  }, []);

  async function runScan() {
    setIsRunning(true);
    setErrorMessage(null);
    if (mode === "full" && !apiBase) {
      setErrorMessage(
        "FULL mode requires NEXT_PUBLIC_API_BASE to be configured.",
      );
      setIsRunning(false);
      return;
    }
    if (mode === "full" && apiBase) {
      try {
        const response = await fetch(`${apiBase}/scan`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt,
            answer,
            top_k: 20,
            method: "logistic",
            backend: "toy",
          }),
        });
        if (!response.ok) {
          throw new Error(`API request failed (${response.status}).`);
        }
        const data = (await response.json()) as ScanResult;
        setResult(data);
      } catch (error: unknown) {
        const message =
          error instanceof Error
            ? error.message
            : "Failed to run FULL mode scan.";
        setErrorMessage(message);
      } finally {
        setIsRunning(false);
      }
      return;
    }

    if (!selectedSample) {
      setIsRunning(false);
      return;
    }
    const selected = examples.find((item) => item.sample_id === selectedSample);
    if (!selected) {
      setIsRunning(false);
      return;
    }
    try {
      const response = await fetch(withBasePath(selected.detail_path));
      const detail = (await response.json()) as {
        prediction: ScanResult["prediction"];
        features: ScanResult["features"];
        top_influential: ScanResult["top_influential"];
      };
      setResult({
        prediction: detail.prediction,
        features: detail.features,
        top_influential: detail.top_influential,
      });
    } catch (error: unknown) {
      const message =
        error instanceof Error
          ? error.message
          : "Failed to run STATIC mode scan.";
      setErrorMessage(message);
    } finally {
      setIsRunning(false);
    }
  }

  const modeHint = useMemo(() => {
    if (mode === "full") {
      return apiBase
        ? `FULL mode via ${apiBase}`
        : "FULL mode requires NEXT_PUBLIC_API_BASE";
    }
    return "STATIC mode from precomputed demo assets";
  }, [apiBase, mode]);

  return (
    <main className="space-y-6">
      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-soft">
        <h1 className="text-3xl font-bold">Scan</h1>
        <p className="mt-2 text-sm text-slate-600">
          Run ADS in FULL mode (local API) or STATIC mode (demo assets).
        </p>
      </section>

      <Card>
        <CardHeader>
          <CardTitle>Mode</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex gap-2">
            <Button
              variant={mode === "static" ? "default" : "outline"}
              onClick={() => setMode("static")}
            >
              STATIC
            </Button>
            <Button
              variant={mode === "full" ? "default" : "outline"}
              onClick={() => setMode("full")}
            >
              FULL
            </Button>
            <Badge variant="outline">{modeHint}</Badge>
          </div>

          {mode === "full" ? (
            <>
              <Input
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
              />
              <Textarea
                rows={4}
                value={answer}
                onChange={(event) => setAnswer(event.target.value)}
              />
            </>
          ) : (
            <select
              className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm"
              value={selectedSample}
              onChange={(event) => setSelectedSample(event.target.value)}
            >
              {examples.map((item) => (
                <option key={item.sample_id} value={item.sample_id}>
                  {item.sample_id} · {item.label}
                </option>
              ))}
            </select>
          )}

          <Button onClick={() => void runScan()} disabled={isRunning}>
            {isRunning ? "Running..." : "Run Scan"}
          </Button>
          {errorMessage && (
            <p className="rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700">
              {errorMessage}
            </p>
          )}
          {isRunning && (
            <p className="rounded-xl bg-slate-100 p-3 text-sm text-slate-600">
              Running scan...
            </p>
          )}
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <CardTitle>Result</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="grid gap-2 md:grid-cols-4">
              <div className="rounded-xl bg-teal-50 p-3">
                <p className="text-xs text-slate-500">Groundedness</p>
                <p className="text-lg font-semibold">
                  {result.prediction.groundedness_score.toFixed(3)}
                </p>
              </div>
              <div className="rounded-xl bg-slate-100 p-3">
                <p className="text-xs text-slate-500">Confidence</p>
                <p className="text-lg font-semibold">
                  {result.prediction.confidence.toFixed(3)}
                </p>
              </div>
              <div className="rounded-xl bg-slate-100 p-3">
                <p className="text-xs text-slate-500">Predicted Label</p>
                <p className="text-lg font-semibold">
                  {result.prediction.predicted_label === 1
                    ? "faithful"
                    : "hallucinated"}
                </p>
              </div>
              <div className="rounded-xl bg-slate-100 p-3">
                <p className="text-xs text-slate-500">Abstain</p>
                <p className="text-lg font-semibold">
                  {result.prediction.abstain_flag ? "yes" : "no"}
                </p>
              </div>
            </div>

            <div className="rounded-xl border border-slate-200">
              <table className="w-full text-left">
                <thead className="bg-slate-100 text-xs uppercase text-slate-500">
                  <tr>
                    <th className="px-3 py-2">Train ID</th>
                    <th className="px-3 py-2">Score</th>
                    <th className="px-3 py-2">Text</th>
                  </tr>
                </thead>
                <tbody>
                  {result.top_influential.slice(0, 8).map((row) => (
                    <tr
                      key={row.train_id}
                      className="border-t border-slate-100 align-top"
                    >
                      <td className="px-3 py-2 font-mono text-xs">
                        {row.train_id}
                      </td>
                      <td className="px-3 py-2">{row.score.toFixed(3)}</td>
                      <td className="px-3 py-2 text-slate-600">{row.text}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </main>
  );
}
