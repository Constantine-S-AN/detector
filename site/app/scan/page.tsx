"use client";

import { useEffect, useMemo, useState } from "react";
import { Copy } from "lucide-react";

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
  thresholds?: {
    decision_threshold: number;
    score_threshold: number;
    max_score_floor: number;
  };
};

function asNumber(value: number | boolean | string | undefined): number {
  if (typeof value === "number") {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }
  return 0;
}

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
  const [topK, setTopK] = useState<number>(12);
  const [decisionThreshold, setDecisionThreshold] = useState<number>(0.5);
  const [scoreThreshold, setScoreThreshold] = useState<number>(0.55);
  const [maxScoreFloor, setMaxScoreFloor] = useState<number>(0.05);
  const [copyStatus, setCopyStatus] = useState<string>("");

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const params = new URLSearchParams(window.location.search);
    const modeParam = params.get("mode");
    const sampleParam = params.get("sample");
    const topKParam = params.get("top_k");
    const decisionParam = params.get("decision_threshold");
    const scoreParam = params.get("score_threshold");
    const floorParam = params.get("max_score_floor");

    if (modeParam === "static" || modeParam === "full") {
      setMode(modeParam);
    }
    if (sampleParam) {
      setSelectedSample(sampleParam);
    }
    if (topKParam) {
      const parsed = Number(topKParam);
      if (Number.isFinite(parsed)) {
        setTopK(Math.min(40, Math.max(5, Math.round(parsed))));
      }
    }
    if (decisionParam) {
      const parsed = Number(decisionParam);
      if (Number.isFinite(parsed)) {
        setDecisionThreshold(Math.min(0.9, Math.max(0.1, parsed)));
      }
    }
    if (scoreParam) {
      const parsed = Number(scoreParam);
      if (Number.isFinite(parsed)) {
        setScoreThreshold(Math.min(0.9, Math.max(0.1, parsed)));
      }
    }
    if (floorParam) {
      const parsed = Number(floorParam);
      if (Number.isFinite(parsed)) {
        setMaxScoreFloor(Math.min(0.5, Math.max(0.0, parsed)));
      }
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const params = new URLSearchParams();
    params.set("mode", mode);
    params.set("top_k", String(topK));
    params.set("decision_threshold", decisionThreshold.toFixed(2));
    params.set("score_threshold", scoreThreshold.toFixed(2));
    params.set("max_score_floor", maxScoreFloor.toFixed(2));
    if (selectedSample) {
      params.set("sample", selectedSample);
    }
    const nextUrl = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({}, "", nextUrl);
  }, [
    mode,
    selectedSample,
    topK,
    decisionThreshold,
    scoreThreshold,
    maxScoreFloor,
  ]);

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
            top_k: topK,
            method: "logistic",
            backend: "toy",
            decision_threshold: decisionThreshold,
            score_threshold: scoreThreshold,
            max_score_floor: maxScoreFloor,
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

  const derivedPrediction = useMemo(() => {
    if (!result) {
      return null;
    }
    const score = result.prediction.groundedness_score;
    const maxScore = asNumber(result.features.max_score);
    const abstainFlag =
      result.prediction.abstain_flag || maxScore < maxScoreFloor;
    if (abstainFlag) {
      return { ...result.prediction, predicted_label: 0, abstain_flag: true };
    }
    const predicted = score >= decisionThreshold ? 1 : 0;
    return {
      ...result.prediction,
      predicted_label: predicted,
      abstain_flag: false,
    };
  }, [decisionThreshold, maxScoreFloor, result]);

  const visibleInfluential = useMemo(() => {
    if (!result) {
      return [];
    }
    return result.top_influential.slice(0, Math.max(1, topK));
  }, [result, topK]);

  async function copyPermalink() {
    if (typeof window === "undefined") {
      return;
    }
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopyStatus("Permalink copied");
      setTimeout(() => setCopyStatus(""), 1200);
    } catch {
      setCopyStatus("Copy failed");
      setTimeout(() => setCopyStatus(""), 1200);
    }
  }

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

          <div className="grid gap-3 rounded-xl border border-slate-200 bg-slate-50 p-3 md:grid-cols-2">
            <label className="text-xs text-slate-600">
              top_k ({topK})
              <input
                type="range"
                min={5}
                max={40}
                step={1}
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
                className="mt-1 w-full"
              />
            </label>
            <label className="text-xs text-slate-600">
              decision_threshold ({decisionThreshold.toFixed(2)})
              <input
                type="range"
                min={0.1}
                max={0.9}
                step={0.01}
                value={decisionThreshold}
                onChange={(event) =>
                  setDecisionThreshold(Number(event.target.value))
                }
                className="mt-1 w-full"
              />
            </label>
            <label className="text-xs text-slate-600">
              score_threshold ({scoreThreshold.toFixed(2)})
              <input
                type="range"
                min={0.1}
                max={0.9}
                step={0.01}
                value={scoreThreshold}
                onChange={(event) =>
                  setScoreThreshold(Number(event.target.value))
                }
                className="mt-1 w-full"
              />
            </label>
            <label className="text-xs text-slate-600">
              max_score_floor ({maxScoreFloor.toFixed(2)})
              <input
                type="range"
                min={0}
                max={0.5}
                step={0.01}
                value={maxScoreFloor}
                onChange={(event) =>
                  setMaxScoreFloor(Number(event.target.value))
                }
                className="mt-1 w-full"
              />
            </label>
          </div>

          <Button onClick={() => void runScan()} disabled={isRunning}>
            {isRunning ? "Running..." : "Run Scan"}
          </Button>
          <Button variant="outline" onClick={() => void copyPermalink()}>
            <Copy className="mr-2 h-4 w-4" />
            Copy Permalink
          </Button>
          {copyStatus && (
            <p className="rounded-xl bg-slate-100 p-3 text-sm text-slate-700">
              {copyStatus}
            </p>
          )}
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

      {result && derivedPrediction && (
        <Card>
          <CardHeader>
            <CardTitle>Result</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="grid gap-2 md:grid-cols-4">
              <div className="rounded-xl bg-teal-50 p-3">
                <p className="text-xs text-slate-500">Groundedness</p>
                <p className="text-lg font-semibold">
                  {derivedPrediction.groundedness_score.toFixed(3)}
                </p>
              </div>
              <div className="rounded-xl bg-slate-100 p-3">
                <p className="text-xs text-slate-500">Confidence</p>
                <p className="text-lg font-semibold">
                  {derivedPrediction.confidence.toFixed(3)}
                </p>
              </div>
              <div className="rounded-xl bg-slate-100 p-3">
                <p className="text-xs text-slate-500">Predicted Label</p>
                <p className="text-lg font-semibold">
                  {derivedPrediction.predicted_label === 1
                    ? "faithful"
                    : "hallucinated"}
                </p>
              </div>
              <div className="rounded-xl bg-slate-100 p-3">
                <p className="text-xs text-slate-500">Abstain</p>
                <p className="text-lg font-semibold">
                  {derivedPrediction.abstain_flag ? "yes" : "no"}
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
                  {visibleInfluential.map((row) => (
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
