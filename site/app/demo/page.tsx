"use client";

import { useEffect, useMemo, useState } from "react";
import { Search } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { withBasePath } from "@/lib/base-path";
import type { DemoDetail, DemoIndexItem } from "@/types/demo";

type DemoIndexPayload = {
  examples: DemoIndexItem[];
  count: number;
};

export default function DemoPage() {
  const [indexPayload, setIndexPayload] = useState<DemoIndexPayload | null>(
    null,
  );
  const [selected, setSelected] = useState<DemoDetail | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [labelFilter, setLabelFilter] = useState<
    "all" | "faithful" | "hallucinated"
  >("all");

  useEffect(() => {
    const controller = new AbortController();
    fetch(withBasePath("/demo/index.json"), { signal: controller.signal })
      .then((response) => response.json())
      .then((data: DemoIndexPayload) => {
        setIndexPayload(data);
        if (data.examples.length > 0) {
          void loadDetail(data.examples[0].detail_path);
        }
        setIsLoading(false);
      })
      .catch((error: unknown) => {
        const message =
          error instanceof Error ? error.message : "Failed to load demo index.";
        setErrorMessage(message);
        setIsLoading(false);
      });
    return () => controller.abort();
  }, []);

  async function loadDetail(path: string) {
    try {
      const response = await fetch(withBasePath(path));
      const payload = (await response.json()) as DemoDetail;
      setSelected(payload);
      setErrorMessage(null);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : "Failed to load demo detail.";
      setErrorMessage(message);
    }
  }

  const filtered = useMemo(() => {
    const all = indexPayload?.examples ?? [];
    return all.filter((item) => {
      const matchQuery = item.prompt_preview
        .toLowerCase()
        .includes(query.toLowerCase());
      const matchLabel =
        labelFilter === "all" ? true : item.label === labelFilter;
      return matchQuery && matchLabel;
    });
  }, [indexPayload, query, labelFilter]);

  return (
    <main className="space-y-6">
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-soft">
        <h1 className="text-3xl font-bold">Demo Gallery</h1>
        <p className="mt-2 text-sm text-slate-600">
          Browse faithful vs hallucinated samples and inspect attribution
          density signatures.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.1fr_1.2fr]">
        <Card>
          <CardHeader>
            <CardTitle>Examples</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
              <Input
                className="pl-9"
                placeholder="Search prompt"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
            </div>
            <div className="flex gap-2">
              {(["all", "faithful", "hallucinated"] as const).map((entry) => (
                <Button
                  key={entry}
                  variant={labelFilter === entry ? "default" : "outline"}
                  onClick={() => setLabelFilter(entry)}
                >
                  {entry}
                </Button>
              ))}
            </div>
            <div className="max-h-[28rem] space-y-2 overflow-auto pr-1">
              {isLoading && (
                <p className="rounded-xl bg-slate-100 p-3 text-sm text-slate-600">
                  Loading examples...
                </p>
              )}
              {errorMessage && (
                <p className="rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                  {errorMessage}
                </p>
              )}
              {!isLoading && filtered.length === 0 && (
                <p className="rounded-xl bg-slate-100 p-3 text-sm text-slate-600">
                  No examples match this filter.
                </p>
              )}
              {filtered.map((item) => (
                <button
                  key={item.sample_id}
                  className="w-full rounded-xl border border-slate-200 p-3 text-left hover:bg-slate-50"
                  onClick={() => void loadDetail(item.detail_path)}
                >
                  <div className="mb-1 flex items-center justify-between">
                    <Badge
                      variant={
                        item.label === "faithful" ? "secondary" : "outline"
                      }
                    >
                      {item.label}
                    </Badge>
                    <span className="text-xs text-slate-500">
                      score {item.groundedness_score.toFixed(3)}
                    </span>
                  </div>
                  <p className="text-sm text-slate-700">
                    {item.prompt_preview}
                  </p>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Demo Detail</CardTitle>
          </CardHeader>
          <CardContent>
            {!selected && <p className="text-sm text-slate-600">Loading...</p>}
            {selected && (
              <div className="space-y-5">
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                    <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-500">
                      Prompt
                    </p>
                    <p className="text-sm text-slate-700">{selected.prompt}</p>
                  </div>
                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                    <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-500">
                      Answer
                    </p>
                    <p className="text-sm text-slate-700">{selected.answer}</p>
                  </div>
                </div>

                <div className="grid gap-2 md:grid-cols-4">
                  <div className="rounded-xl bg-teal-50 p-3">
                    <p className="text-xs text-slate-500">Groundedness</p>
                    <p className="text-lg font-semibold">
                      {selected.prediction.groundedness_score.toFixed(3)}
                    </p>
                  </div>
                  <div className="rounded-xl bg-orange-50 p-3">
                    <p className="text-xs text-slate-500">Confidence</p>
                    <p className="text-lg font-semibold">
                      {selected.prediction.confidence.toFixed(3)}
                    </p>
                  </div>
                  <div className="rounded-xl bg-slate-100 p-3">
                    <p className="text-xs text-slate-500">Top1</p>
                    <p className="text-lg font-semibold">
                      {Number(selected.features.top1_share).toFixed(3)}
                    </p>
                  </div>
                  <div className="rounded-xl bg-slate-100 p-3">
                    <p className="text-xs text-slate-500">Entropy</p>
                    <p className="text-lg font-semibold">
                      {Number(selected.features.entropy_top_k).toFixed(3)}
                    </p>
                  </div>
                </div>

                <div className="rounded-xl border border-slate-200 p-3">
                  <p className="mb-2 text-sm font-semibold">
                    Top Influential Samples
                  </p>
                  <div className="h-56 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={selected.top_influential
                          .slice(0, 8)
                          .map((item, index) => ({
                            name: `#${index + 1}`,
                            score: item.score,
                          }))}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="score" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-3 max-h-52 overflow-auto rounded-lg border border-slate-200">
                    <table className="w-full text-sm">
                      <thead className="bg-slate-100 text-left text-xs uppercase text-slate-500">
                        <tr>
                          <th className="px-3 py-2">train_id</th>
                          <th className="px-3 py-2">score</th>
                          <th className="px-3 py-2">text</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selected.top_influential.slice(0, 12).map((item) => (
                          <tr
                            key={item.train_id}
                            className="border-t border-slate-100 align-top"
                          >
                            <td className="px-3 py-2 font-mono text-xs">
                              {item.train_id}
                            </td>
                            <td className="px-3 py-2">
                              {item.score.toFixed(3)}
                            </td>
                            <td className="px-3 py-2 text-slate-600">
                              {item.text}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
