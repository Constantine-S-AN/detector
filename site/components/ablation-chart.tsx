"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type AblationRow = {
  feature: string;
  roc_auc?: number | null;
  pr_auc?: number | null;
};

type Props = {
  rows: AblationRow[];
};

export function AblationChart({ rows }: Props) {
  const data = rows.map((row) => ({
    feature: row.feature,
    roc_auc: row.roc_auc ?? 0,
    pr_auc: row.pr_auc ?? 0,
  }));

  return (
    <div className="h-80 w-full rounded-2xl border border-slate-200 bg-white p-4 shadow-soft">
      <p className="mb-2 text-sm font-semibold text-slate-700">
        Ablation Leaderboard
      </p>
      <ResponsiveContainer width="100%" height="90%">
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="feature"
            angle={-25}
            textAnchor="end"
            height={60}
            interval={0}
          />
          <YAxis domain={[0, 1]} />
          <Tooltip />
          <Bar dataKey="roc_auc" name="ROC-AUC" />
          <Bar dataKey="pr_auc" name="PR-AUC" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
