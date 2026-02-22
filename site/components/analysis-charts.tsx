"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Point = {
  x: number;
  y: number;
};

type Props = {
  title: string;
  points: Point[];
  xLabel: string;
  yLabel: string;
};

export function AnalysisChart({ title, points, xLabel, yLabel }: Props) {
  return (
    <div className="h-72 w-full rounded-2xl border border-slate-200 bg-white p-4 shadow-soft">
      <p className="mb-2 text-sm font-semibold text-slate-700">{title}</p>
      <ResponsiveContainer width="100%" height="90%">
        <LineChart data={points}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="x"
            label={{ value: xLabel, position: "insideBottom", offset: -4 }}
          />
          <YAxis
            label={{ value: yLabel, angle: -90, position: "insideLeft" }}
          />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="y"
            dot={false}
            strokeWidth={2}
            name={title}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
