"use client";

import {
  CartesianGrid,
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
  const stroke =
    title === "ROC"
      ? "#0f9db8"
      : title === "Precision-Recall"
        ? "#f97316"
        : title === "Calibration"
          ? "#334155"
          : "#0f766e";

  return (
    <div className="h-72 w-full rounded-2xl border border-slate-200 bg-white p-4 shadow-soft">
      <p className="mb-2 text-sm font-semibold text-slate-700">{title}</p>
      <ResponsiveContainer width="100%" height="90%">
        <LineChart data={points}>
          <CartesianGrid stroke="#dce7ee" strokeDasharray="4 4" />
          <XAxis
            dataKey="x"
            tick={{ fill: "#526475", fontSize: 11 }}
            stroke="#9db2c0"
            label={{ value: xLabel, position: "insideBottom", offset: -4 }}
          />
          <YAxis
            tick={{ fill: "#526475", fontSize: 11 }}
            stroke="#9db2c0"
            label={{ value: yLabel, angle: -90, position: "insideLeft" }}
          />
          <Tooltip />
          <Line
            type="monotone"
            dataKey="y"
            dot={false}
            strokeWidth={2.5}
            stroke={stroke}
            name={title}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
