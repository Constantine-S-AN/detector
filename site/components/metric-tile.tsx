import { cn } from "@/lib/utils";

type MetricTileProps = {
  label: string;
  value: string;
  hint?: string;
  tone?: "neutral" | "cyan" | "orange" | "slate";
  className?: string;
};

const toneClassMap: Record<NonNullable<MetricTileProps["tone"]>, string> = {
  neutral: "border-slate-200 bg-white text-slate-900",
  cyan: "border-cyan-200 bg-cyan-50 text-cyan-950",
  orange: "border-orange-200 bg-orange-50 text-orange-950",
  slate: "border-slate-300 bg-slate-100 text-slate-900",
};

export function MetricTile({
  label,
  value,
  hint,
  tone = "neutral",
  className,
}: MetricTileProps) {
  return (
    <div
      className={cn(
        "rounded-xl border p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.5)]",
        toneClassMap[tone],
        className,
      )}
    >
      <p className="text-xs uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1 text-xl font-semibold tracking-tight">{value}</p>
      {hint && <p className="mt-1 text-xs text-slate-600">{hint}</p>}
    </div>
  );
}
