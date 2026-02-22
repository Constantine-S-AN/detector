"use client";

type Props = {
  tp: number;
  tn: number;
  fp: number;
  fn: number;
};

function toAlpha(value: number, total: number): number {
  if (total <= 0) {
    return 0.1;
  }
  const normalized = Math.min(1, Math.max(0, value / total));
  return 0.15 + normalized * 0.75;
}

export function ConfusionMatrix({ tp, tn, fp, fn }: Props) {
  const total = tp + tn + fp + fn;

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-soft">
      <p className="mb-3 text-sm font-semibold text-slate-700">
        Confusion Matrix
      </p>
      <p className="mb-3 text-xs text-slate-500">
        Rows aggregate prediction outcomes on the held-out split.
      </p>
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div
          className="rounded-xl border border-slate-200 p-3"
          style={{
            backgroundColor: `rgba(16, 185, 129, ${toAlpha(tp, total)})`,
          }}
        >
          <p className="text-xs uppercase text-slate-600">TP</p>
          <p className="text-xl font-bold text-slate-900">{tp}</p>
        </div>
        <div
          className="rounded-xl border border-slate-200 p-3"
          style={{
            backgroundColor: `rgba(249, 115, 22, ${toAlpha(fp, total)})`,
          }}
        >
          <p className="text-xs uppercase text-slate-600">FP</p>
          <p className="text-xl font-bold text-slate-900">{fp}</p>
        </div>
        <div
          className="rounded-xl border border-slate-200 p-3"
          style={{
            backgroundColor: `rgba(239, 68, 68, ${toAlpha(fn, total)})`,
          }}
        >
          <p className="text-xs uppercase text-slate-600">FN</p>
          <p className="text-xl font-bold text-slate-900">{fn}</p>
        </div>
        <div
          className="rounded-xl border border-slate-200 p-3"
          style={{
            backgroundColor: `rgba(14, 165, 233, ${toAlpha(tn, total)})`,
          }}
        >
          <p className="text-xs uppercase text-slate-600">TN</p>
          <p className="text-xl font-bold text-slate-900">{tn}</p>
        </div>
      </div>
    </div>
  );
}
