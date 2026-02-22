import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

type PageIntroProps = {
  eyebrow?: string;
  title: string;
  description: string;
  className?: string;
  aside?: ReactNode;
};

export function PageIntro({
  eyebrow,
  title,
  description,
  className,
  aside,
}: PageIntroProps) {
  return (
    <section
      className={cn(
        "animate-rise rounded-2xl border border-[var(--ads-border)] bg-white/88 p-6 shadow-soft",
        className,
      )}
    >
      <div className="grid gap-4 md:grid-cols-[1.2fr_0.8fr] md:items-end">
        <div>
          {eyebrow && (
            <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-cyan-700">
              {eyebrow}
            </p>
          )}
          <h1 className="text-3xl font-bold tracking-tight text-slate-900 md:text-4xl">
            {title}
          </h1>
          <p className="mt-2 text-sm leading-relaxed text-slate-600 md:text-base">
            {description}
          </p>
        </div>
        {aside && <div className="md:justify-self-end">{aside}</div>}
      </div>
    </section>
  );
}
