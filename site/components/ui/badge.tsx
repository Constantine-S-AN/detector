import * as React from "react";

import { cn } from "@/lib/utils";

type BadgeProps = React.HTMLAttributes<HTMLSpanElement> & {
  variant?: "default" | "secondary" | "outline";
};

export function Badge({
  className,
  variant = "default",
  ...props
}: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold",
        variant === "default" && "bg-slate-900 text-white",
        variant === "secondary" && "bg-cyan-100 text-cyan-900",
        variant === "outline" &&
          "border border-slate-300 bg-white/80 text-slate-700",
        className,
      )}
      {...props}
    />
  );
}
