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
        "inline-flex items-center rounded-full px-3 py-1 text-xs font-medium",
        variant === "default" && "bg-slate-900 text-white",
        variant === "secondary" && "bg-teal-100 text-teal-900",
        variant === "outline" && "border border-slate-300 text-slate-700",
        className,
      )}
      {...props}
    />
  );
}
