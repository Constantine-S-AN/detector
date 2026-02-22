import * as React from "react";

import { cn } from "@/lib/utils";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "outline";
};

export function Button({
  className,
  variant = "default",
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center rounded-xl px-4 py-2 text-sm font-semibold transition focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-600/40 disabled:cursor-not-allowed disabled:opacity-50",
        variant === "default" &&
          "bg-gradient-to-r from-cyan-700 to-cyan-500 text-white shadow-[0_10px_24px_rgba(15,157,184,0.28)] hover:from-cyan-600 hover:to-cyan-500",
        variant === "outline" &&
          "border border-slate-300 bg-white/90 text-slate-800 hover:bg-white",
        className,
      )}
      {...props}
    />
  );
}
