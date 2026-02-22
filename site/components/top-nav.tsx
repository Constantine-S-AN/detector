"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

const navItems = [
  { href: "/demo", label: "Demo" },
  { href: "/analysis", label: "Analysis" },
  { href: "/scan", label: "Scan" },
  { href: "/docs", label: "Docs" },
];

export function TopNav() {
  const pathname = usePathname();
  const normalizedPath = pathname.endsWith("/")
    ? pathname.slice(0, -1) || "/"
    : pathname;

  return (
    <nav className="flex flex-wrap items-center gap-2 text-sm text-slate-700">
      {navItems.map((item) => {
        const isActive =
          normalizedPath === item.href || normalizedPath.endsWith(item.href);
        return (
          <Link
            key={item.href}
            href={item.href}
            className={cn(
              "rounded-lg px-3 py-1.5 transition",
              isActive
                ? "bg-slate-900 text-white shadow-soft"
                : "hover:bg-slate-100",
            )}
          >
            {item.label}
          </Link>
        );
      })}
      <a
        href="https://github.com/Constantine-S-AN/detector"
        target="_blank"
        rel="noreferrer"
        className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 transition hover:bg-white"
      >
        GitHub
      </a>
    </nav>
  );
}
