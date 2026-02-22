import type { Metadata } from "next";
import Link from "next/link";
import { IBM_Plex_Mono, Space_Grotesk } from "next/font/google";

import "./globals.css";

const grotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-grotesk",
});
const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "ADS - Attribution Density Scanner",
  description: "Groundedness detector powered by attribution density",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${grotesk.variable} ${plexMono.variable}`}>
        <div className="mx-auto max-w-6xl px-4 pb-16 pt-6 md:px-6">
          <header className="sticky top-4 z-20 mb-8 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-[var(--ads-border)] bg-white/82 px-4 py-3 shadow-soft backdrop-blur-md">
            <Link
              href="/"
              className="text-lg font-bold tracking-tight text-slate-900"
            >
              ADS <span className="text-xs text-slate-500">v0.1</span>
            </Link>
            <nav className="flex flex-wrap gap-2 text-sm text-slate-700">
              <Link
                href="/demo"
                className="rounded-lg px-3 py-1.5 hover:bg-slate-100"
              >
                Demo
              </Link>
              <Link
                href="/analysis"
                className="rounded-lg px-3 py-1.5 hover:bg-slate-100"
              >
                Analysis
              </Link>
              <Link
                href="/scan"
                className="rounded-lg px-3 py-1.5 hover:bg-slate-100"
              >
                Scan
              </Link>
              <Link
                href="/docs"
                className="rounded-lg px-3 py-1.5 hover:bg-slate-100"
              >
                Docs
              </Link>
            </nav>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}
