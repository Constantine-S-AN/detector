import type { Metadata } from "next";
import Link from "next/link";
import { IBM_Plex_Mono, Space_Grotesk } from "next/font/google";

import { TopNav } from "@/components/top-nav";

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
  description:
    "Attribution Density Scanner: groundedness detection from attribution distribution geometry.",
  keywords: [
    "groundedness",
    "hallucination detection",
    "attribution",
    "machine learning",
    "evaluation",
  ],
  openGraph: {
    title: "ADS - Attribution Density Scanner",
    description:
      "Groundedness detection from attribution distribution geometry.",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "ADS - Attribution Density Scanner",
    description:
      "Groundedness detection from attribution distribution geometry.",
  },
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
            <TopNav />
          </header>
          {children}
          <footer className="mt-10 rounded-2xl border border-[var(--ads-border)] bg-white/75 px-4 py-3 text-xs text-slate-600 shadow-soft">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p>
                ADS portfolio demo. STATIC mode data in{" "}
                <code>site/public/demo</code>.
              </p>
              <div className="flex items-center gap-3">
                <a
                  href="https://github.com/Constantine-S-AN/detector"
                  target="_blank"
                  rel="noreferrer"
                  className="hover:text-slate-900"
                >
                  Repository
                </a>
                <Link href="/docs" className="hover:text-slate-900">
                  Documentation
                </Link>
              </div>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
