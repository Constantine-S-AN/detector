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
          <header className="mb-8 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-200 bg-white/80 px-4 py-3 backdrop-blur">
            <Link href="/" className="text-lg font-bold tracking-tight">
              ADS
            </Link>
            <nav className="flex gap-4 text-sm text-slate-700">
              <Link href="/demo">Demo</Link>
              <Link href="/analysis">Analysis</Link>
              <Link href="/scan">Scan</Link>
              <Link href="/docs">Docs</Link>
            </nav>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}
