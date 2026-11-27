import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";
import { ThemeToggle } from "@/components/ThemeToggle";
import "./globals.css";

export const metadata: Metadata = {
  title: "FilingLens",
  description: "SEC filing cadence, financials and cited risk factors.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <header className="topbar">
          <Link href="/" className="brand">
            FilingLens
          </Link>
          <nav aria-label="Main">
            <Link href="/">Companies</Link>
            <Link href="/evals">Evals</Link>
            <Link href="/review">Review queue</Link>
          </nav>
          <span className="spacer" />
          <ThemeToggle />
        </header>
        <main className="shell">
          {children}
          <p className="disclaimer">
            Figures are computed from SEC XBRL data; risk-factor summaries are generated and cite the filing text.
            For information only. Not investment advice.
          </p>
        </main>
      </body>
    </html>
  );
}
