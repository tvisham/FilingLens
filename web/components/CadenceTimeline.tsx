"use client";

import { useState } from "react";
import type { FilingRef } from "@/lib/api";

const FORMS = ["10-K", "10-Q", "8-K", "DEF 14A"] as const;
const W = 1040; // ~ rendered width of a full-width card
const ROW = 30;
const PAD = { left: 70, right: 12, top: 8, bottom: 24 };

/** Filing dates on a shared time axis, one row per form type (rows carry identity, so the
 * marks need only one color). */
export function CadenceTimeline({ filings, start, end }: { filings: FilingRef[]; start: string; end: string }) {
  const [hover, setHover] = useState<FilingRef | null>(null);
  const t0 = Date.parse(start);
  const t1 = Date.parse(end);
  const H = PAD.top + FORMS.length * ROW + PAD.bottom;
  const x = (d: string) => PAD.left + ((Date.parse(d) - t0) / (t1 - t0 || 1)) * (W - PAD.left - PAD.right);
  const years: number[] = [];
  for (let yr = new Date(t0).getUTCFullYear() + 1; yr <= new Date(t1).getUTCFullYear(); yr++) years.push(yr);
  const rowOf = (f: FilingRef) => FORMS.indexOf((f.form.replace("/A", "") as (typeof FORMS)[number]));

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Filing timeline by form type">
        {years.map((yr) => (
          <g key={yr}>
            <line x1={x(`${yr}-01-01`)} x2={x(`${yr}-01-01`)} y1={PAD.top} y2={H - PAD.bottom} stroke="var(--grid)" />
            <text x={x(`${yr}-01-01`)} y={H - 6} fontSize={11} textAnchor="middle" fill="var(--text-muted)">
              {yr}
            </text>
          </g>
        ))}
        {FORMS.map((form, r) => (
          <g key={form}>
            <line x1={PAD.left} x2={W - PAD.right} y1={PAD.top + r * ROW + ROW / 2} y2={PAD.top + r * ROW + ROW / 2} stroke="var(--grid)" />
            <text x={PAD.left - 10} y={PAD.top + r * ROW + ROW / 2 + 4} fontSize={12} textAnchor="end" fill="var(--text-secondary)">
              {form}
            </text>
          </g>
        ))}
        {filings
          .filter((f) => rowOf(f) >= 0)
          .map((f) => {
            const cy = PAD.top + rowOf(f) * ROW + ROW / 2;
            const amended = f.form.endsWith("/A");
            return (
              <g key={f.accession}>
                <circle
                  cx={x(f.filing_date)}
                  cy={cy}
                  r={5}
                  fill={amended ? "var(--surface-1)" : "var(--series-1)"}
                  stroke={amended ? "var(--series-1)" : "var(--surface-1)"}
                  strokeWidth={2}
                />
                <circle
                  cx={x(f.filing_date)}
                  cy={cy}
                  r={11}
                  fill="transparent"
                  onMouseEnter={() => setHover(f)}
                  onMouseLeave={() => setHover(null)}
                />
              </g>
            );
          })}
      </svg>
      {hover && (
        <div className="tooltip" style={{ left: `${(x(hover.filing_date) / W) * 100}%`, top: PAD.top + rowOf(hover) * ROW }}>
          <strong>{hover.form}</strong> filed {hover.filing_date}
          {hover.report_date ? ` (period ${hover.report_date})` : ""}
          {hover.items.length > 0 && <> · items {hover.items.join(", ")}</>}
          <br />
          <span className="muted mono">{hover.accession}</span>
        </div>
      )}
      <p className="small muted" style={{ margin: "4px 0 0" }}>
        Filled dot: original filing. Hollow dot: amendment.
      </p>
    </div>
  );
}
