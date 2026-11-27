"use client";

import { useState } from "react";
import type { QuarterValue } from "@/lib/api";
import { quarterLabel, usd, usdExact } from "@/lib/format";

const W = 520; // ~ rendered width in a half-width card, so text renders near its nominal size
const H = 220;
const PAD = { top: 22, right: 8, bottom: 28, left: 56 };

/** One measure per chart (no dual axes). The latest `highlight` quarters are emphasised
 * and direct-labelled; older ones use a lighter step of the same hue. */
export function QuarterBars({ title, quarters, highlight = 2 }: { title: string; quarters: QuarterValue[]; highlight?: number }) {
  const [hover, setHover] = useState<number | null>(null);
  if (quarters.length === 0) return <p className="muted">No quarterly data.</p>;

  const values = quarters.map((q) => Number(q.value_usd));
  const max = Math.max(0, ...values);
  const min = Math.min(0, ...values);
  const span = max - min || 1;
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const y = (v: number) => PAD.top + ((max - v) / span) * plotH;
  const step = plotW / quarters.length;
  const barW = Math.max(6, Math.min(48, step - 2)); // >= 2px surface gap between bars
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => min + t * span);
  const zero = y(0);

  return (
    <figure className="chart" style={{ margin: 0 }}>
      <figcaption>
        <h3>{title}</h3>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}, last ${quarters.length} fiscal quarters`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} stroke="var(--grid)" strokeWidth={1} />
            <text x={PAD.left - 8} y={y(t) + 4} textAnchor="end" fontSize={11} fill="var(--text-muted)">
              {usd(t, 0)}
            </text>
          </g>
        ))}
        <line x1={PAD.left} x2={W - PAD.right} y1={zero} y2={zero} stroke="var(--axis)" strokeWidth={1} />
        {quarters.map((q, i) => {
          const v = values[i] ?? 0;
          const x = PAD.left + i * step + (step - barW) / 2;
          const top = Math.min(y(v), zero);
          const h = Math.max(1, Math.abs(zero - y(v)));
          const latest = i >= quarters.length - highlight;
          const r = Math.min(4, h, barW / 2);
          // 4px rounded data-end, square at the baseline.
          const d =
            v >= 0
              ? `M${x},${zero} V${top + r} Q${x},${top} ${x + r},${top} H${x + barW - r} Q${x + barW},${top} ${x + barW},${top + r} V${zero} Z`
              : `M${x},${zero} V${top + h - r} Q${x},${top + h} ${x + r},${top + h} H${x + barW - r} Q${x + barW},${top + h} ${x + barW},${top + h - r} V${zero} Z`;
          return (
            <g key={q.period_end}>
              <path d={d} fill={latest ? "var(--series-1)" : "var(--series-1-muted)"} />
              {/* hit target wider and taller than the mark */}
              <rect
                x={PAD.left + i * step}
                y={PAD.top}
                width={step}
                height={plotH}
                fill="transparent"
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)}
                onFocus={() => setHover(i)}
                onBlur={() => setHover(null)}
                tabIndex={0}
                aria-label={`${quarterLabel(q.fiscal_year, q.fiscal_quarter)}: ${usd(v)}`}
              />
              {latest && (
                <text x={x + barW / 2} y={v >= 0 ? top - 6 : top + h + 14} textAnchor="middle" fontSize={11} fill="var(--text-primary)">
                  {usd(v, 1)}
                </text>
              )}
              <text x={x + barW / 2} y={H - 8} textAnchor="middle" fontSize={11} fill="var(--text-secondary)">
                {quarterLabel(q.fiscal_year, q.fiscal_quarter)}
              </text>
            </g>
          );
        })}
      </svg>
      {hover !== null && quarters[hover] && (
        <div className="tooltip" style={{ left: `${((PAD.left + hover * step + step / 2) / W) * 100}%`, top: 24 }}>
          <strong>{quarterLabel(quarters[hover].fiscal_year, quarters[hover].fiscal_quarter)}</strong> ·{" "}
          {quarters[hover].period_start} to {quarters[hover].period_end}
          <br />
          {usdExact(quarters[hover].value_usd)}
          {quarters[hover].derived ? " (derived: FY − 9M)" : ""}
          <br />
          <span className="muted mono">{quarters[hover].accession}</span>
        </div>
      )}
      <details>
        <summary>Table view</summary>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Quarter</th>
                <th>Period</th>
                <th>Value (USD)</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {quarters.map((q) => (
                <tr key={q.period_end}>
                  <td>{quarterLabel(q.fiscal_year, q.fiscal_quarter)}</td>
                  <td>
                    {q.period_start} to {q.period_end}
                  </td>
                  <td className="mono">
                    {usdExact(q.value_usd)}
                    {q.derived ? " (derived)" : ""}
                  </td>
                  <td className="mono">{q.accession}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
