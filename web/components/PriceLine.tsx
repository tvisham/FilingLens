"use client";

import { useState } from "react";
import type { PricePoint } from "@/lib/api";

const W = 1040;
const H = 220;
const PAD = { top: 16, right: 64, bottom: 26, left: 56 };

/** Daily closes, one series. The last close is direct-labelled; hover shows each day. */
export function PriceLine({ points }: { points: PricePoint[] }) {
  const [hover, setHover] = useState<number | null>(null);
  if (points.length < 2) return <p className="muted">Not enough price history.</p>;

  const closes = points.map((p) => Number(p.close));
  let lo = Math.min(...closes);
  let hi = Math.max(...closes);
  const pad = (hi - lo) * 0.08 || 1;
  lo -= pad;
  hi += pad;
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (i: number) => PAD.left + (i / (points.length - 1)) * plotW;
  const y = (v: number) => PAD.top + ((hi - v) / (hi - lo)) * plotH;
  const path = closes.map((c, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(c).toFixed(1)}`).join(" ");
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => lo + t * (hi - lo));
  const months = points
    .map((p, i) => ({ i, label: p.day.slice(0, 7) }))
    .filter((m, k, all) => k === 0 || m.label !== all[k - 1]?.label);
  const last = closes.length - 1;
  const step = plotW / (points.length - 1);

  return (
    <figure className="chart" style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Daily closing price, ${points.length} trading days`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
            <text x={PAD.left - 8} y={y(t) + 4} textAnchor="end" fontSize={11} fill="var(--text-muted)">
              ${t.toFixed(0)}
            </text>
          </g>
        ))}
        {months.map((m) => (
          <text key={m.i} x={x(m.i)} y={H - 6} fontSize={11} fill="var(--text-secondary)">
            {new Date(`${m.label}-01T00:00:00Z`).toLocaleString("en-US", { month: "short", timeZone: "UTC" })}
          </text>
        ))}
        <path d={path} fill="none" stroke="var(--series-1)" strokeWidth={2} strokeLinejoin="round" />
        <circle cx={x(last)} cy={y(closes[last] ?? 0)} r={4} fill="var(--series-1)" stroke="var(--surface-1)" strokeWidth={2} />
        <text x={x(last) + 8} y={y(closes[last] ?? 0) + 4} fontSize={12} fill="var(--text-primary)">
          ${closes[last]?.toFixed(2)}
        </text>
        {hover !== null && (
          <>
            <line x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={H - PAD.bottom} stroke="var(--axis)" />
            <circle cx={x(hover)} cy={y(closes[hover] ?? 0)} r={4} fill="var(--series-1)" stroke="var(--surface-1)" strokeWidth={2} />
          </>
        )}
        <rect
          x={PAD.left}
          y={PAD.top}
          width={plotW}
          height={plotH}
          fill="transparent"
          onMouseMove={(e) => {
            const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
            const vx = ((e.clientX - box.left) / box.width) * W;
            setHover(Math.max(0, Math.min(points.length - 1, Math.round((vx - PAD.left) / step))));
          }}
          onMouseLeave={() => setHover(null)}
        />
      </svg>
      {hover !== null && points[hover] && (
        <div className="tooltip" style={{ left: `${(x(hover) / W) * 100}%`, top: 20 }}>
          <strong>{points[hover].day}</strong> · close ${Number(points[hover].close).toFixed(2)}
          <br />
          <span className="muted">volume {points[hover].volume.toLocaleString("en-US")}</span>
        </div>
      )}
      <details>
        <summary>Table view</summary>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Day</th>
                <th>Close (USD)</th>
                <th>Volume</th>
              </tr>
            </thead>
            <tbody>
              {[...points].reverse().map((p) => (
                <tr key={p.day}>
                  <td>{p.day}</td>
                  <td className="mono">{Number(p.close).toFixed(2)}</td>
                  <td className="mono">{p.volume.toLocaleString("en-US")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
