"use client";

import { useState } from "react";

export type TrendPoint = { runId: string; at: string; value: number; passed: boolean | null };

const W = 240;
const H = 110;
const PAD = { top: 12, right: 10, bottom: 18, left: 38 };

/** Small multiple: one metric over runs, with its threshold as a dashed rule. */
export function MetricTrend({ metric, points, threshold }: { metric: string; points: TrendPoint[]; threshold: Record<string, number> }) {
  const [hover, setHover] = useState<number | null>(null);
  const bounds = [threshold.min, threshold.max].filter((v): v is number => typeof v === "number");
  const values = [...points.map((p) => p.value), ...bounds];
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (lo === hi) {
    lo -= Math.abs(lo) * 0.1 || 1;
    hi += Math.abs(hi) * 0.1 || 1;
  }
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (i: number) => PAD.left + (points.length === 1 ? plotW / 2 : (i / (points.length - 1)) * plotW);
  const y = (v: number) => PAD.top + ((hi - v) / (hi - lo)) * plotH;
  const fmt = (v: number) => (Math.abs(v) >= 1000 ? `${Math.round(v / 1000)}k` : Number(v.toFixed(2)).toString());
  const line = points.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.value)}`).join(" ");

  return (
    <figure className="chart card" style={{ margin: 0 }}>
      <figcaption className="small" style={{ fontWeight: 600 }}>
        {metric}
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${metric} over ${points.length} runs`}>
        {[lo, hi].map((v) => (
          <text key={v} x={PAD.left - 6} y={y(v) + 4} fontSize={10} textAnchor="end" fill="var(--text-muted)">
            {fmt(v)}
          </text>
        ))}
        <line x1={PAD.left} x2={W - PAD.right} y1={y(lo)} y2={y(lo)} stroke="var(--axis)" />
        {bounds.map((b) => (
          <line key={b} x1={PAD.left} x2={W - PAD.right} y1={y(b)} y2={y(b)} stroke="var(--text-muted)" strokeDasharray="4 3" />
        ))}
        {points.length > 1 && <path d={line} fill="none" stroke="var(--series-1)" strokeWidth={2} />}
        {points.map((p, i) => (
          <g key={p.runId}>
            <circle cx={x(i)} cy={y(p.value)} r={4} fill="var(--series-1)" stroke="var(--surface-1)" strokeWidth={2} />
            <rect
              x={x(i) - 10}
              y={PAD.top}
              width={20}
              height={plotH}
              fill="transparent"
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            />
          </g>
        ))}
        {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={H - PAD.bottom} stroke="var(--axis)" />}
      </svg>
      {hover !== null && points[hover] && (
        <div className="tooltip" style={{ left: `${(x(hover) / W) * 100}%`, top: 28 }}>
          {fmt(points[hover].value)} · {points[hover].passed === null ? "skipped" : points[hover].passed ? "pass" : "fail"}
          <br />
          <span className="muted">{points[hover].at.slice(0, 16).replace("T", " ")}</span>
        </div>
      )}
      <p className="small muted" style={{ margin: 0 }}>
        Dashed: threshold ({Object.entries(threshold).map(([k, v]) => `${k} ${v}`).join(", ") || "none"})
      </p>
    </figure>
  );
}
