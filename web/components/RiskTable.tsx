"use client";

import { useState } from "react";
import { api, type ChunkDetail, type RiskFactorOut } from "@/lib/api";
import { CATEGORY_LABELS, filingText } from "@/lib/format";

export function Categories({ values }: { values: string[] }) {
  return (
    <>
      {values.map((c) => (
        <span key={c} className="chip">
          {CATEGORY_LABELS[c] ?? c}
        </span>
      ))}
    </>
  );
}

/** Citation buttons open the chunk text and a link to the filing on EDGAR. */
export function Citations({ ids }: { ids: number[] }) {
  const [chunk, setChunk] = useState<ChunkDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function open(id: number) {
    setError(null);
    try {
      setChunk(await api.chunk(id));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <>
      {ids.map((id) => (
        <button key={id} className="link small" onClick={() => void open(id)} aria-label={`Open cited chunk ${id}`}>
          [{id}]
        </button>
      ))}
      {error && <span className="small" style={{ color: "var(--critical)" }}> {error}</span>}
      {chunk && (
        <aside className="panel" role="dialog" aria-label={`Chunk ${chunk.chunk_id}`}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h2>
              {chunk.ticker} FY{chunk.fiscal_year} · chunk {chunk.chunk_id}
            </h2>
            <button onClick={() => setChunk(null)}>Close</button>
          </div>
          {chunk.heading && <h3>{chunk.heading}</h3>}
          <p className="small muted">Quoted from the 10-K. This is the text the summary cites.</p>
          <pre>{filingText(chunk.text)}</pre>
          <a href={chunk.filing_url} target="_blank" rel="noreferrer">
            Open the filing on EDGAR
          </a>
        </aside>
      )}
    </>
  );
}

export function RiskTable({ factors }: { factors: RiskFactorOut[] }) {
  const [sort, setSort] = useState<"filing" | "severity">("filing");
  const rows = [...factors].sort((a, b) =>
    sort === "severity" ? (b.severity ?? 0) - (a.severity ?? 0) || a.factor_id - b.factor_id : a.factor_id - b.factor_id,
  );
  return (
    <>
      <div className="row small" style={{ marginBottom: 8 }}>
        Sort:
        <button className={sort === "filing" ? "primary" : ""} onClick={() => setSort("filing")}>
          Filing order
        </button>
        <button className={sort === "severity" ? "primary" : ""} onClick={() => setSort("severity")}>
          Severity
        </button>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th style={{ width: "34%" }}>Risk factor</th>
              <th>Categories</th>
              <th>Severity</th>
              <th>Summary</th>
              <th>Cites</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((f) => (
              <tr key={f.factor_id}>
                <td>
                  {f.heading ?? <span className="muted">(untitled section)</span>}
                  {f.section_heading && <div className="small muted">{f.section_heading}</div>}
                </td>
                <td>{f.status === "classified" ? <Categories values={f.categories} /> : <span className="muted small">{f.status}</span>}</td>
                <td>{f.severity ?? "—"}</td>
                <td className="small">{f.summary ?? "—"}</td>
                <td>
                  <Citations ids={f.cited_chunk_ids} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
