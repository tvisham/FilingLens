"use client";

import { useCallback, useEffect, useState } from "react";
import { ErrorBanner } from "@/components/Status";
import { api, type ReviewItemDetail, type ReviewItemOut } from "@/lib/api";

type Issue = { check: string; severity: string; message: string; path?: string };

function readToken(): string {
  try {
    return window.sessionStorage.getItem("filinglens-api-token") ?? "";
  } catch {
    return "";
  }
}

export default function ReviewPage() {
  const [status, setStatus] = useState("pending");
  const [items, setItems] = useState<ReviewItemOut[]>([]);
  const [selected, setSelected] = useState<ReviewItemDetail | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [reviewer, setReviewer] = useState("");
  const [reason, setReason] = useState("");
  const [edited, setEdited] = useState("");
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      setItems(await api.reviews(status));
    } catch (e) {
      setError(e);
    }
  }, [status]);

  useEffect(() => setToken(readToken()), []);
  useEffect(() => void load(), [load]);

  async function open(id: number) {
    setNotice(null);
    try {
      const d = await api.review(id);
      setSelected(d);
      setEdited(JSON.stringify(d.edited_output ?? d.output, null, 2));
    } catch (e) {
      setError(e);
    }
  }

  async function decide(decision: "approve" | "edit" | "reject") {
    if (!selected) return;
    let editedOutput: Record<string, unknown> | null = null;
    if (decision === "edit") {
      try {
        editedOutput = JSON.parse(edited) as Record<string, unknown>;
      } catch {
        setNotice("The edited output is not valid JSON.");
        return;
      }
    }
    setBusy(true);
    try {
      window.sessionStorage.setItem("filinglens-api-token", token);
    } catch {
      /* storage unavailable */
    }
    try {
      const d = await api.decide(selected.id, { decision, reviewer, reason, edited_output: editedOutput }, token);
      setSelected(d);
      setNotice(`#${d.id} ${d.status}.`);
      setReason("");
      await load();
    } catch (e) {
      setNotice(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const issues = ((selected?.guardrails?.issues as Issue[] | undefined) ?? []).filter(Boolean);
  const pending = selected?.status === "pending";
  const canDecide = pending && reviewer.trim() !== "" && reason.trim() !== "" && !busy;

  return (
    <>
      <h1>Review queue</h1>
      <p className="muted">Agent runs the guardrails flagged. Every decision is recorded in the audit log.</p>
      {error ? <ErrorBanner error={error} /> : null}
      <div className="row" style={{ marginBottom: 12 }}>
        <label htmlFor="status" style={{ margin: 0 }}>
          Status
        </label>
        <select id="status" value={status} onChange={(e) => setStatus(e.target.value)} style={{ width: "auto" }}>
          {["pending", "approved", "edited", "rejected", "all"].map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
        <span className="muted small">{items.length} item(s)</span>
      </div>

      <div className="grid-2" style={{ alignItems: "start" }}>
        <section className="card">
          {items.length === 0 ? (
            <p className="muted">Nothing here.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Run</th>
                  <th>Severity</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {items.map((it) => (
                  <tr key={it.id} style={{ background: selected?.id === it.id ? "var(--chip)" : undefined }}>
                    <td>
                      <button className="link" onClick={() => void open(it.id)}>
                        {it.id}
                      </button>
                    </td>
                    <td className="small">
                      {it.scenario}
                      <br />
                      <span className="muted">{Object.values(it.input).join(", ")}</span>
                    </td>
                    <td>
                      <span className={`status ${it.severity === "block" ? "fail" : "review"}`}>
                        <span className="dot" aria-hidden />
                        {it.severity === "block" ? "Blocked" : "Review"}
                      </span>
                    </td>
                    <td className="small">{it.reason.slice(0, 160)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        {selected && (
          <section className="card">
            <h2>
              #{selected.id} · {selected.scenario} · {selected.status}
            </h2>
            <ul className="small" style={{ paddingLeft: 18 }}>
              {issues.map((i, n) => (
                <li key={n}>
                  <strong>{i.severity}</strong> {i.check}: {i.message}
                  {i.path ? <span className="muted mono"> ({i.path})</span> : null}
                </li>
              ))}
            </ul>
            {selected.decision_reason && (
              <p className="small">
                Decided by {selected.reviewer}: {selected.decision_reason}
              </p>
            )}
            <label htmlFor="output">Output {pending ? "(edit, then choose Save edit)" : ""}</label>
            <textarea id="output" value={edited} onChange={(e) => setEdited(e.target.value)} readOnly={!pending} />
            {pending && (
              <>
                <label htmlFor="reviewer">Reviewer</label>
                <input id="reviewer" value={reviewer} onChange={(e) => setReviewer(e.target.value)} />
                <label htmlFor="reason">Reason (required)</label>
                <input id="reason" value={reason} onChange={(e) => setReason(e.target.value)} />
                <label htmlFor="token">API token (if the server requires one)</label>
                <input id="token" type="password" value={token} onChange={(e) => setToken(e.target.value)} />
                <div className="row" style={{ marginTop: 12 }}>
                  <button className="primary" disabled={!canDecide} onClick={() => void decide("approve")}>
                    Approve
                  </button>
                  <button disabled={!canDecide} onClick={() => void decide("edit")}>
                    Save edit
                  </button>
                  <button className="danger" disabled={!canDecide} onClick={() => void decide("reject")}>
                    Reject
                  </button>
                </div>
              </>
            )}
            {notice && <p className="small">{notice}</p>}
          </section>
        )}
      </div>
    </>
  );
}
