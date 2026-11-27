import { MetricTrend, type TrendPoint } from "@/components/MetricTrend";
import { ErrorBanner, Status } from "@/components/Status";
import { api, type EvalRunDetail, type EvalRunSummary } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function EvalsPage({ searchParams }: { searchParams: Promise<{ run?: string }> }) {
  const { run: selected } = await searchParams;
  let runs: EvalRunSummary[] = [];
  let detail: EvalRunDetail | null = null;
  let error: unknown = null;
  try {
    runs = await api.evals();
    const target = selected ?? runs.at(-1)?.id;
    if (target) detail = await api.evalRun(target);
  } catch (e) {
    error = e;
  }

  // One small multiple per metric (no shared axis across metrics of different scales).
  const series = new Map<string, { points: TrendPoint[]; threshold: Record<string, number> }>();
  for (const r of runs) {
    for (const m of r.metrics) {
      if (m.value === null || m.value === undefined) continue;
      const s = series.get(m.metric) ?? { points: [], threshold: m.threshold };
      s.points.push({ runId: r.id, at: r.started_at, value: m.value, passed: m.passed ?? null });
      series.set(m.metric, s);
    }
  }

  return (
    <>
      <h1>Evals</h1>
      <p className="muted">Smoke runs gate every pull request; the full suite (LLM metrics, cost, latency) runs nightly.</p>
      {error ? <ErrorBanner error={error} /> : null}
      {runs.length === 0 && !error && <p className="muted">No eval runs yet: <code>uv run filinglens eval run --suite smoke</code></p>}

      {detail && (
        <section className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h2>
              {detail.suite} run · {detail.started_at.slice(0, 16).replace("T", " ")} UTC
            </h2>
            <span className="row">
              Gate: <Status passed={detail.passed} />
              <span className="muted small">cost ${detail.cost_usd.toFixed(4)}</span>
            </span>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Value</th>
                  <th>Threshold</th>
                  <th>Cases</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {detail.metrics.map((m) => (
                  <tr key={m.metric}>
                    <td>{m.metric}</td>
                    <td className="mono">{m.value ?? "—"}</td>
                    <td className="mono">
                      {Object.entries(m.threshold)
                        .map(([k, v]) => `${k} ${v}`)
                        .join(", ")}
                    </td>
                    <td>{m.n_cases}</td>
                    <td>
                      <Status passed={m.passed} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <h3 style={{ marginTop: 16 }}>Failed cases ({detail.failed_cases.length})</h3>
          {detail.failed_cases.length === 0 ? (
            <p className="muted small">None.</p>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Case</th>
                    <th>Metric</th>
                    <th>Details</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.failed_cases.map((c) => (
                    <tr key={`${c.metric}-${c.case_id}`}>
                      <td className="mono">{c.case_id}</td>
                      <td>{c.metric}</td>
                      <td className="small mono" style={{ whiteSpace: "pre-wrap" }}>
                        {JSON.stringify(c.details, null, 1).slice(0, 600)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {series.size > 0 && (
        <section>
          <h2>Trends</h2>
          <div className="grid-2" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))" }}>
            {[...series].map(([metric, s]) => (
              <MetricTrend key={metric} metric={metric} points={s.points} threshold={s.threshold} />
            ))}
          </div>
        </section>
      )}

      {runs.length > 1 && (
        <section className="card" style={{ marginTop: 16 }}>
          <h2>All runs</h2>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {[...runs].reverse().map((r) => (
              <li key={r.id}>
                <a href={`/evals?run=${encodeURIComponent(r.id)}`}>
                  {r.suite} · {r.started_at.slice(0, 16).replace("T", " ")}
                </a>{" "}
                <Status passed={r.passed} />
              </li>
            ))}
          </ul>
        </section>
      )}
    </>
  );
}
