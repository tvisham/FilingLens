import Link from "next/link";
import { CadenceTimeline } from "@/components/CadenceTimeline";
import { PriceLine } from "@/components/PriceLine";
import { QuarterBars } from "@/components/QuarterBars";
import { Categories, Citations, RiskTable } from "@/components/RiskTable";
import { ErrorBanner } from "@/components/Status";
import { api, ApiError, type CompanyDetail, type RiskChange } from "@/lib/api";
import { edgarCompanyUrl, usd } from "@/lib/format";

export const dynamic = "force-dynamic";

function ChangeGroup({ title, items }: { title: string; items: RiskChange[] }) {
  if (items.length === 0) return null;
  return (
    <div style={{ marginBottom: 16 }}>
      <h3>
        {title} ({items.length})
      </h3>
      <ul style={{ margin: 0, paddingLeft: 18 }}>
        {items.map((c, i) => {
          const f = c.latest ?? c.prior;
          return (
            <li key={`${c.status}-${f?.factor_id ?? i}`} style={{ marginBottom: 6 }}>
              {f?.heading ?? "(untitled)"} {f && <Categories values={f.categories} />}
              {c.text_similarity !== null && c.text_similarity !== undefined && (
                <span className="small muted"> · wording {Math.round(c.text_similarity * 100)}% the same</span>
              )}{" "}
              {f && <Citations ids={f.cited_chunk_ids} />}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export default async function CompanyPage({ params }: { params: Promise<{ ticker: string }> }) {
  const { ticker } = await params;
  let data: CompanyDetail;
  try {
    data = await api.company(ticker);
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) {
      return (
        <>
          <h1>{ticker.toUpperCase()}</h1>
          <div className="banner">Not found: {e.message}</div>
          <Link href="/">Back to companies</Link>
        </>
      );
    }
    return <ErrorBanner error={e} />;
  }
  const { company, filings, financials, risks, changes, prices } = data;
  const counts = Object.fromEntries(filings.counts.map((c) => [c.form, c]));
  const by = (s: string) => changes.changes.filter((c) => c.status === s);

  return (
    <>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div>
          <h1>
            {company.name} <span className="muted">({company.ticker})</span>
          </h1>
          <p className="muted small" style={{ margin: 0 }}>
            CIK {company.cik} · {company.sic_description ?? "SIC n/a"} · fiscal year ends {company.fiscal_year_end ?? "n/a"}
            {company.predecessor_ciks.length > 0 && <> · includes predecessor CIK {company.predecessor_ciks.join(", ")}</>}
          </p>
        </div>
        <a href={edgarCompanyUrl(company.cik)} target="_blank" rel="noreferrer" className="small">
          Filings on EDGAR
        </a>
      </div>

      <section className="card" style={{ marginTop: 16 }}>
        <h2>Filing cadence</h2>
        <p className="small muted">
          {filings.window_start} to {filings.window_end}:{" "}
          {["10-K", "10-Q", "8-K", "DEF 14A"]
            .map((f) => `${counts[f]?.count ?? 0} ${f}${counts[f]?.amendments ? ` (+${counts[f]?.amendments} amended)` : ""}`)
            .join(" · ")}
          .{" "}
          {filings.fiscal_years.every((y) => y.complete || y.in_progress)
            ? "Every completed fiscal year had its three 10-Qs."
            : "Some fiscal years are missing 10-Qs."}
        </p>
        <CadenceTimeline filings={filings.filings} start={filings.window_start} end={filings.window_end} />
      </section>

      <div className="grid-2">
        <section className="card">
          <QuarterBars title={`Revenue · ${financials.revenue.concept ?? "n/a"}`} quarters={financials.revenue.quarters} />
          {financials.revenue.warnings.map((w) => (
            <p key={w} className="small muted">
              {w}
            </p>
          ))}
        </section>
        <section className="card">
          <QuarterBars title={`Net income · ${financials.net_income.concept ?? "n/a"}`} quarters={financials.net_income.quarters} />
          {financials.net_income.warnings.map((w) => (
            <p key={w} className="small muted">
              {w}
            </p>
          ))}
        </section>
      </div>
      <p className="small muted">The latest two fiscal quarters are highlighted. Values are exact XBRL figures (hover for the source).</p>

      <section className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2>Share price</h2>
          {prices.status === "available" && prices.market_cap_usd ? (
            <span className="small">
              Approx. market cap <strong>{usd(prices.market_cap_usd, 1)}</strong>{" "}
              <span className="muted">
                (last close × {prices.shares_outstanding?.toLocaleString("en-US")} shares as of {prices.shares_as_of})
              </span>
            </span>
          ) : null}
        </div>
        {prices.status === "available" ? (
          <>
            <PriceLine points={prices.closes} />
            <p className="small muted" style={{ marginBottom: 0 }}>
              {prices.source}. Last trading day {prices.last_trading_day}. Shares outstanding from {prices.shares_source}.
              Market data for context only: FilingLens does not give ratings, recommendations or price targets.
            </p>
          </>
        ) : (
          <p className="muted">{prices.message}</p>
        )}
      </section>

      <section className="card">
        <h2>Risk factors {risks.fiscal_year ? `· FY${risks.fiscal_year} 10-K` : ""}</h2>
        {risks.needs_review && (
          <div className="banner">
            Low-confidence extraction ({risks.extraction_confidence}): a person should check it against the filing.
          </div>
        )}
        {risks.status === "available" ? <RiskTable factors={risks.factors} /> : <p className="muted">{risks.message}</p>}
      </section>

      <section className="card">
        <h2>
          What changed {changes.from_year ? `· FY${changes.from_year} → FY${changes.to_year}` : ""}
        </h2>
        {changes.status === "available" ? (
          <>
            <p className="small muted">
              {Object.entries(changes.counts)
                .map(([k, v]) => `${v} ${k}`)
                .join(" · ")}
            </p>
            <ChangeGroup title="New" items={by("new")} />
            <ChangeGroup title="Removed" items={by("removed")} />
            <ChangeGroup title="Changed" items={by("changed")} />
          </>
        ) : (
          <p className="muted">{changes.message}</p>
        )}
      </section>
    </>
  );
}
