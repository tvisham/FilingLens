import Link from "next/link";
import { ErrorBanner } from "@/components/Status";
import { api, type CompanyListItem } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function Home() {
  let companies: CompanyListItem[] = [];
  let error: unknown = null;
  try {
    companies = await api.companies();
  } catch (e) {
    error = e;
  }
  const bySector = new Map<string, string[]>();
  for (const c of companies) bySector.set(c.sector, [...(bySector.get(c.sector) ?? []), c.ticker]);

  return (
    <>
      <h1>Companies</h1>
      <p className="muted">Filing cadence, the last two fiscal quarters, and cited 10-K risk factors.</p>
      {error ? <ErrorBanner error={error} /> : null}
      <div className="grid-2">
        {[...bySector].map(([sector, tickers]) => (
          <section key={sector} className="card">
            <h2 style={{ textTransform: "capitalize" }}>{sector}</h2>
            <div className="row">
              {tickers.map((t) => (
                <Link key={t} href={`/companies/${t}`} className="chip" style={{ fontSize: "0.9rem" }}>
                  {t}
                </Link>
              ))}
            </div>
          </section>
        ))}
      </div>
    </>
  );
}
