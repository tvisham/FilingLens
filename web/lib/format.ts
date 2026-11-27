// Display formatting. Values arrive as exact decimal strings from the API.

export function usd(value: string | number, digits = 2): string {
  const n = typeof value === "string" ? Number(value) : value;
  const abs = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  if (abs >= 1e12) return `${sign}$${(abs / 1e12).toFixed(digits)}T`;
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(digits)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(digits)}M`;
  return `${sign}$${abs.toLocaleString("en-US")}`;
}

export function usdExact(value: string): string {
  const [whole = "0", frac] = value.replace("-", "").split(".");
  const grouped = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return `${value.startsWith("-") ? "-" : ""}$${grouped}${frac ? `.${frac}` : ""}`;
}

export function quarterLabel(fy: number | null | undefined, fq: number | null | undefined): string {
  return fy && fq ? `FY${String(fy).slice(2)} Q${fq}` : "—";
}

/** EDGAR company filing list (accession prefixes are often a filing agent's CIK). */
export function edgarCompanyUrl(cik: string): string {
  return `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=${cik}&type=10-&dateb=&owner=include&count=40`;
}

/** Chunk text arrives wrapped in <filing_text> with tags escaped; show it as plain text. */
export function filingText(wrapped: string): string {
  return wrapped
    .replace(/^<filing_text>/, "")
    .replace(/<\/filing_text>$/, "")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">");
}

export const CATEGORY_LABELS: Record<string, string> = {
  macro: "Macro",
  regulatory: "Regulatory",
  cyber: "Cyber",
  supply_chain: "Supply chain",
  competition: "Competition",
  litigation: "Litigation",
  ai_technology: "AI / technology",
  liquidity: "Liquidity",
  esg_climate: "ESG / climate",
  other: "Other",
};
