"""SEC's nightly bulk archives, for ingesting hundreds of companies without per-company calls.

SEC documents two ZIPs for this purpose (EDGAR API documentation, "Bulk data"):
  companyfacts.zip  every company's XBRL facts, one `CIK##########.json` per company
  submissions.zip   every company's filing history (`CIK##########.json` plus paged
                    `CIK##########-submissions-NNN.json`)
Each is over a gigabyte. `BulkSource` reads members on demand and answers the same calls
as `SecClient` (`submissions`, `submissions_page`, `company_facts`, `company_tickers`), so
the Phase 1 pipeline runs on it unchanged.

`public_float_ranking` orders companies by the latest `dei:EntityPublicFloat` they reported
(the market value of shares held by non-affiliates, from each 10-K's cover page): a size
measure that comes from the filings themselves, so building a ~500-company universe needs
no scraped index list.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from filinglens.edgar.client import SecClient, SecNotFoundError, format_cik

BULK_URLS = {
    "companyfacts": "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip",
    "submissions": "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip",
}


def download_bulk(
    client: SecClient, directory: Path, names: tuple[str, ...] = ("companyfacts", "submissions")
) -> dict[str, int]:
    """Download the documented bulk archives into `directory`. Returns bytes per archive."""
    out = {}
    for name in names:
        out[name] = client.download(BULK_URLS[name], directory / f"{name}.zip")
    return out


class BulkSource:
    """Reads company JSON from the bulk ZIPs; ticker lookups go to `client` (a small file)."""

    def __init__(self, directory: Path, client: SecClient | None = None) -> None:
        self._facts = zipfile.ZipFile(directory / "companyfacts.zip")
        self._subs = zipfile.ZipFile(directory / "submissions.zip")
        self._facts_names = set(self._facts.namelist())
        self._subs_names = set(self._subs.namelist())
        self._client = client

    def close(self) -> None:
        self._facts.close()
        self._subs.close()

    def __enter__(self) -> BulkSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _read(self, archive: zipfile.ZipFile, names: set[str], member: str) -> dict[str, Any]:
        if member not in names:
            raise SecNotFoundError(f"{member} not in {archive.filename}")
        data: dict[str, Any] = json.loads(archive.read(member))
        return data

    def company_tickers(self) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("BulkSource needs a SecClient for company_tickers.json")
        return self._client.company_tickers()

    def submissions(self, cik: int | str) -> dict[str, Any]:
        return self._read(self._subs, self._subs_names, f"CIK{format_cik(cik)}.json")

    def submissions_page(self, name: str) -> dict[str, Any]:
        if "/" in name or "\\" in name or not name.endswith(".json"):
            raise ValueError(f"unexpected submissions page name: {name!r}")
        return self._read(self._subs, self._subs_names, name)

    def company_facts(self, cik: int | str) -> dict[str, Any]:
        return self._read(self._facts, self._facts_names, f"CIK{format_cik(cik)}.json")

    def company_fact_ciks(self) -> Iterator[str]:
        for name in self._facts_names:
            if name.startswith("CIK") and name.endswith(".json") and name[3:13].isdigit():
                yield name[3:13]


@dataclass(frozen=True)
class FloatRank:
    cik: str
    ticker: str
    name: str
    public_float_usd: float
    as_of: date
    sic: str | None


def latest_public_float(facts: dict[str, Any], not_before: date) -> tuple[float, date] | None:
    rows = facts.get("facts", {}).get("dei", {}).get("EntityPublicFloat", {}).get("units", {}).get("USD", [])
    dated = [(date.fromisoformat(r["end"]), float(r["val"])) for r in rows if r.get("end") and r.get("val")]
    dated = [d for d in dated if d[0] >= not_before]
    if not dated:
        return None
    end, value = max(dated)
    return value, end


def public_float_ranking(
    source: BulkSource, top: int, *, as_of: date | None = None, max_age_days: int = 550
) -> list[FloatRank]:
    """The `top` companies by most recent public float, among those with a ticker on file."""
    as_of = as_of or date.today()
    tickers = source.company_tickers()
    by_cik: dict[str, tuple[str, str]] = {}
    for row in tickers.values():
        cik = format_cik(row["cik_str"])
        by_cik.setdefault(cik, (str(row["ticker"]).upper(), str(row["title"])))  # first listed share class
    ranked: list[FloatRank] = []
    for cik in source.company_fact_ciks():
        if cik not in by_cik:
            continue
        found = latest_public_float(source.company_facts(cik), as_of - timedelta(days=max_age_days))
        if found is None:
            continue
        ticker, name = by_cik[cik]
        ranked.append(FloatRank(cik, ticker, name, found[0], found[1], None))
    ranked.sort(key=lambda r: -r.public_float_usd)
    out = []
    for r in ranked[:top]:
        try:
            sic = str(source.submissions(r.cik).get("sic") or "") or None
        except SecNotFoundError:
            sic = None
        out.append(FloatRank(r.cik, r.ticker, r.name, r.public_float_usd, r.as_of, sic))
    return out


# SEC SIC code divisions, used to group a generated universe into sectors.
SIC_DIVISIONS = [
    (100, 999, "agriculture"),
    (1000, 1499, "mining"),
    (1500, 1799, "construction"),
    (2000, 3999, "manufacturing"),
    (4000, 4999, "transportation_utilities"),
    (5000, 5199, "wholesale"),
    (5200, 5999, "retail"),
    (6000, 6799, "finance"),
    (7000, 8999, "services"),
    (9100, 9999, "public_administration"),
]


def sic_sector(sic: str | None) -> str:
    if not sic or not sic.isdigit():
        return "unclassified"
    code = int(sic)
    return next((name for lo, hi, name in SIC_DIVISIONS if lo <= code <= hi), "unclassified")
