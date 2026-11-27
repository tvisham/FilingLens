"""Record trimmed real SEC responses into tests/fixtures/recorded/ for regression tests.

Usage:  uv run python scripts/record_fixtures.py AAPL COST JPM

Keeps only the revenue / net income concepts from companyfacts and the most recent
filings from submissions, so the files stay small enough to commit. SEC EDGAR data is
public, but check the file sizes before committing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from filinglens.edgar.cache import DiskCache
from filinglens.edgar.client import SecClient
from filinglens.financials import NET_INCOME_CONCEPTS, REVENUE_CONCEPTS
from filinglens.pipeline import load_ticker_index
from filinglens.settings import get_settings

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "recorded"
KEEP = set(REVENUE_CONCEPTS) | set(NET_INCOME_CONCEPTS)


def main(tickers: list[str]) -> None:
    s = get_settings()
    OUT.mkdir(parents=True, exist_ok=True)
    with SecClient(s.sec_user_agent, DiskCache(s.cache_dir)) as client:
        index = load_ticker_index(client)
        for ticker in tickers:
            cik = index.resolve(ticker).cik
            facts = client.company_facts(cik)
            gaap = facts.get("facts", {}).get("us-gaap", {})
            facts["facts"] = {"us-gaap": {k: v for k, v in gaap.items() if k in KEEP}}
            (OUT / f"{ticker.lower()}_companyfacts.json").write_text(
                json.dumps(facts, indent=1), encoding="utf-8"
            )
            subs = client.submissions(cik)
            recent = subs["filings"]["recent"]
            subs["filings"]["recent"] = {k: v[:200] for k, v in recent.items()}
            (OUT / f"{ticker.lower()}_submissions.json").write_text(
                json.dumps(subs, indent=1), encoding="utf-8"
            )
            print(f"recorded {ticker} (CIK {cik})")


if __name__ == "__main__":
    main(sys.argv[1:] or ["AAPL", "COST", "JPM"])
