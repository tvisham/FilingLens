"""Phase 7: bulk archives, public-float universe ranking, classification cost estimate/cap."""

from __future__ import annotations

import json
import zipfile
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_risk_pipeline import AnsweringMessages, routes

from filinglens.edgar.bulk import BulkSource, latest_public_float, public_float_ranking, sic_sector
from filinglens.pipeline import load_ticker_index, snapshot_company
from filinglens.risk.classify import Classifier, FactorInput, estimate_cost
from filinglens.risk.embed import HashingEmbedder
from filinglens.risk.ingest import ingest_company_risks
from filinglens.risk.store import InMemoryRiskStore
from helpers import companyfacts, fact, mock_sec_client, submissions


def _floats(*rows: tuple[str, float]) -> dict[str, Any]:
    return {"EntityPublicFloat": {"units": {"USD": [{"end": end, "val": val} for end, val in rows]}}}


def _write_bulk(directory: Path) -> None:
    facts = {
        1: companyfacts(
            {"Revenues": [fact("2026-07-01", "2026-09-30", 160, "a3", fy=2026, fp="Q3", filed="2026-11-01")]}
        ),
        2: companyfacts({}),
        3: companyfacts({}),
        4: companyfacts({}),
    }
    facts[1]["facts"]["dei"] = _floats(("2025-06-30", 9e11), ("2026-06-30", 1e12))  # latest wins
    facts[2]["facts"]["dei"] = _floats(("2026-03-31", 5e11))
    facts[3]["facts"]["dei"] = _floats(("2019-06-30", 9e12))  # too old to count
    facts[4]["facts"]["dei"] = _floats(("2026-06-30", 8e12))  # no ticker on file
    with zipfile.ZipFile(directory / "companyfacts.zip", "w") as z:
        for cik, body in facts.items():
            z.writestr(f"CIK{cik:010d}.json", json.dumps(body))
    with zipfile.ZipFile(directory / "submissions.zip", "w") as z:
        subs = submissions(
            "Example Corp", [("0000000001-26-000030", "10-Q", "2026-11-01", "2026-09-30", "", "")], sic="3571"
        )
        z.writestr("CIK0000000001.json", json.dumps(subs))
        z.writestr("CIK0000000002.json", json.dumps(submissions("Bank", [], sic="6021")))


@pytest.fixture()
def bulk(tmp_path: Path) -> BulkSource:
    _write_bulk(tmp_path)
    client = mock_sec_client(
        tmp_path / "cache",
        {
            "https://www.sec.gov/files/company_tickers.json": {
                "0": {"cik_str": 1, "ticker": "EXMP", "title": "Example Corp"},
                "1": {"cik_str": 2, "ticker": "BANK", "title": "Bank"},
                "2": {"cik_str": 3, "ticker": "OLD", "title": "Old Co"},
                "3": {"cik_str": 1, "ticker": "EXMP-B", "title": "Example Corp"},  # second share class
            }
        },
    )
    return BulkSource(tmp_path, client)


def test_bulk_source_feeds_the_pipeline(bulk: BulkSource) -> None:
    snap = snapshot_company(bulk, load_ticker_index(bulk), "EXMP", as_of=date(2026, 11, 24))
    assert snap.cadence.name == "Example Corp"
    assert next(f for f in snap.cadence.forms if f.form == "10-Q").count == 1
    assert snap.financials is not None and snap.financials.revenue.quarters[-1].value == Decimal(160)
    with pytest.raises(Exception, match="not in"):
        bulk.company_facts(999)
    with pytest.raises(ValueError):
        bulk.submissions_page("../../etc/passwd.json")


def test_public_float_ranking(bulk: BulkSource) -> None:
    ranked = public_float_ranking(bulk, 10, as_of=date(2026, 11, 24))
    assert [(r.ticker, r.public_float_usd, r.sic) for r in ranked] == [
        ("EXMP", 1e12, "3571"),
        ("BANK", 5e11, "6021"),
    ]
    assert [sic_sector(r.sic) for r in ranked] == ["manufacturing", "finance"]
    assert public_float_ranking(bulk, 1, as_of=date(2026, 11, 24))[0].ticker == "EXMP"
    assert latest_public_float({"facts": {}}, date(2020, 1, 1)) is None
    assert sic_sector(None) == "unclassified" and sic_sector("9999") == "public_administration"


# -- cost estimate and budget cap ----------------------------------------------------------


def _classifier() -> Classifier:
    return Classifier(
        SimpleNamespace(beta=SimpleNamespace(messages=AnsweringMessages())), model="claude-opus-5-5"
    )


def test_estimate_is_on_the_high_side() -> None:
    clf = _classifier()
    factors = [FactorInput(f"F{i}", "A heading.", "word " * 300, (i,)) for i in range(10)]
    tokens_in, tokens_out, usd = estimate_cost(clf, factors)
    run = clf.classify(factors)  # the fake reports 500 in / 100 out per request
    assert tokens_in > 300 * 10 * 4 / 3 and tokens_out == 3000
    assert usd > run.cost_usd and usd > 0


def test_budget_cap_skips_classification(tmp_path: Path) -> None:
    store = InMemoryRiskStore()
    client = mock_sec_client(tmp_path, routes())
    clf = _classifier()
    res = ingest_company_risks(
        client,
        store,
        HashingEmbedder(),
        load_ticker_index(client),
        "EXMP",
        classifier=clf,
        as_of=date(2026, 11, 24),
        max_cost_usd=Decimal("0.0001"),
    )
    assert (
        res.classification is None and res.estimated_cost_usd and res.estimated_cost_usd > Decimal("0.0001")
    )
    assert any("exceeds the remaining budget" in w for w in res.warnings)
    assert clf.spent_usd == 0

    res = ingest_company_risks(
        client,
        store,
        HashingEmbedder(),
        load_ticker_index(client),
        "EXMP",
        classifier=clf,
        as_of=date(2026, 11, 24),
        max_cost_usd=Decimal("5"),
    )
    assert res.classification is not None and res.classification.status == "ok"


def test_mcp_cli_parses_http_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    from filinglens.mcp_server import server

    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(server.mcp, "run", lambda *a, **kw: calls.append((a, kw)))
    server.main(["--http", "--port", "9999"])
    server.main([])
    assert calls == [
        (("streamable-http",), {"host": "127.0.0.1", "port": 9999, "stateless_http": True}),
        (("stdio",), {}),
    ]
