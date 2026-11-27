"""The risk pipeline end to end on a mocked SEC, plus the MCP risk tools over its output."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from filinglens.agent.records import AgentRunResult
from filinglens.config import Universe
from filinglens.mcp_server import tools
from filinglens.pipeline import load_ticker_index
from filinglens.risk.classify import Classifier
from filinglens.risk.embed import HashingEmbedder
from filinglens.risk.ingest import CLASSIFY_SCENARIO, ingest_company_risks
from filinglens.risk.store import InMemoryRiskStore
from helpers import (
    BOLD,
    TOPICS,
    apple_style_factors,
    companyfacts,
    div,
    fact,
    mock_sec_client,
    span,
    submissions,
    ten_k_html,
)

CIK = "0000000001"
ARCH = "https://www.sec.gov/Archives/edgar/data/1"
K26, K25 = "0000000001-26-000005", "0000000001-25-000005"


AI_TEXT = " ".join(["artificial intelligence features may produce errors that damage our reputation"] * 12)


def _later_year_factors() -> list[str]:
    """The next 10-K: the same factors, one with an added paragraph, plus a new AI factor."""
    blocks = apple_style_factors()
    extra = div(span("We also face new union organizing efforts in several regions where we hire. " * 4))
    blocks.append(extra)  # appended to the last factor (talent): same factor, edited wording
    blocks += [
        div(span("Artificial intelligence features may harm our reputation.", BOLD)),
        div(span(AI_TEXT)),
    ]
    return blocks


def routes(*, wrapper_2026: bool = False) -> dict[str, Any]:
    rows = [
        (K26, "10-K", "2026-02-20", "2025-12-31", "", "10-K"),
        (K25, "10-K", "2025-02-20", "2024-12-31", "", "10-K"),
    ]
    subs = submissions("Example Corp", rows, sic="1234")
    subs["filings"]["recent"]["primaryDocument"] = ["k26.htm", "k25.htm"]
    r: dict[str, Any] = {
        "https://www.sec.gov/files/company_tickers.json": {
            "0": {"cik_str": 1, "ticker": "EXMP", "title": "Example"}
        },
        f"https://data.sec.gov/submissions/CIK{CIK}.json": subs,
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json": companyfacts(
            {
                "Revenues": [
                    fact(
                        "2025-01-01",
                        "2025-12-31",
                        400,
                        K26,
                        fy=2025,
                        fp="FY",
                        form="10-K",
                        filed="2026-02-20",
                    ),
                    fact(
                        "2024-01-01",
                        "2024-12-31",
                        380,
                        K25,
                        fy=2024,
                        fp="FY",
                        form="10-K",
                        filed="2025-02-20",
                    ),
                ]
            }
        ),
        f"{ARCH}/000000000125000005/k25.htm": ten_k_html(apple_style_factors()),
        f"{ARCH}/000000000126000005/k26.htm": ten_k_html(_later_year_factors()),
    }
    if wrapper_2026:  # WFC-style: the primary document only cross-references the annual report
        r[f"{ARCH}/000000000126000005/k26.htm"] = (
            "<html><body>" + div(span("Item 1A. Risk Factors", BOLD)) + div(span("See the Annual Report."))
            + div(span("Item 1B. Unresolved Staff Comments", BOLD)) + "</body></html>"
        )  # fmt: skip
        r[f"{ARCH}/000000000126000005/index.json"] = {
            "directory": {
                "item": [
                    {"name": "k26.htm", "size": "900"},
                    {"name": "R1.htm", "size": "99999"},
                    {"name": "exmp-ex21.htm", "size": "5000"},
                    {"name": "exmp-annual-report.htm", "size": "800000"},
                ]
            }
        }
        r[f"{ARCH}/000000000126000005/exmp-annual-report.htm"] = ten_k_html(_later_year_factors())
    return r


class AnsweringMessages:
    """Answers every factor in the request, citing its first chunk."""

    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        content = kwargs["messages"][0]["content"]
        results = [
            {
                "key": key,
                "categories": ["ai_technology"]
                if "Artificial" in content.split(f'key="{key}"')[1][:200]
                else ["macro"],
                "severity": 4,
                "summary": "A risk the filing describes.",
                "chunk_ids": [int(ids.split(",")[0])],
            }
            for key, ids in re.findall(r'key="([^"]+)" chunk_ids="([\d,]+)"', content)
        ]
        usage = SimpleNamespace(input_tokens=500, output_tokens=100, cache_read_input_tokens=0,
                                cache_creation_input_tokens=0, iterations=None)  # fmt: skip
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps({"results": results}))],
            stop_reason="end_turn",
            model=kwargs["model"],
            usage=usage,
        )


@pytest.fixture()
def classifier() -> Classifier:
    return Classifier(
        SimpleNamespace(beta=SimpleNamespace(messages=AnsweringMessages())), model="claude-opus-5-5"
    )


def _ingest(tmp_path: Path, store: InMemoryRiskStore, classifier: Classifier | None, **kw: Any) -> Any:
    logged: list[AgentRunResult] = []
    client = mock_sec_client(tmp_path, routes(wrapper_2026=kw.pop("wrapper_2026", False)))

    def log_run(run: AgentRunResult) -> int:
        logged.append(run)
        return len(logged)

    result = ingest_company_risks(
        client,
        store,
        HashingEmbedder(),
        load_ticker_index(client),
        "EXMP",
        sector="tech",
        classifier=classifier,
        log_run=log_run,
        as_of=date(2026, 11, 24),
        **kw,
    )
    return result, logged


def test_ingest_extracts_classifies_and_diffs(tmp_path: Path, classifier: Classifier) -> None:
    store = InMemoryRiskStore()
    res, logged = _ingest(tmp_path, store, classifier)
    assert [(s.fiscal_year, s.factors, s.method, s.needs_review) for s in res.sections] == [
        (2025, len(TOPICS) + 1, "item_1a", False),
        (2024, len(TOPICS), "item_1a", False),
    ]  # fiscal years from XBRL `fy`, not the filing date
    assert res.classification is not None and res.classification.status == "ok"
    assert len(res.classification.results) == 2 * len(TOPICS) + 1
    assert res.diff_counts == {"new": 1, "changed": 1, "unchanged": len(TOPICS) - 1}

    (run,) = logged
    assert run.scenario == CLASSIFY_SCENARIO and run.status == "ok"
    assert run.cost_usd is not None and run.cost_usd > 0

    latest = store.sections(CIK)[0]
    factors = store.factors(latest.id)
    assert all(f.status == "classified" and f.agent_run_id == 1 for f in factors)
    assert all(set(f.cited_chunk_ids) <= set(f.chunk_ids) for f in factors)
    ai = next(f for f in factors if (f.heading or "").startswith("Artificial"))
    assert ai.categories == ["ai_technology"]


def test_reingest_reuses_classified_sections(tmp_path: Path, classifier: Classifier) -> None:
    store = InMemoryRiskStore()
    _ingest(tmp_path, store, classifier)
    calls = classifier._client.beta.messages.calls
    res, logged = _ingest(tmp_path, store, classifier)
    assert all(s.reused for s in res.sections)
    assert res.classification is None and logged == []
    assert classifier._client.beta.messages.calls == calls  # no new API calls
    res, _ = _ingest(tmp_path, store, classifier, force=True)
    assert not any(s.reused for s in res.sections)


def test_ingest_without_classifier_leaves_factors_unclassified(tmp_path: Path) -> None:
    store = InMemoryRiskStore()
    res, logged = _ingest(tmp_path, store, None)
    assert res.classification is None and logged == []
    assert {f.status for s in store.sections(CIK) for f in store.factors(s.id)} == {"unclassified"}
    assert res.diff_counts  # the diff needs only embeddings


def test_annual_report_next_to_a_wrapper_document(tmp_path: Path) -> None:
    store = InMemoryRiskStore()
    res, _ = _ingest(tmp_path, store, None, wrapper_2026=True)
    latest = store.sections(CIK)[0]
    assert latest.source_url.endswith("exmp-annual-report.htm")  # not R1.htm or an exhibit
    assert latest.needs_review and latest.confidence == 0.6
    assert res.sections[0].factors == len(TOPICS) + 1
    assert any("needs review" in w for w in res.warnings)


# -- MCP risk tools over the ingested store -------------------------------------------


@pytest.fixture()
def ctx(tmp_path: Path, classifier: Classifier) -> tools.ToolContext:
    store = InMemoryRiskStore()
    _ingest(tmp_path / "ingest", store, classifier)
    client = mock_sec_client(tmp_path / "tools", routes())
    return tools.ToolContext(
        client=client,
        universe=Universe.model_validate({"sectors": {"tech": ["EXMP"]}}),
        today=lambda: date(2026, 11, 24),
        store=store,
        embedder_factory=HashingEmbedder,
    )


def test_get_risk_factors(ctx: tools.ToolContext) -> None:
    rf = tools.get_risk_factors(ctx, "EXMP")
    assert rf.status == "available" and rf.fiscal_year == 2025 and rf.accession == K26
    assert len(rf.factors) == len(TOPICS) + 1
    assert all(f.cited_chunk_ids for f in rf.factors)
    assert tools.get_risk_factors(ctx, "EXMP", fiscal_year=2024).fiscal_year == 2024
    missing = tools.get_risk_factors(ctx, "EXMP", fiscal_year=2019)
    assert missing.status == "not_available" and "2025" in missing.message


def test_get_risk_changes(ctx: tools.ToolContext) -> None:
    ch = tools.get_risk_changes(ctx, "EXMP")
    assert (ch.from_year, ch.to_year) == (2024, 2025)
    assert ch.counts["unchanged"] == len(TOPICS) - 1
    new = [c for c in ch.changes if c.status == "new"]
    assert len(new) == 1 and new[0].latest is not None and new[0].prior is None
    assert (new[0].latest.heading or "").startswith("Artificial")


def test_search_and_read_chunks(ctx: tools.ToolContext) -> None:
    res = tools.search_risk_factors(ctx, "artificial intelligence reputation", sector="tech")
    assert res.hits and res.hits[0].fiscal_year == 2025  # latest section only
    assert (res.hits[0].heading or "").startswith("Artificial")
    got = tools.get_chunks(ctx, [res.hits[0].chunk_id, 999999])
    assert got.chunks[0].ticker == "EXMP" and got.chunks[0].source_url.endswith("k26.htm")
    assert got.missing_ids == [999999]
    with pytest.raises(tools.ToolInputError, match="unknown sector"):
        tools.search_risk_factors(ctx, "x", sector="nope")


def test_compare_sector_risks(ctx: tools.ToolContext) -> None:
    sector = tools.compare_sector_risks(ctx, "TECH", top=2)
    (company,) = sector.companies
    assert company.status == "available" and company.classified_factors == len(TOPICS) + 1
    assert company.category_counts == {"macro": len(TOPICS), "ai_technology": 1}
    assert company.mean_severity == 4.0 and len(company.top_risks) == 2
    assert sector.category_totals["macro"] == len(TOPICS)


def test_risk_tools_without_store_or_ingest(tmp_path: Path) -> None:
    client = mock_sec_client(tmp_path, routes())
    no_store = tools.ToolContext(client=client)
    with pytest.raises(tools.ToolInputError, match="not configured"):
        tools.get_risk_factors(no_store, "EXMP")
    empty = tools.ToolContext(client=client, store=InMemoryRiskStore())
    rf = tools.get_risk_factors(empty, "EXMP")
    assert rf.status == "not_available" and "risk ingest EXMP" in rf.message
    assert tools.get_risk_changes(empty, "EXMP").status == "not_available"
