"""Silver labels: filer section headings as an automatic check on stored classifications."""

from __future__ import annotations

from datetime import date
from typing import Any

from filinglens.evals.silver import silver_categories, silver_eval
from filinglens.risk.classify import Classification
from filinglens.risk.store import CompanyRef, InMemoryRiskStore, NewFactor, NewSection


def test_heading_rules() -> None:
    assert silver_categories("Legal and Regulatory Risks") == {"regulatory", "litigation"}
    assert silver_categories("Liquidity") == {"liquidity"}
    assert silver_categories("Climate Change and Energy Transition") == {"esg_climate"}
    assert silver_categories("Risks Related to Our Common Stock") == {"other"}
    # generic or catch-all headings say nothing specific
    for generic in ("Business Risks", "Operational", "General Risks", None, ""):
        assert silver_categories(generic) is None
    assert silver_categories("Market, Legal, Financial, Regulatory, Cyber, Climate and Supply Risks") is None


def _section(fiscal_year: int, accession: str) -> NewSection:
    return NewSection(
        company=CompanyRef(cik="0000000001", ticker="EXMP", name="Example", sector="tech"),
        accession=accession,
        filer_cik="0000000001",
        primary_document="x.htm",
        form="10-K",
        filing_date=date(fiscal_year + 1, 2, 1),
        report_date=date(fiscal_year, 12, 31),
        fiscal_year=fiscal_year,
        source_url="https://example.invalid",
        method="test",
        split_method="test",
        confidence=1.0,
        needs_review=False,
        word_count=10,
        text="",
        embedding_model="hash",
    )


def test_silver_eval_scores_the_latest_year_only() -> None:
    store = InMemoryRiskStore()
    factors = [
        NewFactor(1, "We face lawsuits", "Legal and Regulatory Risks", "text"),
        NewFactor(2, "Hackers", "Legal and Regulatory Risks", "text"),
        NewFactor(3, "Funding could dry up", "Liquidity", "text"),
        NewFactor(4, "Execution", "Business Risks", "text"),  # generic: not scored
    ]
    cats: dict[int, list[Any]] = {
        1: ["litigation"],
        2: ["cyber", "regulatory"],
        3: ["liquidity"],
        4: ["other"],
    }
    _, stored = store.save_section(_section(2025, "a-2025"), factors, [])
    store.save_classifications(
        {f.id: Classification(str(f.ordinal), cats[f.ordinal], None, "s", []) for f in stored},
        {},
        model="local",
        agent_run_id=None,
    )
    _, old = store.save_section(_section(2024, "a-2024"), factors[:1], [])
    store.save_classifications(
        {old[0].id: Classification("1", ["macro"], None, "s", [])}, {}, model="local", agent_run_id=None
    )

    cases, summary = silver_eval(store, ["0000000001"])
    assert (summary.total, summary.labelled) == (4, 3)  # FY2024 ignored; "Business Risks" skipped
    assert summary.agreement_primary == 2 / 3  # factor 2's primary "cyber" isn't legal/regulatory
    assert summary.agreement_any == 1.0
    assert summary.by_heading["Legal and Regulatory Risks"] == (1, 2)
    failed = [c for c in cases if not c.passed]
    assert [c.case_id for c in failed] == ["EXMP-FY2025-2"]
    assert failed[0].details["classifier"] == ["cyber", "regulatory"]
    assert silver_eval(store, ["0000000099"])[0] == []
