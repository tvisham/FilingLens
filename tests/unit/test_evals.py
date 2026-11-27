"""Eval package: metrics, deterministic checks, golden files, the gate, and both suites end
to end with fakes for every LLM call (no network, no cost)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner
from test_risk_pipeline import K26, AnsweringMessages, routes

from filinglens.agent.records import AgentRunResult
from filinglens.config import Universe
from filinglens.evals import golden
from filinglens.evals.deterministic import check_quarter, financial_golden_match
from filinglens.evals.judge import Claim, Judge
from filinglens.evals.metrics import cohen_kappa, lexical_support, multilabel_prf, percentile, spearman
from filinglens.evals.results import CaseResult, EvalReport, judge_metric, load_thresholds, share_passed
from filinglens.evals.runner import EvalContext, run_suite
from filinglens.financials import compute_financials
from filinglens.mcp_server import tools
from filinglens.mcp_server.server import build_server
from filinglens.pipeline import load_ticker_index
from filinglens.risk.classify import Classifier
from helpers import companyfacts, fact, mock_sec_client

THRESHOLDS = load_thresholds(Path(__file__).parents[2] / "evals" / "thresholds.yaml")


# -- metrics -------------------------------------------------------------------------------


def test_multilabel_prf() -> None:
    gold = [{"macro"}, {"cyber", "regulatory"}, {"liquidity"}]
    pred = [{"macro"}, {"cyber"}, {"macro"}]
    micro, per = multilabel_prf(gold, pred, ["macro", "cyber", "regulatory", "liquidity"])
    assert (micro.precision, micro.recall) == (pytest.approx(2 / 3, abs=1e-3), 0.5)
    assert per["macro"].precision == 0.5 and per["macro"].recall == 1.0
    assert per["regulatory"].f1 == 0.0 and per["regulatory"].support == 1


def test_kappa_spearman_percentile() -> None:
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    assert cohen_kappa([], []) is None
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert spearman([1, 2], [1, 2]) is None and spearman([1, 1, 1], [1, 2, 3]) is None
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5 and percentile([], 95) is None


def test_lexical_support() -> None:
    sources = ["We depend on single-source suppliers in Asia for chips."]
    assert lexical_support("Depend on single-source suppliers for chips.", sources) == 1.0
    assert lexical_support("Depends on single-source suppliers for chips.", sources) == 0.8  # no stemming
    assert lexical_support("Faces cryptocurrency regulation in Europe.", sources) == 0.0


# -- deterministic -----------------------------------------------------------------------


def _facts(q4_shift: int = 0) -> dict[str, Any]:
    rows = [
        fact("2025-01-01", "2025-03-31", 100, "q1", fy=2025, fp="Q1", filed="2025-05-01"),
        fact("2025-04-01", "2025-06-30", 110, "q2", fy=2025, fp="Q2", filed="2025-08-01"),
        fact("2025-07-01", "2025-09-30", 120, "q3", fy=2025, fp="Q3", filed="2025-11-01"),
        fact("2025-01-01", "2025-09-30", 330, "q3", fy=2025, fp="Q3", filed="2025-11-01"),
        fact(
            "2025-01-01", "2025-12-31", 460 + q4_shift, "k", fy=2025, fp="FY", form="10-K", filed="2026-02-20"
        ),
    ]
    return companyfacts({"Revenues": rows})


def test_check_quarter_reported_and_derived() -> None:
    facts = _facts()
    q3, q4 = compute_financials(facts, cik="1", name="X").revenue.latest_two
    assert check_quarter(facts, q3) == (True, "reported value matches")
    assert q4.derived and check_quarter(facts, q4)[0]
    # Without the 9M YTD fact, FY - (Q1+Q2+Q3) is used instead.
    no_ytd = json.loads(json.dumps(facts))
    usd = no_ytd["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    usd[:] = [r for r in usd if not (r["start"] == "2025-01-01" and r["end"] == "2025-09-30")]
    assert check_quarter(no_ytd, q4) == (True, "FY - (Q1+Q2+Q3) recomputed from raw facts")


def test_check_quarter_catches_a_wrong_value() -> None:
    q3 = compute_financials(_facts(), cik="1", name="X").revenue.latest_two[0]
    tampered = q3.model_copy(update={"value": q3.value + 1})
    ok, why = check_quarter(_facts(), tampered)
    assert not ok and "!=" in why


def test_only_verified_golden_rows_count(tmp_path: Path) -> None:
    client = mock_sec_client(
        tmp_path, routes() | {"https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json": _facts()}
    )
    index = load_ticker_index(client)
    base = dict(
        ticker="EXMP",
        metric="revenue",
        fiscal_year=2025,
        fiscal_quarter=3,
        period_end=date(2025, 9, 30),
        accession="q3",
    )
    rows = [
        golden.FinancialGolden(case_id="ok", value_usd="120", verified_by="t", **base),
        golden.FinancialGolden(case_id="wrong", value_usd="121", verified_by="t", **base),
        golden.FinancialGolden(case_id="unverified", value_usd="999", **base),
    ]
    results = {c.case_id: c.passed for c in financial_golden_match(client, index, rows)}
    assert results == {"ok": True, "wrong": False}


# -- golden files and the gate -------------------------------------------------------------


def test_golden_round_trip_and_upsert(tmp_path: Path) -> None:
    a = golden.CadenceGolden(case_id="b", ticker="X", as_of=date(2026, 1, 1), form="10-K", count=3)
    b = golden.CadenceGolden(case_id="a", ticker="X", as_of=date(2026, 1, 1), form="10-Q", count=9)
    golden.save([a, b], "c.jsonl", tmp_path)
    loaded = golden.load(golden.CadenceGolden, "c.jsonl", tmp_path)
    assert [r.case_id for r in loaded] == ["a", "b"]  # sorted for clean diffs
    merged = golden.upsert(loaded, [a.model_copy(update={"count": 4})])
    assert {r.case_id: r.count for r in merged} == {"a": 9, "b": 4}
    assert golden.load(golden.CadenceGolden, "missing.jsonl", tmp_path) == []
    assert golden.heading_key("Supply Chain Risk.", "") == golden.heading_key("supply chain  risk", "x")


def test_gate_ignores_skipped_metrics() -> None:
    passed = judge_metric("faithfulness", 0.95, THRESHOLDS, 10)
    skipped = judge_metric("risk_category_f1_micro", None, THRESHOLDS, 0)
    failed = judge_metric("cost_per_report_usd", 1.50, THRESHOLDS, 3)
    assert (passed.passed, skipped.passed, failed.passed) == (True, None, False)
    report = EvalReport("smoke", "", "", None, {}, [passed, skipped], [])
    assert report.passed
    report.metrics.append(failed)
    assert not report.passed
    assert share_passed([CaseResult("a", "m", 1, True), CaseResult("b", "m", None, None)]) == 1.0


def test_thresholds_file_is_valid(tmp_path: Path) -> None:
    assert THRESHOLDS["financial_xbrl_match"] == {"min": 1.0}
    bad = tmp_path / "t.yaml"
    bad.write_text("x: 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected"):
        load_thresholds(bad)


# -- suites end to end with fakes -------------------------------------------------------


class FakeJudgeMessages:
    def create(self, **kwargs: Any) -> Any:
        content = kwargs["messages"][0]["content"]
        if kwargs["system"].startswith("You check whether claims"):
            ids = [part.split('"')[0] for part in content.split('<claim id="')[1:]]
            payload: dict[str, Any] = {
                "verdicts": [{"claim_id": i, "supported": True, "reason": "ok"} for i in ids]
            }
        else:
            payload = {"score": 4, "reason": "clear"}
        usage = SimpleNamespace(
            input_tokens=300,
            output_tokens=50,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            iterations=None,
        )
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(payload))],
            stop_reason="end_turn",
            model=kwargs["model"],
            usage=usage,
        )


async def fake_agent(scenario: str, value: str) -> AgentRunResult:
    if scenario == "question":
        output = {
            "answer": "I can't give investment advice.",
            "data_available": "State Farm" not in value,
            "declined_investment_advice": "State Farm" not in value,
            "chunk_ids": [],
            "accessions": [],
            "disclaimer": "Not investment advice.",
        }
    else:
        output = {"ticker": value, "observations": ["fine"]}
    return AgentRunResult(
        scenario,
        {"v": value},
        "ok",
        output=output,
        latency_ms=1200,
        cost_usd=None,
        tool_calls=[{"name": "mcp__filinglens__lookup_company", "input": {}}],
    )


def _ctx(tmp_path: Path, golden_dir: Path) -> EvalContext:
    r = routes() | {"https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json": _facts()}
    client = mock_sec_client(tmp_path / "sec", r)
    universe = Universe.model_validate({"sectors": {"tech": ["EXMP"]}})
    return EvalContext(
        client=client,
        index=load_ticker_index(client),
        universe=universe,
        thresholds=THRESHOLDS,
        server_factory=lambda: build_server(lambda: tools.ToolContext(client=client, universe=universe)),
        smoke_tickers=("EXMP",),
        faithfulness_tickers=("EXMP",),
        faithfulness_per_ticker=2,
        golden_dir=golden_dir,
    )


async def _anyio_run(coro: Any) -> Any:
    return await coro


@pytest.mark.anyio
async def test_smoke_suite_end_to_end(tmp_path: Path) -> None:
    report = await run_suite("smoke", _ctx(tmp_path, tmp_path / "golden"))
    by = {m.metric: m for m in report.metrics}
    assert by["financial_xbrl_match"].passed and by["financial_xbrl_match"].n_cases == 2
    assert by["financial_golden_match"].passed is None  # no verified rows: skipped
    assert by["adversarial_tools"].value == 1.0 and by["adversarial_tools"].n_cases == 6
    assert by["extraction_confidence"].passed and by["extraction_confidence"].n_cases == 1
    assert report.cost_usd == 0.0


@pytest.mark.anyio
async def test_full_suite_with_fakes(tmp_path: Path) -> None:
    gdir = tmp_path / "golden"
    golden.save(
        [
            golden.RiskLabel(
                case_id="EXMP-FY2025-x",
                ticker="EXMP",
                accession=K26,
                fiscal_year=2025,
                heading_key=golden.heading_key(
                    "The Company may be harmed by changes in economy conditions.", ""
                ),
                heading="economy",
                categories=["macro"],
                labeler="t",
                labeled_at=date(2026, 11, 26),
            )
        ],
        golden.RISK_LABELS,
        gdir,
    )
    golden.save(
        [
            golden.CoherenceLabel(
                case_id=f"c{i}",
                scenario="risk_profile",
                output_sha="x",
                score=s,
                judge_score=j,
                labeler="t",
                labeled_at=date(2026, 11, 26),
            )
            for i, (s, j) in enumerate([(2, 2), (3, 4), (5, 5), (4, 3)])
        ],
        golden.COHERENCE_LABELS,
        gdir,
    )
    ctx = _ctx(tmp_path, gdir)
    api = SimpleNamespace(beta=SimpleNamespace(messages=AnsweringMessages()))
    ctx.classifier = Classifier(api, model="claude-opus-5-5")
    ctx.judge = Judge(
        SimpleNamespace(beta=SimpleNamespace(messages=FakeJudgeMessages())), model="claude-opus-5-5"
    )
    ctx.agent = fake_agent
    ctx.agent_plan = (("risk_profile", "EXMP"), ("what_changed", "EXMP"))
    report = await run_suite("full", ctx)
    by = {m.metric: m for m in report.metrics}
    assert by["risk_category_f1_micro"].value == 1.0 and by["risk_category_f1_micro"].n_cases == 1
    # 1 labelled + 2 sampled factors. The fake judge says "supported" to everything, but the
    # fake classifier's generic summary shares no words with its chunks: the lexical floor
    # catches a rubber-stamping judge.
    faith = [c for c in report.cases if c.metric == "faithfulness"]
    assert by["faithfulness"].n_cases == 3 and by["faithfulness"].value == 0.0
    assert all(c.details["judge_supported"] and c.details["lexical"] < 0.2 for c in faith)
    assert by["agent_success_rate"].value == 1.0
    assert by["coherence_mean"].value == 4.0
    assert by["judge_human_spearman"].value == pytest.approx(0.8)
    assert by["latency_p95_ms"].value == 1200
    assert by["tool_calls_per_run"].value == 1.0
    # The fake classifier obeys nothing but the taxonomy, so the injection case passes only
    # if the summary stays clean; State Farm must be reported as data not available.
    adv = {c.case_id: c.passed for c in report.cases if c.metric == "adversarial_llm"}
    assert adv["non_sec_company"] and adv["investment_advice"]
    assert report.cost_usd > 0  # classifier + judge tokens priced


def test_judge_skips_unanswered_claims() -> None:
    class Silent:
        def create(self, **_: Any) -> Any:
            raise RuntimeError("overloaded")

    judge = Judge(SimpleNamespace(beta=SimpleNamespace(messages=Silent())), model="claude-opus-5-5")
    assert judge.faithfulness([Claim("c1", "x", ("y",))]) == {}
    assert judge.coherence({"a": 1}) is None
    assert judge.usage.failures and judge.usage.requests == 2


# -- label CLI -------------------------------------------------------------------------------


def test_label_financials_cli(tmp_path: Path) -> None:
    from filinglens.cli import main

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):  # cwd-relative evals/golden, like real use
        row = dict(
            ticker="EXMP",
            metric="revenue",
            fiscal_year=2025,
            fiscal_quarter=3,
            period_end=date(2025, 9, 30),
            accession="0000000001-25-000001",
        )
        golden.save(
            [
                golden.FinancialGolden(case_id="a", value_usd="120", **row),
                golden.FinancialGolden(case_id="b", value_usd="5", **row),
                golden.FinancialGolden(case_id="c", value_usd="7", **row),
            ],
            golden.FINANCIALS,
        )
        result = runner.invoke(
            main, ["eval", "label", "financials", "--labeler", "tvisha"], input="y\n$1,234\ns\n"
        )
        assert result.exit_code == 0, result.output
        rows = {r.case_id: r for r in golden.load(golden.FinancialGolden, golden.FINANCIALS)}
    assert rows["a"].verified_by == "tvisha" and rows["a"].value_usd == "120"
    assert rows["b"].value_usd == "1234" and rows["b"].note == "corrected by tvisha"
    assert rows["c"].verified_by is None


@pytest.mark.anyio
async def test_free_suite_scores_the_local_classifier_against_risk_labels(tmp_path: Path) -> None:
    from filinglens.risk.embed import HashingEmbedder
    from filinglens.risk.local_classify import LocalClassifier

    gdir = tmp_path / "golden"
    heading = "The Company may be harmed by changes in economy conditions."
    golden.save(
        [
            golden.RiskLabel(
                case_id="EXMP-FY2025-x",
                ticker="EXMP",
                accession=K26,
                fiscal_year=2025,
                heading_key=golden.heading_key(heading, ""),
                heading=heading,
                categories=["macro"],
                labeler="t",
                labeled_at=date(2026, 11, 27),
            )
        ],
        golden.RISK_LABELS,
        gdir,
    )
    ctx = _ctx(tmp_path, gdir)
    ctx.local_classifier = LocalClassifier(HashingEmbedder())
    report = await run_suite("smoke", ctx)
    by = {m.metric: m for m in report.metrics}
    assert by["local_category_f1_micro"].n_cases == 1 and by["local_category_f1_micro"].value is not None
    assert by["local_category_f1_micro"].passed is not False  # report-only: no threshold
    assert report.cost_usd == 0.0
