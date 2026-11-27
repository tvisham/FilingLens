from __future__ import annotations

from datetime import date
from decimal import Decimal

from filinglens.financials import (
    NET_INCOME_CONCEPTS,
    REVENUE_CONCEPTS,
    Metric,
    PeriodType,
    classify_duration,
    compute_financials,
    dedupe,
    extract_rows,
    fiscal_quarters,
    metric_result,
)
from helpers import companyfacts, fact

# A calendar-year company. Accessions encode the filing they came from.
Q1_25 = "0000000001-25-000010"  # Q1 FY2025 10-Q
Q2_25 = "0000000001-25-000020"
Q3_25 = "0000000001-25-000030"
K_25 = "0000000001-26-000005"  # FY2025 10-K
Q1_26 = "0000000001-26-000010"
Q2_26 = "0000000001-26-000020"


def calendar_year_rows() -> list[dict[str, object]]:
    return [
        fact("2025-01-01", "2025-03-31", 100, Q1_25, fy=2025, fp="Q1", filed="2025-05-01"),
        fact("2025-04-01", "2025-06-30", 110, Q2_25, fy=2025, fp="Q2", filed="2025-08-01"),
        fact("2025-01-01", "2025-06-30", 210, Q2_25, fy=2025, fp="Q2", filed="2025-08-01"),
        fact("2025-07-01", "2025-09-30", 120, Q3_25, fy=2025, fp="Q3", filed="2025-11-01"),
        fact("2025-01-01", "2025-09-30", 330, Q3_25, fy=2025, fp="Q3", filed="2025-11-01"),
        fact("2025-01-01", "2025-12-31", 460, K_25, fy=2025, fp="FY", form="10-K", filed="2026-02-15"),
        # FY2026 Q1 10-Q, including the prior-year comparative tagged fy=2026 (the trap).
        fact("2026-01-01", "2026-03-31", 140, Q1_26, fy=2026, fp="Q1", filed="2026-05-01"),
        fact("2025-01-01", "2025-03-31", 100, Q1_26, fy=2026, fp="Q1", filed="2026-05-01"),
        fact("2026-04-01", "2026-06-30", 150, Q2_26, fy=2026, fp="Q2", filed="2026-08-01"),
    ]


def test_classify_duration_bounds() -> None:
    assert classify_duration(date(2025, 1, 1), date(2025, 3, 31)) is PeriodType.QUARTER
    assert classify_duration(date(2025, 1, 1), date(2025, 6, 30)) is PeriodType.OTHER  # half year
    assert classify_duration(date(2025, 1, 1), date(2025, 9, 30)) is PeriodType.YTD9
    assert classify_duration(date(2025, 1, 1), date(2025, 12, 31)) is PeriodType.ANNUAL
    # 53-week year
    assert classify_duration(date(2024, 9, 29), date(2025, 10, 4)) is PeriodType.ANNUAL
    # 16-week quarter
    assert classify_duration(date(2025, 5, 12), date(2025, 8, 31)) is PeriodType.QUARTER


def test_extract_ignores_instants_and_disallowed_forms() -> None:
    cf = companyfacts(
        {
            "Revenues": [
                fact("2025-01-01", "2025-03-31", 1, "a", fy=2025, fp="Q1", form="8-K", filed="2025-04-01"),
                {
                    "end": "2025-03-31",
                    "val": 5,
                    "accn": "b",
                    "fy": 2025,
                    "fp": "Q1",
                    "form": "10-Q",
                    "filed": "2025-05-01",
                },
                fact("2025-01-01", "2025-03-31", 2, "c", fy=2025, fp="Q1", filed="2025-05-01"),
            ]
        }
    )
    rows = extract_rows(cf, "Revenues")
    assert [r.accession for r in rows] == ["c"]


def test_comparative_fact_does_not_relabel_prior_year_quarter() -> None:
    cf = companyfacts({"Revenues": calendar_year_rows()})
    result = metric_result(cf, Metric.REVENUE, REVENUE_CONCEPTS)
    q1_2025 = next(q for q in result.quarters if q.period_end == date(2025, 3, 31))
    assert (q1_2025.fiscal_year, q1_2025.fiscal_quarter) == (2025, 1)


def test_q4_derived_from_fy_minus_ytd() -> None:
    cf = companyfacts({"Revenues": calendar_year_rows()})
    result = metric_result(cf, Metric.REVENUE, REVENUE_CONCEPTS)
    q4 = next(q for q in result.quarters if q.period_end == date(2025, 12, 31))
    assert q4.derived
    assert q4.derivation == "FY - 9M YTD"
    assert q4.value == Decimal(130)  # 460 - 330
    assert q4.period_start == date(2025, 10, 1)
    assert (q4.fiscal_year, q4.fiscal_quarter) == (2025, 4)
    assert q4.source_accessions == [K_25, Q3_25]


def test_latest_two_and_labels_for_current_year_without_10k() -> None:
    cf = companyfacts({"Revenues": calendar_year_rows()})
    result = metric_result(cf, Metric.REVENUE, REVENUE_CONCEPTS)
    assert [(q.fiscal_year, q.fiscal_quarter, q.value) for q in result.latest_two] == [
        (2026, 1, Decimal(140)),
        (2026, 2, Decimal(150)),
    ]
    assert result.latest_two[-1].accession == Q2_26


def test_restatement_uses_latest_filed_value_and_accession() -> None:
    rows = [
        *calendar_year_rows(),
        # FY2026 Q2 10-Q restates the FY2025 Q2 comparative.
        fact("2025-04-01", "2025-06-30", 112, Q2_26, fy=2026, fp="Q2", filed="2026-08-01"),
    ]
    periods = dedupe(extract_rows(companyfacts({"Revenues": rows}), "Revenues"))
    q2 = next(p for p in periods if p.period_end == date(2025, 6, 30) and p.period_type == "quarter")
    assert q2.value == Decimal(112)
    assert q2.accession == Q2_26
    assert (q2.label_fy, q2.label_fp) == (2025, "Q2")  # label from the original filing


def test_q4_fallback_to_sum_of_quarters_when_no_ytd() -> None:
    rows = [r for r in calendar_year_rows() if r["end"] != "2025-09-30" or r["start"] != "2025-01-01"]
    cf = companyfacts({"Revenues": rows})
    result = metric_result(cf, Metric.REVENUE, REVENUE_CONCEPTS)
    q4 = next(q for q in result.quarters if q.period_end == date(2025, 12, 31))
    assert q4.derivation == "FY - (Q1 + Q2 + Q3)"
    assert q4.value == Decimal(130)  # 460 - 100 - 110 - 120


def test_directly_reported_q4_is_not_derived() -> None:
    rows = [
        *calendar_year_rows(),
        fact("2025-10-01", "2025-12-31", 131, K_25, fy=2025, fp="FY", form="10-K", filed="2026-02-15"),
    ]
    result = metric_result(companyfacts({"Revenues": rows}), Metric.REVENUE, REVENUE_CONCEPTS)
    q4 = next(q for q in result.quarters if q.period_end == date(2025, 12, 31))
    assert not q4.derived
    assert q4.value == Decimal(131)
    assert q4.fiscal_quarter == 4


def test_stale_concept_is_not_chosen() -> None:
    stale = [fact("2016-01-01", "2016-03-31", 9, "old", fy=2016, fp="Q1", filed="2016-05-01")]
    cf = companyfacts({"SalesRevenueNet": stale, "Revenues": calendar_year_rows()})
    result = metric_result(cf, Metric.REVENUE, REVENUE_CONCEPTS)
    assert result.concept == "Revenues"
    assert result.concepts_considered == ["Revenues", "SalesRevenueNet"]


def test_fallback_order_prefers_earlier_concept_covering_latest_quarter() -> None:
    rows = calendar_year_rows()
    cf = companyfacts(
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": rows,
            "Revenues": [r for r in rows if r["end"] < "2026-01-01"],  # stopped using it
        }
    )
    result = metric_result(cf, Metric.REVENUE, REVENUE_CONCEPTS)
    assert result.concept == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_bank_net_revenue_preferred() -> None:
    rows = calendar_year_rows()
    gross = [{**r, "val": float(r["val"]) * 2} for r in rows]  # type: ignore[arg-type]
    cf = companyfacts({"Revenues": gross, "RevenuesNetOfInterestExpense": rows})
    result = metric_result(cf, Metric.REVENUE, REVENUE_CONCEPTS)
    assert result.concept == "RevenuesNetOfInterestExpense"


def test_costco_style_16_week_fourth_quarter() -> None:
    # 52-week year starting Sep 2, 2024: 12 + 12 + 12 + 16 weeks.
    rows = [
        fact("2024-09-02", "2024-11-24", 60, "q1", fy=2025, fp="Q1", filed="2024-12-20"),
        fact("2024-11-25", "2025-02-16", 63, "q2", fy=2025, fp="Q2", filed="2025-03-14"),
        fact("2025-02-17", "2025-05-11", 62, "q3", fy=2025, fp="Q3", filed="2025-06-06"),
        fact("2024-09-02", "2025-05-11", 185, "q3", fy=2025, fp="Q3", filed="2025-06-06"),
        fact("2024-09-02", "2025-08-31", 270, "k", fy=2025, fp="FY", form="10-K", filed="2025-10-08"),
    ]
    quarters, warnings = fiscal_quarters(dedupe(extract_rows(companyfacts({"Revenues": rows}), "Revenues")))
    assert warnings == []
    q4 = quarters[-1]
    assert q4.derived and q4.duration_days == 111
    assert q4.value == Decimal(85)
    assert [q.fiscal_quarter for q in quarters] == [1, 2, 3, 4]
    assert {q.fiscal_year for q in quarters} == {2025}


def test_non_calendar_fiscal_year_label_comes_from_10k() -> None:
    # Fiscal year ending in late January, labelled by the company as the prior year
    # (some retailers do this): the 10-K says fy=2024 for the year ending 2025-02-02.
    rows = [
        fact("2024-02-05", "2024-05-05", 10, "q1", fy=2024, fp="Q1", filed="2024-05-30"),
        fact("2024-05-06", "2024-08-04", 11, "q2", fy=2024, fp="Q2", filed="2024-08-30"),
        fact("2024-08-05", "2024-11-03", 12, "q3", fy=2024, fp="Q3", filed="2024-11-27"),
        fact("2024-02-05", "2024-11-03", 33, "q3", fy=2024, fp="Q3", filed="2024-11-27"),
        fact("2024-02-05", "2025-02-02", 47, "k", fy=2024, fp="FY", form="10-K", filed="2025-03-20"),
    ]
    quarters, _ = fiscal_quarters(dedupe(extract_rows(companyfacts({"Revenues": rows}), "Revenues")))
    assert [(q.fiscal_year, q.fiscal_quarter) for q in quarters] == [
        (2024, 1),
        (2024, 2),
        (2024, 3),
        (2024, 4),
    ]


def test_negative_net_income_q4() -> None:
    rows = [
        fact("2025-01-01", "2025-09-30", 50, "q3", fy=2025, fp="Q3", filed="2025-11-01"),
        fact("2025-01-01", "2025-12-31", -20, "k", fy=2025, fp="FY", form="10-K", filed="2026-02-15"),
    ]
    result = metric_result(companyfacts({"NetIncomeLoss": rows}), Metric.NET_INCOME, NET_INCOME_CONCEPTS)
    assert result.latest_two[-1].value == Decimal(-70)


def test_missing_ytd_and_quarters_warns() -> None:
    rows = [fact("2025-01-01", "2025-12-31", 460, K_25, fy=2025, fp="FY", form="10-K", filed="2026-02-15")]
    quarters, warnings = fiscal_quarters(dedupe(extract_rows(companyfacts({"Revenues": rows}), "Revenues")))
    assert quarters == []
    assert warnings and "could not derive Q4" in warnings[0]


def test_q4_gap_warnings_only_for_recent_years() -> None:
    rows = [
        # 2010: annual only, no quarters -> Q4 can't be derived, but it's old: no warning.
        fact("2010-01-01", "2010-12-31", 400, "old", fy=2010, fp="FY", form="10-K", filed="2011-02-15"),
        # 2025: annual only as well -> recent gap, must warn.
        fact("2025-01-01", "2025-12-31", 460, K_25, fy=2025, fp="FY", form="10-K", filed="2026-02-15"),
        fact("2026-01-01", "2026-03-31", 120, Q1_26, fy=2026, fp="Q1", filed="2026-05-01"),
    ]
    result = metric_result(companyfacts({"Revenues": rows}), Metric.REVENUE, REVENUE_CONCEPTS)
    assert len(result.warnings) == 1
    assert "2025-12-31" in result.warnings[0]
    # Without a cutoff, fiscal_quarters still reports every gap.
    _, every = fiscal_quarters(dedupe(extract_rows(companyfacts({"Revenues": rows}), "Revenues")))
    assert len(every) == 2


def test_no_data_at_all() -> None:
    report = compute_financials(companyfacts({}), cik="0000000001", name="Example")
    assert report.revenue.concept is None
    assert report.revenue.latest_two == []
    assert report.revenue.warnings


def test_restated_quarter_with_shifted_start_is_not_a_fifth_quarter() -> None:
    """MSFT's FY2018 10-K restates Q1 FY2017 as 2016-07-02..2016-09-30 (the 10-Q said 07-01)."""
    rows = [
        fact("2016-07-01", "2016-09-30", 100, "q1-orig", fy=2017, fp="Q1", filed="2016-10-20"),
        fact(
            "2016-07-02", "2016-09-30", 101, "q1-restated", fy=2018, fp="FY", form="10-K", filed="2018-08-03"
        ),
        fact("2016-10-01", "2016-12-31", 110, "q2", fy=2017, fp="Q2", filed="2017-01-26"),
        fact("2017-01-01", "2017-03-31", 120, "q3", fy=2017, fp="Q3", filed="2017-04-27"),
        fact("2017-04-01", "2017-06-30", 130, "k", fy=2017, fp="FY", form="10-K", filed="2017-08-02"),
        fact("2016-07-01", "2017-06-30", 461, "k", fy=2017, fp="FY", form="10-K", filed="2017-08-02"),
    ]
    quarters, warnings = fiscal_quarters(dedupe(extract_rows(companyfacts({"Revenues": rows}), "Revenues")))
    assert [(q.fiscal_quarter, q.value, q.accession) for q in quarters] == [
        (1, Decimal(101), "q1-restated"),  # latest filing wins
        (2, Decimal(110), "q2"),
        (3, Decimal(120), "q3"),
        (4, Decimal(130), "k"),
    ]
    assert warnings == []
