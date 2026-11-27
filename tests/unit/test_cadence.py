from __future__ import annotations

from datetime import date

import pytest

from filinglens.cadence import (
    compute_cadence,
    merge_filings,
    pages_overlapping,
    parse_filing_block,
    years_before,
)
from helpers import load_fixture

AS_OF = date(2026, 11, 20)


def _all_filings() -> list:  # type: ignore[type-arg]
    subs = load_fixture("submissions_example.json")
    page = load_fixture("submissions_example_page1.json")
    return merge_filings([parse_filing_block(subs["filings"]["recent"]), parse_filing_block(page)])


def _report():  # type: ignore[no-untyped-def]
    return compute_cadence(
        cik="0000000001",
        name="Example Corp",
        fiscal_year_end="1231",
        filings=_all_filings(),
        as_of=AS_OF,
    )


def test_parse_block_handles_blank_report_dates() -> None:
    subs = load_fixture("submissions_example.json")
    filings = parse_filing_block(subs["filings"]["recent"])
    assert len(filings) == 15
    proxy = next(f for f in filings if f.form == "DEF 14A")
    assert proxy.report_date is None


def test_parse_block_rejects_ragged_columns() -> None:
    with pytest.raises(ValueError, match="length"):
        parse_filing_block(
            {"accessionNumber": ["a", "b"], "form": ["10-K"], "filingDate": ["2025-01-01"] * 2}
        )


def test_pages_overlapping_only_returns_pages_in_window() -> None:
    subs = load_fixture("submissions_example.json")
    assert pages_overlapping(subs, date(2023, 11, 20)) == ["CIK0000000001-submissions-001.json"]


def test_merge_dedupes_by_accession() -> None:
    subs = load_fixture("submissions_example.json")
    block = parse_filing_block(subs["filings"]["recent"])
    merged = merge_filings([block, block])
    assert len(merged) == len(block)
    assert merged == sorted(merged, key=lambda f: (f.filing_date, f.accession))


def test_years_before_leap_day() -> None:
    assert years_before(date(2028, 2, 29), 3) == date(2025, 2, 28)


def test_window_excludes_older_filings() -> None:
    report = _report()
    assert report.window_start == date(2023, 11, 20)
    eight_k = next(f for f in report.forms if f.form == "8-K")
    # 2023-01-05 8-K is outside the window; the page-1 2025-01-30 one is inside.
    assert date(2023, 1, 5) not in eight_k.dates
    assert date(2025, 1, 30) in eight_k.dates
    assert eight_k.count == 5


def test_counts_amendments_separately() -> None:
    ten_k = next(f for f in _report().forms if f.form == "10-K")
    assert ten_k.count == 3  # FY2023, FY2024, FY2025
    assert ten_k.amendments == 1


def test_def14a_counted_without_lag() -> None:
    proxy = next(f for f in _report().forms if f.form == "DEF 14A")
    assert proxy.count == 2
    assert proxy.median_days_after_period_end is None


def test_median_filing_lag() -> None:
    ten_q = next(f for f in _report().forms if f.form == "10-Q")
    assert ten_q.count == 9
    assert ten_q.median_days_after_period_end == 32.0


def test_fiscal_period_checks() -> None:
    checks = _report().fiscal_periods
    closed = [c for c in checks if not c.in_progress]
    assert [(c.period_end, c.ten_q_count, c.complete) for c in closed] == [
        (date(2024, 12, 31), 3, True),
        (date(2025, 12, 31), 3, True),
    ]
    current = checks[-1]
    assert current.in_progress and current.ten_q_count == 3 and current.complete


def test_missing_quarterly_report_is_flagged() -> None:
    filings = [f for f in _all_filings() if f.accession != "0000000001-25-000020"]  # drop Q2 2025
    report = compute_cadence(
        cik="0000000001", name="Example Corp", fiscal_year_end="1231", filings=filings, as_of=AS_OF
    )
    fy2025 = next(c for c in report.fiscal_periods if c.period_end == date(2025, 12, 31))
    assert fy2025.ten_q_count == 2 and not fy2025.complete
