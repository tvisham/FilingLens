"""Builders for test data.

All data here is SYNTHETIC: fictional companies whose JSON follows the exact shape of
SEC's submissions and companyfacts endpoints. Recorded real responses can be added under
tests/fixtures/recorded/ with scripts/record_fixtures.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> Any:
    with (FIXTURES / name).open(encoding="utf-8") as fh:
        return json.load(fh)


def fact(
    start: str,
    end: str,
    val: float,
    accn: str,
    *,
    fy: int | None,
    fp: str | None,
    form: str = "10-Q",
    filed: str,
) -> dict[str, Any]:
    """One companyfacts duration fact, shaped like SEC's JSON."""
    return {
        "start": start,
        "end": end,
        "val": val,
        "accn": accn,
        "fy": fy,
        "fp": fp,
        "form": form,
        "filed": filed,
    }


def companyfacts(concepts: dict[str, list[dict[str, Any]]], *, cik: int = 1) -> dict[str, Any]:
    return {
        "cik": cik,
        "entityName": "Example Corp",
        "facts": {
            "us-gaap": {
                concept: {"label": concept, "description": "", "units": {"USD": rows}}
                for concept, rows in concepts.items()
            }
        },
    }


UA = "FilingLens Test test@example.edu"


def submissions(
    name: str,
    rows: list[tuple[str, str, str, str, str, str]],
    *,
    fiscal_year_end: str = "1231",
    **extra: Any,
) -> dict[str, Any]:
    """Columnar submissions JSON from (accession, form, filingDate, reportDate, items, description)."""
    return {
        "name": name,
        "fiscalYearEnd": fiscal_year_end,
        **extra,
        "filings": {
            "recent": {
                "accessionNumber": [r[0] for r in rows],
                "form": [r[1] for r in rows],
                "filingDate": [r[2] for r in rows],
                "reportDate": [r[3] for r in rows],
                "items": [r[4] for r in rows],
                "primaryDocDescription": [r[5] for r in rows],
                "primaryDocument": [f"doc{i}.htm" for i in range(len(rows))],
            },
            "files": [],
        },
    }


def mock_sec_client(tmp_path: Path, routes: dict[str, Any]) -> Any:
    """A SecClient whose HTTP goes to `routes` (URL -> JSON); anything else is a 404."""
    import httpx

    from filinglens.edgar.cache import DiskCache
    from filinglens.edgar.client import SecClient
    from filinglens.edgar.retry import RetryPolicy

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        return httpx.Response(200, json=routes[url]) if url in routes else httpx.Response(404)

    return SecClient(
        UA,
        DiskCache(tmp_path),
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
        retry=RetryPolicy(max_attempts=2, base_delay=0.0),
    )


# -- synthetic 10-K HTML (mirrors real EDGAR layouts) -----------------------------------

BOLD = "font-weight:700"
ITALIC = "font-style:italic"
NORMAL = "font-weight:400"


def span(text: str, style: str = NORMAL) -> str:
    return f'<span style="{style}">{text}</span>'


def div(*spans: str) -> str:
    return "<div>" + "".join(spans) + "</div>"


def body_paragraph(topic: str, words: int = 120) -> str:
    filler = " ".join(f"{topic}{i % 7}" for i in range(words))
    return div(span(f"Risks about {topic}: {filler}."))


def ten_k_html(
    factor_blocks: list[str], *, toc: bool = True, footer: str = "Example Corp | 2026 Form 10-K"
) -> str:
    """A 10-K with a table of contents, a hidden ix:header, Item 1A, Item 1B and page footers."""
    parts = ['<html><body><div style="display:none"><ix:header>hidden xbrl 10-K Item 1A</ix:header></div>']
    if toc:
        parts.append(
            "<table><tr><td>Item 1A.</td><td>Risk Factors</td><td>12</td></tr>"
            "<tr><td>Item 1B.</td><td>Unresolved Staff Comments</td><td>30</td></tr>"
            "<tr><td>Item 7A.</td><td>Market Risk</td><td>40</td></tr></table>"
        )
    parts.append(div(span("Item 1. Business", BOLD)) + body_paragraph("business", 300))
    parts.append(div(span("Item 1A.&#160;&#160;Risk Factors", BOLD)))
    parts.append(div(span("The following risks could affect our business. Read them carefully.")))
    for i, block in enumerate(factor_blocks):
        parts.append(block)
        if i % 3 == 2:
            parts.append(div(span(f"{footer} | {10 + i}")))
    parts.append(div(span("Item 1B. Unresolved Staff Comments", BOLD)) + div(span("None.")))
    parts.append(div(span("Item 7A. Market Risk", BOLD)) + body_paragraph("market", 200))
    for n in range(4):
        parts.append(div(span(f"{footer} | {50 + n}")))
    parts.append("</body></html>")
    return "".join(parts)


TOPICS = ["economy", "competition", "suppliers", "cyber", "regulation", "litigation", "talent"]


def apple_style_factors() -> list[str]:
    """Bold short section titles over bold full-sentence factor headings."""
    out = [div(span("Macroeconomic and Industry Risks", BOLD))]
    for i, topic in enumerate(TOPICS):
        if i == 3:
            out.append(div(span("Business Risks", BOLD)))
        out.append(div(span(f"The Company may be harmed by changes in {topic} conditions.", BOLD)))
        out.append(body_paragraph(topic))
        out.append(body_paragraph(topic + "x", 80))
    return out


def amazon_style_factors() -> list[str]:
    """Bold section titles; factor headings bold+italic Title Case without a period."""
    out = [div(span("Business and Industry Risks", BOLD))]
    for topic in TOPICS:
        out.append(div(span(f"We Face Risks Related to {topic.title()}", f"{BOLD};{ITALIC}")))
        out.append(body_paragraph(topic))
    return out


def run_in_factors(punctuated: bool = True) -> list[str]:
    """Bold section title, then paragraphs led by a bold run-in heading (XOM / CVX)."""
    out = [div(span("Supply and Demand", BOLD)), body_paragraph("intro", 40)]
    for topic in TOPICS:
        lead = f"{topic.title()} conditions." if punctuated else f"The company is exposed to {topic} effects"
        out.append(
            div(span(lead, BOLD if punctuated else f"{BOLD};{ITALIC}"), span(" " + " ".join([topic] * 150)))
        )
    return out
