"""10-K HTML -> blocks -> Item 1A -> risk factors -> chunks, on synthetic filings that copy
the layouts seen in real 10-Ks (AAPL, AMZN, XOM, CVX, ABBV, WFC)."""

from __future__ import annotations

import pytest

from filinglens.risk.chunk import OVERLAP_WORDS, WINDOW_WORDS, chunk_factors, estimate_tokens, split_factors
from filinglens.risk.extract import confidence, locate_item_1a
from filinglens.risk.html import html_to_blocks
from helpers import (
    BOLD,
    ITALIC,
    TOPICS,
    amazon_style_factors,
    apple_style_factors,
    body_paragraph,
    div,
    run_in_factors,
    span,
    ten_k_html,
)

# -- html ------------------------------------------------------------------------------


def test_blocks_keep_styling_and_drop_hidden_content() -> None:
    blocks = html_to_blocks(
        "<html><body>"
        '<div style="display:none">secret</div>'
        f"<div>{span('Bold lead.', BOLD)}{span(' then plain text')}</div>"
        f"<div>{span('All italic', ITALIC)}</div>"
        "<script>var x = 1;</script>"
        "</body></html>"
    )
    assert [b.text for b in blocks] == ["Bold lead. then plain text", "All italic"]
    assert blocks[0].leading_styled_text == "Bold lead."
    assert blocks[0].styled_ratio < 0.5
    assert blocks[1].styled_ratio == 1.0


def test_mixed_content_text_next_to_a_table_is_kept() -> None:
    """ABBV: the heading span sits in the same <div> as a layout table."""
    heading = span("ITEM 1A. RISK FACTORS", BOLD)
    html = f"<html><body><div>{heading}<table><tr><td>x</td></tr></table></div></body></html>"
    assert [b.text for b in html_to_blocks(html)] == ["ITEM 1A. RISK FACTORS", "x"]


def test_page_furniture_removed_but_item_headings_kept() -> None:
    footer = "Example Corp | 2026 Form 10-K"
    html = (
        "<html><body>"
        + "".join(
            div(span(t))
            for t in [
                "Item 1A.",
                "Item 7A.",
                f"{footer} | 1",
                "Item 1A.",
                "Item 7A.",
                f"{footer} | 2",
                f"{footer} | 3",
                f"{footer} | 4",
                "17",
            ]
        )
        + "</body></html>"
    )
    texts = [b.text for b in html_to_blocks(html)]
    assert texts == ["Item 1A.", "Item 7A.", "Item 1A.", "Item 7A."]  # footers and page number gone


# -- locating Item 1A ----------------------------------------------------------------------


def test_body_heading_wins_over_table_of_contents() -> None:
    blocks = html_to_blocks(ten_k_html(apple_style_factors()))
    section = locate_item_1a(blocks)
    assert section.method == "item_1a"
    assert section.end_found
    assert section.candidates == 2  # TOC entry + body heading
    assert section.blocks[0].text.startswith("The following risks")
    assert all("Unresolved" not in b.text for b in section.blocks)
    assert all("Example Corp" not in b.text for b in section.blocks)


def test_annual_report_fallback_ends_at_next_sibling_heading() -> None:
    """WFC: no Item 1A heading in the annual report; 'Risk Factors' ends at the next heading
    in the same style, not at the end of the document."""
    factors = [
        x
        for t in TOPICS
        for x in (div(span(f"Our results may suffer from {t}.", f"{BOLD};{ITALIC}")), body_paragraph(t, 250))
    ]
    html = (
        "<html><body>"
        + div(span("Risk Factors"))  # unstyled index entry: ignored
        + div(span("Overview", BOLD))
        + body_paragraph("overview", 400)
        + div(span("Risk Factors", BOLD))
        + "".join(factors)
        + div(span("Controls and Procedures", BOLD))
        + body_paragraph("controls", 3000)
        + "</body></html>"
    )
    section = locate_item_1a(html_to_blocks(html))
    assert section.method == "risk_factors_heading"
    assert section.end_found
    assert "controls" not in section.text
    factors_found, method = split_factors(section.blocks)
    assert method == "headings" and len(factors_found) == len(TOPICS)
    assert confidence(section, len(factors_found)) == pytest.approx(0.7)


def test_cross_reference_only_is_not_found() -> None:
    html = (
        "<html><body>"
        + div(span("Item 1A. Risk Factors", BOLD))
        + div(span("See page 40."))
        + div(span("Item 1B. Unresolved Staff Comments", BOLD))
        + "</body></html>"
    )
    section = locate_item_1a(html_to_blocks(html))
    assert section.method == "not_found"
    assert confidence(section, 0) == 0.0


# -- splitting into factors -------------------------------------------------------------


def _factors(factor_blocks: list[str]) -> tuple[list[str | None], list[str | None], str]:
    section = locate_item_1a(html_to_blocks(ten_k_html(factor_blocks)))
    factors, method = split_factors(section.blocks)
    return [f.heading for f in factors], [f.section_heading for f in factors], method


def test_apple_style_sentence_headings_under_title_sections() -> None:
    headings, sections, method = _factors(apple_style_factors())
    assert method == "headings"
    assert headings == [f"The Company may be harmed by changes in {t} conditions." for t in TOPICS]
    assert sections[0] == "Macroeconomic and Industry Risks"
    assert sections[-1] == "Business Risks"


def test_amazon_style_bold_italic_titles_are_factors() -> None:
    headings, sections, method = _factors(amazon_style_factors())
    assert method == "headings"
    assert headings == [f"We Face Risks Related to {t.title()}" for t in TOPICS]
    assert set(sections) == {"Business and Industry Risks"}


@pytest.mark.parametrize("punctuated", [True, False])
def test_run_in_headings(punctuated: bool) -> None:
    headings, sections, method = _factors(run_in_factors(punctuated))
    assert method == "headings"
    assert len(headings) == len(TOPICS)
    assert set(sections) == {"Supply and Demand"}
    section = locate_item_1a(html_to_blocks(ten_k_html(run_in_factors(punctuated))))
    first = split_factors(section.blocks)[0][0]
    assert first.body.startswith(TOPICS[0])  # body is the text after the run-in lead


def test_unstyled_section_falls_back_to_windows() -> None:
    plain = [body_paragraph(t, 250) for t in TOPICS]  # long enough to pass the word-count check
    section = locate_item_1a(html_to_blocks(ten_k_html(plain)))
    factors, method = split_factors(section.blocks)
    assert method == "windows"
    assert all(f.heading is None for f in factors)
    assert confidence(section, 0) == pytest.approx(0.8)  # headings missing: -0.2


# -- chunks -------------------------------------------------------------------------------


def test_long_factor_is_windowed_with_overlap_and_heading_context() -> None:
    long_factor = [
        div(span("Supply chain disruption could hurt us.", BOLD)),
        body_paragraph("supply", 700),
    ] + [x for t in TOPICS[:3] for x in (div(span(f"{t.title()} could hurt us.", BOLD)), body_paragraph(t))]
    section = locate_item_1a(html_to_blocks(ten_k_html(long_factor)))
    factors, _ = split_factors(section.blocks)
    chunks = chunk_factors(factors)
    first = [c for c in chunks if c.factor_ordinal == 1]
    assert len(first) >= 3
    assert all(len(c.text.split()) <= WINDOW_WORDS for c in first)
    a, b = first[0].text.split(), first[1].text.split()
    assert a[-OVERLAP_WORDS:] == b[:OVERLAP_WORDS]
    assert first[0].embedding_text.startswith("Supply chain disruption could hurt us.")
    assert [c.ordinal for c in chunks] == list(range(1, len(chunks) + 1))
    assert all(c.token_count == estimate_tokens(c.text) for c in chunks)


def test_running_part_headers_are_not_risk_factors() -> None:
    """JPM repeats a bold "Part I" on every page; it must not split or become a factor."""
    blocks_html = []
    for t in TOPICS:
        blocks_html += [
            div(span(f"Results may suffer from {t} pressures.", BOLD)),
            body_paragraph(t, 60),
            div(span("Part I", BOLD)),
            body_paragraph(t + "more", 60),
        ]
    section = locate_item_1a(html_to_blocks(ten_k_html(blocks_html)))
    factors, method = split_factors(section.blocks)
    assert method == "headings"
    assert [f.heading for f in factors] == [f"Results may suffer from {t} pressures." for t in TOPICS]
    assert all(len(f.paragraphs) == 2 for f in factors)  # the text after "Part I" stays with its factor
