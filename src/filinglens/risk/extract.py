"""Locate Item 1A (Risk Factors) in a 10-K's blocks and score the extraction.

The section starts at the body "Item 1A ... Risk Factors" heading, not the table of
contents: of all candidate headings, the one followed by the most text before the next
"Item 1B" / "Item 1C" / "Item 2" wins (a TOC entry is followed almost immediately by the
next item). Some filers (e.g. banks that wrap an annual report) have no usable Item 1A
heading; a bare "Risk Factors" heading is then tried, with lower confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from filinglens.risk.html import Block, dominant_style, is_caps

MAX_HEADING_CHARS = 100
_ITEM_1A = re.compile(r"^item\s*1a\b\.?\s*[:.\-\u2013\u2014]?\s*(risk\s+factors)?\.?$", re.I)
_RISK_FACTORS = re.compile(r"^risk\s+factors\.?$", re.I)
_END = re.compile(r"^item\s*(1b|1c|2)\b", re.I)
_FALLBACK_END = re.compile(
    r"^(item\s*(1b|1c|2)\b|unresolved\s+staff\s+comments|cybersecurity|properties|legal\s+proceedings)",
    re.I,
)
MIN_WORDS, MAX_WORDS = 1500, 80_000
STYLED = 0.9
SIBLING_MAX_WORDS = 8


@dataclass(frozen=True)
class Section:
    blocks: list[Block]  # body blocks, heading excluded
    method: str  # "item_1a" | "risk_factors_heading" | "not_found"
    end_found: bool
    candidates: int
    word_count: int

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


def _is_heading(block: Block, pattern: re.Pattern[str]) -> bool:
    return len(block.text) <= MAX_HEADING_CHARS and bool(pattern.match(block.text))


def _join_split_heading(blocks: list[Block], i: int) -> bool:
    """'Item 1A.' in one block and 'Risk Factors' in the next (common in table layouts)."""
    return (
        re.fullmatch(r"item\s*1a\.?", blocks[i].text, re.I) is not None
        and i + 1 < len(blocks)
        and _RISK_FACTORS.match(blocks[i + 1].text) is not None
    )


def _style_key(b: Block) -> tuple[tuple[bool, bool], bool]:
    return dominant_style(b.runs), is_caps(b.text)


def _span(
    blocks: list[Block], start: int, end_pattern: re.Pattern[str], sibling_of: Block | None = None
) -> tuple[int, bool]:
    """End of the section starting after `start`: the first end-marker heading, or (for an
    annual-report "Risk Factors" heading) the next short heading styled like it."""
    sibling = _style_key(sibling_of) if sibling_of is not None else None
    for j in range(start + 1, len(blocks)):
        b = blocks[j]
        if len(b.text) <= MAX_HEADING_CHARS and end_pattern.match(b.text):
            return j, True
        if (
            sibling is not None
            and len(b.text.split()) <= SIBLING_MAX_WORDS
            and b.styled_ratio >= STYLED
            and _style_key(b) == sibling
        ):
            return j, True
    return len(blocks), False


def _words(blocks: list[Block]) -> int:
    return sum(len(b.text.split()) for b in blocks)


def locate_item_1a(blocks: list[Block]) -> Section:
    for method, pattern, end_pattern in (
        ("item_1a", _ITEM_1A, _END),
        ("risk_factors_heading", _RISK_FACTORS, _FALLBACK_END),
    ):
        fallback = method == "risk_factors_heading"
        starts: list[tuple[int, int]] = []  # (heading index, first body index)
        for i, b in enumerate(blocks):
            if not fallback and _join_split_heading(blocks, i):
                starts.append((i, i + 2))
            elif _is_heading(b, pattern) and (fallback or "risk" in b.text.lower()):
                if fallback and b.styled_ratio < STYLED:
                    continue  # unstyled "Risk Factors" lines are index or cross-reference entries
                starts.append((i, i + 1))
        best: Section | None = None
        for heading, body_start in starts:
            end, found = _span(blocks, body_start - 1, end_pattern, blocks[heading] if fallback else None)
            body = blocks[body_start:end]
            words = _words(body)
            if words > MAX_WORDS:
                continue  # ran past the section: wrong start
            if best is None or words > best.word_count:
                best = Section(body, method, found, len(starts), words)
        if best is not None and best.word_count >= 200:
            return best
    return Section([], "not_found", False, 0, 0)


def confidence(section: Section, factor_headings: int) -> float:
    """0..1. Below 0.7 the extraction should be reviewed by a person."""
    if section.method == "not_found":
        return 0.0
    score = 1.0 if section.method == "item_1a" else 0.7
    if not section.end_found:
        score -= 0.3
    if not MIN_WORDS <= section.word_count <= MAX_WORDS:
        score -= 0.3
    if factor_headings < 5:
        score -= 0.2
    return round(max(0.0, min(1.0, score)), 2)
