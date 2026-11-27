"""Split an Item 1A section into risk factors, and risk factors into embedding chunks.

Filers mark heading levels with styling, not wording, so headings are grouped by a style
signature and the most frequent signature is taken as the risk-factor level:

  AAPL  section "Business Risks" (bold title)    factor: bold full sentence
  AMZN  section in bold                          factor: bold+italic Title Case, no period
  XOM   section in bold                          factor: bold run-in lead ("Economic conditions. ...")
  CVX   section in bold CAPS                     factor: bold+italic run-in lead, no period
  PFE   section in bold CAPS ending ":"          factor: bold CAPS topic title

Other short styled headings are kept as `section_heading` context. A factor-level heading
with no body before the next heading is treated as a section heading. Text before the
first factor (introductions, summary lists) is not a factor. With fewer than
`MIN_HEADINGS` factor headings, the section falls back to fixed windows.

Token counts are estimates (words x 4/3); bge-small truncates at 512 tokens, and chunks
stay well under that.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from filinglens.risk.html import Block, Run, dominant_style, is_caps

MIN_HEADINGS = 3
FULL_STYLED = 0.9
TITLE_MAX_WORDS = 12
SECTION_MAX_WORDS = 25
MAX_HEADING_CHARS = 700
RUN_IN_MAX_CHARS = 400
RUN_IN_MIN_WORDS_PUNCTUATED, RUN_IN_MIN_WORDS_UNPUNCTUATED = 2, 5
WINDOW_WORDS, OVERLAP_WORDS = 300, 38  # ~400 / ~50 tokens
_TERMINAL = re.compile("[.?!:;][\"'\u201d\u2019)]*$")  # closing quotes may follow


def _is_structural(text: str) -> bool:
    """Running headers and document-structure lines ("Part I", "Item 1A", "Table of Contents").

    JPMorgan repeats a bold "Part I" header on every page; treated as a heading it split real
    risk factors apart. These lines are never risk factors and are dropped from the section.
    """
    words = text.strip().rstrip(".").lower().split()
    if not words or len(words) > 6:
        return False
    if words[0] == "part" and len(words) >= 2 and set(words[1].rstrip(".:,")) <= set("ivx"):
        return True
    if words[0] == "item" and len(words) >= 2 and words[1].rstrip(".:")[:1].isdigit():
        return True
    return " ".join(words) == "table of contents"


def estimate_tokens(text: str) -> int:
    return round(len(text.split()) * 4 / 3)


@dataclass
class RiskFactor:
    ordinal: int
    heading: str | None  # None for window-fallback factors
    section_heading: str | None
    paragraphs: list[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        return "\n\n".join(self.paragraphs)

    @property
    def text(self) -> str:
        return f"{self.heading}\n\n{self.body}" if self.heading else self.body


@dataclass(frozen=True)
class Chunk:
    factor_ordinal: int
    ordinal: int  # within the section
    heading: str | None
    section_heading: str | None
    text: str
    token_count: int

    @property
    def embedding_text(self) -> str:
        """What gets embedded: the heading gives each window its context."""
        return f"{self.heading}\n\n{self.text}" if self.heading else self.text


@dataclass(frozen=True)
class Signature:
    bold: bool
    italic: bool
    caps: bool
    kind: Literal["block", "run_in"]


@dataclass(frozen=True)
class _Candidate:
    index: int
    heading: str
    rest: str  # body text that follows a run-in heading in the same block
    signature: Signature
    shape: Literal["title", "sentence"]


def _shape(text: str) -> Literal["title", "sentence"]:
    return "title" if len(text.split()) <= TITLE_MAX_WORDS and not _TERMINAL.search(text) else "sentence"


def _candidate(i: int, b: Block) -> _Candidate | None:
    if b.styled_ratio >= FULL_STYLED and len(b.text) <= MAX_HEADING_CHARS:
        bold, italic = dominant_style(b.runs)
        return _Candidate(i, b.text, "", Signature(bold, italic, is_caps(b.text), "block"), _shape(b.text))
    lead = b.leading_styled_text
    words = len(lead.split())
    long_enough = words >= (
        RUN_IN_MIN_WORDS_PUNCTUATED if _TERMINAL.search(lead) else RUN_IN_MIN_WORDS_UNPUNCTUATED
    )
    if long_enough and len(lead) <= RUN_IN_MAX_CHARS and len(lead) < len(b.text):
        lead_runs: list[Run] = []
        for r in b.runs:
            if not (r.bold or r.italic) and r.text.strip():
                break
            lead_runs.append(r)
        bold, italic = dominant_style(lead_runs)
        rest = b.text[len(lead) :].strip() if b.text.startswith(lead) else b.text
        rest = rest.lstrip(".:;,- ")  # the bold run can stop just before its own period
        return _Candidate(i, lead, rest, Signature(bold, italic, is_caps(lead), "run_in"), "sentence")
    return None


def split_factors(blocks: list[Block]) -> tuple[list[RiskFactor], str]:
    """Returns (factors, method) where method is "headings" or "windows"."""
    blocks = [b for b in blocks if not _is_structural(b.text)]
    candidates = [c for i, b in enumerate(blocks) if (c := _candidate(i, b)) is not None]
    counts = Counter(c.signature for c in candidates)
    factor_level = max(counts, key=lambda s: (counts[s], s.kind == "block"), default=None)
    if factor_level is None or counts[factor_level] < MIN_HEADINGS:
        return _window_factors(blocks), "windows"

    # Same style, two levels (AAPL: bold "Business Risks" titles over bold sentences): when
    # sentences clearly dominate, titles in that style are section headings.
    level = [c for c in candidates if c.signature == factor_level]
    titles = sum(c.shape == "title" for c in level)
    sentences = len(level) - titles
    split_titles = sentences >= MIN_HEADINGS and titles * 2 <= sentences

    def is_factor(c: _Candidate) -> bool:
        return c.signature == factor_level and not (split_titles and c.shape == "title")

    by_index = {c.index: c for c in candidates}
    factors: list[RiskFactor] = []
    section: str | None = None
    current: RiskFactor | None = None
    for i, b in enumerate(blocks):
        cand = by_index.get(i)
        if cand is not None and is_factor(cand):
            if current is not None and not current.paragraphs:
                factors.pop()  # heading directly followed by another heading: it titles what follows
                section = current.heading
            current = RiskFactor(len(factors) + 1, cand.heading, section, [cand.rest] if cand.rest else [])
            factors.append(current)
        elif (
            cand is not None
            and cand.signature.kind == "block"
            and len(cand.heading.split()) <= SECTION_MAX_WORDS
        ):
            if current is not None and not current.paragraphs:
                factors.pop()
            section, current = cand.heading, None
        elif current is not None:
            current.paragraphs.append(b.text)

    kept = [f for f in factors if f.paragraphs]
    if len(kept) < MIN_HEADINGS:
        return _window_factors(blocks), "windows"
    for n, f in enumerate(kept, 1):
        f.ordinal = n
    return kept, "headings"


def _windows(words: list[str]) -> list[str]:
    if len(words) <= WINDOW_WORDS + OVERLAP_WORDS:
        return [" ".join(words)] if words else []
    step = WINDOW_WORDS - OVERLAP_WORDS
    return [" ".join(words[i : i + WINDOW_WORDS]) for i in range(0, len(words) - OVERLAP_WORDS, step)]


def _window_factors(blocks: list[Block]) -> list[RiskFactor]:
    words = " ".join(b.text for b in blocks).split()
    return [RiskFactor(i, None, None, [w]) for i, w in enumerate(_windows(words), 1)]


def chunk_factors(factors: list[RiskFactor]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for f in factors:
        for text in _windows(f.body.split()) or [f.heading or ""]:
            chunks.append(
                Chunk(
                    factor_ordinal=f.ordinal,
                    ordinal=len(chunks) + 1,
                    heading=f.heading,
                    section_heading=f.section_heading,
                    text=text,
                    token_count=estimate_tokens(text),
                )
            )
    return chunks
