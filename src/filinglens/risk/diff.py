"""Year-over-year risk-factor diff.

Factors are matched across years by a blend of embedding similarity (mean of the factor's
chunk vectors) and heading similarity, greedily from the best pair down, one-to-one.
Matched pairs are `unchanged` when their text is nearly identical and `changed` otherwise;
unmatched factors are `new` (this year only) or `removed` (last year only).
Thresholds are starting points, to be tuned against hand-labelled pairs in Phase 4.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from filinglens.risk.embed import cosine

MATCH_THRESHOLD = 0.80
# Near-identical text is the same factor even when its heading was rewritten: JPM's FY2025
# 10-K renamed most headings (heading similarity 0.2-0.4) over text at cosine 0.93-0.96, while
# Apple's genuinely new AI risk factor is at 0.86 to its nearest predecessor.
SAME_TEXT_COSINE = 0.92
UNCHANGED_TEXT_RATIO = 0.97
EMBEDDING_WEIGHT = 0.75

Status = Literal["new", "removed", "changed", "unchanged"]
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class FactorVector:
    id: int
    heading: str | None
    text: str
    vector: Sequence[float]


@dataclass(frozen=True)
class DiffRow:
    status: Status
    from_id: int | None  # prior year
    to_id: int | None  # latest year
    similarity: float | None  # match score for matched pairs
    text_similarity: float | None


def _words(text: str | None) -> list[str]:
    return _WORD.findall((text or "").lower())


def heading_similarity(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, _words(a), _words(b), autojunk=False).ratio()


def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _words(a), _words(b), autojunk=False).ratio()


def match_score(a: FactorVector, b: FactorVector) -> float:
    emb = cosine(a.vector, b.vector)
    if a.heading and b.heading:
        return EMBEDDING_WEIGHT * emb + (1 - EMBEDDING_WEIGHT) * heading_similarity(a.heading, b.heading)
    return emb


def diff_factors(
    prior: Sequence[FactorVector],
    latest: Sequence[FactorVector],
    *,
    match_threshold: float = MATCH_THRESHOLD,
    unchanged_ratio: float = UNCHANGED_TEXT_RATIO,
) -> list[DiffRow]:
    def rank(p: FactorVector, n: FactorVector) -> float:
        """Blended score, or the text cosine when the text alone shows it is the same factor
        (ranked by that cosine, so near-identical pairs are matched best-first, not by ties)."""
        blended, text = match_score(p, n), cosine(p.vector, n.vector)
        return max(blended, text) if text >= SAME_TEXT_COSINE else blended

    pairs = sorted(
        ((rank(p, n), i, j) for i, p in enumerate(prior) for j, n in enumerate(latest)),
        reverse=True,
    )
    used_prior: set[int] = set()
    used_latest: set[int] = set()
    rows: list[DiffRow] = []
    for score, i, j in pairs:
        if score < match_threshold:
            break
        if i in used_prior or j in used_latest:
            continue
        used_prior.add(i)
        used_latest.add(j)
        ratio = text_similarity(prior[i].text, latest[j].text)
        status: Status = "unchanged" if ratio >= unchanged_ratio else "changed"
        reported = match_score(prior[i], latest[j])  # the real blended score, not the 1.0 shortcut
        rows.append(DiffRow(status, prior[i].id, latest[j].id, round(reported, 4), round(ratio, 4)))
    rows += [DiffRow("removed", p.id, None, None, None) for i, p in enumerate(prior) if i not in used_prior]
    rows += [DiffRow("new", None, n.id, None, None) for j, n in enumerate(latest) if j not in used_latest]
    order = {"new": 0, "changed": 1, "removed": 2, "unchanged": 3}
    return sorted(rows, key=lambda r: (order[r.status], r.to_id or 0, r.from_id or 0))
