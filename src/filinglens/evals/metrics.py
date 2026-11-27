"""Pure metric functions for the eval suites (no I/O, no LLM)."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "this",
        "to",
        "us",
        "we",
        "which",
        "with",
        "will",
        "would",
    ]
)


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    support: int  # gold positives


def _prf(tp: int, fp: int, fn: int) -> PRF:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return PRF(round(p, 4), round(r, 4), round(f, 4), tp + fn)


def multilabel_prf(
    gold: Sequence[set[str]], predicted: Sequence[set[str]], labels: Sequence[str]
) -> tuple[PRF, dict[str, PRF]]:
    """Micro-averaged and per-label precision/recall/F1 for multi-label classification."""
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted differ in length")
    per: dict[str, PRF] = {}
    total_tp = total_fp = total_fn = 0
    for label in labels:
        tp = sum(label in g and label in p for g, p in zip(gold, predicted, strict=True))
        fp = sum(label not in g and label in p for g, p in zip(gold, predicted, strict=True))
        fn = sum(label in g and label not in p for g, p in zip(gold, predicted, strict=True))
        per[label] = _prf(tp, fp, fn)
        total_tp, total_fp, total_fn = total_tp + tp, total_fp + fp, total_fn + fn
    return _prf(total_tp, total_fp, total_fn), per


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Agreement between two raters on one label per item, corrected for chance."""
    if len(a) != len(b):
        raise ValueError("raters differ in length")
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    labels = set(a) | set(b)
    expected = sum((a.count(lbl) / n) * (b.count(lbl) / n) for lbl in labels)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return round((observed - expected) / (1 - expected), 4)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # average rank for ties
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman rank correlation (Pearson on ranks, ties averaged). None if undefined."""
    if len(a) != len(b):
        raise ValueError("series differ in length")
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if va == 0 or vb == 0:
        return None
    return round(cov / (va * vb), 4)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 2}


def lexical_support(claim: str, sources: Sequence[str]) -> float:
    """Share of the claim's content words found in its cited sources (a cheap sanity check
    next to the LLM judge: a summary sharing almost no words with its chunks is suspect)."""
    words = content_words(claim)
    if not words:
        return 0.0
    pool = set().union(*(content_words(s) for s in sources)) if sources else set()
    return round(len(words & pool) / len(words), 4)
