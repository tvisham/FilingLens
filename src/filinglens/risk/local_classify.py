"""Free, local risk-factor classification: no LLM, no API key, no cost.

Categories: zero-shot by embedding similarity. Each taxonomy category, including "other",
is described by the concrete terms a 10-K would use (`LOCAL_PROMPTS`) and embedded as a
*query*; each factor's heading + text is embedded as a *passage* (bge-small's asymmetric
retrieval setup). The closest category is
always assigned; up to two more are added when they are almost as close.

Summary: extractive. The filing's own heading when it reads as a sentence (most 10-Ks
write each risk as a one-sentence heading), otherwise the factor's first sentence. It cites
the factor's first chunk, which contains that text.

Severity: not assigned (None). Judging how severe a risk is needs a model that reads the
text; a keyword heuristic would be noise dressed up as a score.

Every result is stored with model name `LOCAL_MODEL`, so it can never be mistaken for an
LLM classification. It is a baseline: compare it with the LLM classifier on the human
labels (`eval label risk`) before relying on it.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from decimal import Decimal

from filinglens.risk.classify import Category, Classification, ClassificationRun, FactorInput
from filinglens.risk.embed import Embedder, cosine

LOCAL_MODEL = "local-zero-shot (bge-small-en-v1.5, extractive summary)"
SECOND_CATEGORY_MARGIN = 0.03  # extra categories must be within this of the best match
MAX_SUMMARY_CHARS = 300
MIN_SUMMARY_CHARS = 12  # catches "." or "Security." left by run-in headings
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


# Written for embedding similarity (concrete nouns a 10-K would use), not as instructions.
LOCAL_PROMPTS: dict[Category, str] = {
    Category.MACRO: (
        "economic conditions, recession, inflation, interest rates, currency exchange rates, "
        "geopolitical events, wars, tariffs and trade disputes, pandemics"
    ),
    Category.REGULATORY: (
        "laws and regulations, government regulators, tax rates and tax law changes, "
        "sanctions, licenses, compliance obligations, antitrust law, data privacy laws"
    ),
    Category.CYBER: (
        "cybersecurity attacks, data breaches, hackers, ransomware, "
        "unauthorized access to systems and confidential information"
    ),
    Category.SUPPLY_CHAIN: (
        "suppliers, components, manufacturing and assembly partners, logistics, "
        "single-source parts, production capacity, raw materials and input shortages"
    ),
    Category.COMPETITION: (
        "competitors, competitive pressure on prices, losing market share, "
        "new market entrants, customer demand for new products"
    ),
    Category.LITIGATION: (
        "lawsuits, legal proceedings, government investigations, "
        "intellectual property disputes, fines and settlements"
    ),
    Category.AI_TECHNOLOGY: (
        "artificial intelligence, new technologies, software and platform development, "
        "technology failures and product defects, research and development"
    ),
    Category.LIQUIDITY: (
        "liquidity, funding, debt and credit ratings, capital requirements, "
        "counterparty and credit risk on receivables, investment portfolio values, cash flow"
    ),
    Category.ESG_CLIMATE: (
        "climate change, energy transition, environmental regulation, emissions, "
        "social and governance expectations of stakeholders"
    ),
    Category.OTHER: (
        "key employees and talent, reputation and brand, stock price volatility, "
        "accounting estimates and impairments, inventory write-downs, acquisitions"
    ),
}


def category_prompts() -> dict[Category, str]:
    return {c: f"Risk factor about {LOCAL_PROMPTS[c]}." for c in Category}


def extractive_summary(heading: str | None, text: str) -> str:
    if heading and len(heading.split()) >= 6 and heading.rstrip().endswith((".", "?", "!")):
        summary = heading.strip()
    else:
        body = text[len(heading) :].strip() if heading and text.startswith(heading) else text.strip()
        body = body.lstrip(".:;,- ")  # a run-in heading's period can be left at the start
        first = _SENTENCE_END.split(body, maxsplit=1)[0].strip() if body else ""
        if len(first) < MIN_SUMMARY_CHARS:  # a fragment: lead with the heading instead
            first = " ".join(x for x in (heading, first) if x).strip()
        summary = first
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[: MAX_SUMMARY_CHARS - 1].rsplit(" ", 1)[0] + "…"
    return summary


class LocalClassifier:
    """Same interface as `Classifier.classify`; runs entirely on this machine."""

    model = LOCAL_MODEL

    def __init__(self, embedder: Embedder, batch_size: int = 64) -> None:
        self._embedder = embedder
        self._batch_size = batch_size
        prompts = category_prompts()
        self._categories = list(prompts)
        self._prototypes = [embedder.embed_query(p) for p in prompts.values()]
        self.spent_usd = Decimal(0)

    def categorise(self, vector: Sequence[float]) -> list[Category]:
        scored = sorted(
            (
                (cosine(vector, proto), cat)
                for cat, proto in zip(self._categories, self._prototypes, strict=True)
            ),
            reverse=True,
        )
        best, best_cat = scored[0]
        extra = [c for s, c in scored[1:3] if s >= best - SECOND_CATEGORY_MARGIN and c is not Category.OTHER]
        return [best_cat] + ([] if best_cat is Category.OTHER else extra)

    def classify(self, factors: Sequence[FactorInput]) -> ClassificationRun:
        run = ClassificationRun(model=LOCAL_MODEL)
        started = time.monotonic()
        for i in range(0, len(factors), self._batch_size):
            batch = factors[i : i + self._batch_size]
            texts = [f"{f.heading}\n\n{f.text}" if f.heading else f.text for f in batch]
            run.requests += 1
            for f, vector in zip(batch, self._embedder.embed_passages(texts), strict=True):
                if not f.chunk_ids:
                    run.failures[f.key] = "factor has no chunks to cite"
                    continue
                run.results[f.key] = Classification(
                    key=f.key,
                    categories=self.categorise(vector),
                    severity=None,
                    summary=extractive_summary(f.heading, f.text),
                    chunk_ids=[f.chunk_ids[0]],
                )
        run.served_by.add(LOCAL_MODEL)
        run.latency_ms = round((time.monotonic() - started) * 1000)
        return run
