"""Silver labels: the risk categories companies assign themselves, via their section headings.

Most 10-Ks group risk factors under headings such as "Legal and Regulatory Risks",
"Liquidity" or "Climate Change and Energy Transition". Those headings were written by the
filer, and the local classifier never sees them (it reads only each factor's own heading
and text), so they are an independent, automatic reference.

A heading maps to a *set* of acceptable categories ("Legal and Regulatory" accepts
`regulatory` or `litigation`). Generic headings ("Business Risks", "Operational") match no
rule and are left out, as are catch-all headings spanning more than `MAX_GROUPS` areas.

Silver labels are weaker than human labels (`eval label risk`): a factor under "Legal and
Regulatory" can really be about cybersecurity. Treat the score as a sanity check and a
list of disagreements to look at, not as accuracy.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from filinglens.evals.results import CaseResult
from filinglens.risk.store import RiskStore

MAX_GROUPS = 3

# keyword (matched against the lower-cased heading) -> categories it signals
RULES: list[tuple[tuple[str, ...], frozenset[str]]] = [
    (("regulat", "legal", "law", "polic", "government", "compliance", "tax"), frozenset({"regulatory"})),
    (("legal", "litigation", "proceeding", "intellectual property"), frozenset({"litigation"})),
    (("cyber", "data privacy", "information security"), frozenset({"cyber"})),
    (("liquidity", "credit", "financial risk", "indebtedness", "capital markets", "funding", "capital"),
     frozenset({"liquidity"})),
    (("macroeconomic", "economic", "geopolit", "political", "interest rate", "market", "financial risk"),
     frozenset({"macro"})),
    (("climate", "environment", "esg", "energy transition", "weather", "sustainab"),
     frozenset({"esg_climate"})),
    (("supply", "manufactur"), frozenset({"supply_chain"})),
    (("competition", "competitive", "industry"), frozenset({"competition"})),
    (("artificial intelligence", "technolog", "intellectual property"), frozenset({"ai_technology"})),
    (("stock", "reputation", "personnel", "human capital", "structural", "organizational", "shareholder"),
     frozenset({"other"})),
]  # fmt: skip


def silver_categories(section_heading: str | None) -> frozenset[str] | None:
    """Categories the filer's own section heading allows, or None if it says nothing specific."""
    if not section_heading:
        return None
    text = section_heading.lower()
    matched = [cats for keywords, cats in RULES if any(k in text for k in keywords)]
    if not matched or len(matched) > MAX_GROUPS:
        return None
    return frozenset().union(*matched)


@dataclass(frozen=True)
class SilverSummary:
    labelled: int
    total: int
    agreement_primary: float | None  # classifier's first category is allowed by the heading
    agreement_any: float | None  # any of the classifier's categories is allowed
    by_heading: dict[str, tuple[int, int]]  # heading -> (agree, labelled)


def silver_eval(store: RiskStore, ciks: Sequence[str]) -> tuple[list[CaseResult], SilverSummary]:
    """Score stored classifications (latest 10-K per company) against the filers' headings."""
    cases: list[CaseResult] = []
    total = 0
    agree_primary = agree_any = 0
    per_heading: Counter[str] = Counter()
    per_heading_ok: Counter[str] = Counter()
    for cik in ciks:
        sections = store.sections(cik)
        if not sections:
            continue
        latest = sections[0]
        for f in store.factors(latest.id):
            if f.status != "classified" or not f.categories:
                continue
            total += 1
            allowed = silver_categories(f.section_heading)
            if allowed is None:
                continue
            primary_ok = f.categories[0] in allowed
            any_ok = bool(allowed.intersection(f.categories))
            agree_primary += primary_ok
            agree_any += any_ok
            key = f.section_heading or ""
            per_heading[key] += 1
            per_heading_ok[key] += primary_ok
            cases.append(
                CaseResult(
                    case_id=f"{latest.ticker}-FY{latest.fiscal_year}-{f.ordinal}",
                    metric="silver_category",
                    score=1.0 if primary_ok else 0.0,
                    passed=primary_ok,
                    details={
                        "section_heading": f.section_heading,
                        "allowed": sorted(allowed),
                        "classifier": f.categories,
                        "heading": (f.heading or "")[:200],
                        "model": f.model,
                    },
                )
            )
    n = len(cases)
    summary = SilverSummary(
        labelled=n,
        total=total,
        agreement_primary=agree_primary / n if n else None,
        agreement_any=agree_any / n if n else None,
        by_heading={h: (per_heading_ok[h], per_heading[h]) for h in per_heading},
    )
    return cases, summary
