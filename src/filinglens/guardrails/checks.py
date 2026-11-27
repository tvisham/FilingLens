"""Output guardrails for agent runs (Phase 5). Pure functions; no I/O.

numbers   Every structured financial figure (`metric` + `period_end` + `value_usd`) must equal
          the deterministic XBRL value exactly. Every dollar amount written in free text
          ("$114.48 billion") must match a known figure at its stated precision, or appear
          verbatim in a tool result from the same run (e.g. quoted from a filing).
policy    No investment recommendations: buy/sell/hold advice, ratings, price targets,
          valuation calls. Negated phrasing ("I can't recommend buying") is allowed.
disclaimer Every output's `disclaimer` must carry the standard text; it is added if missing.

Limits: percentages and counts in free text are not checked; the policy patterns are a
backstop behind the system prompt, not a classifier.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

Severity = Literal["block", "review"]

_SCALE = {
    "thousand": Decimal("1e3"),
    "million": Decimal("1e6"),
    "mn": Decimal("1e6"),
    "m": Decimal("1e6"),
    "billion": Decimal("1e9"),
    "bn": Decimal("1e9"),
    "b": Decimal("1e9"),
    "trillion": Decimal("1e12"),
    "tn": Decimal("1e12"),
    "t": Decimal("1e12"),
}
_MONEY = re.compile(
    r"\$\s?(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<frac>\d+))?\s*(?P<scale>thousand|million|billion|trillion|mn|bn|tn|[mbt])?\b",
    re.I,
)
_ADVICE = [
    re.compile(
        r"\b(recommend|suggest|advise)\w*\s+(that\s+\w+\s+)?(buying|selling|holding|to\s+(buy|sell|hold))\b",
        re.I,
    ),
    re.compile(
        r"\b(strong\s+)?(buy|sell|hold|outperform|underperform)\s+(rating|recommendation|call)\b", re.I
    ),
    re.compile(r"\bprice\s+target\b", re.I),
    re.compile(r"\bstrong\s+(buy|sell)\b", re.I),
    # "fair value" alone is ordinary accounting language in 10-Ks; only flag it with a $ figure.
    re.compile(r"\b(target|fair)\s+(price|value)\s+(is\s+|of\s+|at\s+)?(about\s+|around\s+)?\$", re.I),
    re.compile(  # a trade at a price level: "buy at $240", "sell above $300"
        r"\b(buy|sell|accumulate|add|trim)\w*\s+(it\s+|the\s+stock\s+|shares\s+)?"
        r"(at|below|above|under|near|around)\s+\$",
        re.I,
    ),
    re.compile(
        r"\b(is|looks|appears|seems)\s+(significantly\s+)?(undervalued|overvalued|cheap|expensive)\b", re.I
    ),
    re.compile(r"\b(rate|rating|we\s+are)\s+(it\s+)?(overweight|underweight)\b", re.I),
    re.compile(r"\b(investors|you)\s+should\s+(buy|sell|hold|add|trim|avoid)\b", re.I),
    re.compile(r"\bgood\s+(time|moment)\s+to\s+(buy|sell)\b", re.I),
]
_NEGATION = re.compile(
    r"\b(not|cannot|can't|won't|unable|no|never|don't|do not|decline|refuse)\b[^.]{0,40}$", re.I
)  # applied after curly apostrophes are normalised (see `_plain`)


def _plain(text: str) -> str:
    return text.replace(chr(0x2019), "'")


@dataclass(frozen=True)
class Issue:
    check: str  # numbers | policy | disclaimer | citations | schema | status | extraction
    severity: Severity
    message: str
    path: str = ""


@dataclass
class GuardrailResult:
    issues: list[Issue] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(i.severity == "block" for i in self.issues)

    @property
    def needs_review(self) -> bool:
        return bool(self.issues)

    @property
    def confidence(self) -> float:
        blocks = sum(i.severity == "block" for i in self.issues)
        reviews = sum(i.severity == "review" for i in self.issues)
        return round(max(0.0, 1.0 - 0.5 * blocks - 0.15 * reviews), 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "needs_review": self.needs_review,
            "confidence": self.confidence,
            "issues": [i.__dict__ for i in self.issues],
            "fixes": self.fixes,
        }


# -- walking outputs ------------------------------------------------------------------------


def strings(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Every string in a JSON-like value, with its path (e.g. `observations[2]`)."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from strings(v, f"{path}.{k}" if path else str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from strings(v, f"{path}[{i}]")


def figures(value: Any, path: str = "") -> Iterator[tuple[str, dict[str, Any]]]:
    """Structured financial figures: dicts carrying metric + period_end + value_usd."""
    if isinstance(value, dict):
        if {"metric", "period_end", "value_usd"} <= set(value):
            yield path, value
        for k, v in value.items():
            yield from figures(v, f"{path}.{k}" if path else str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from figures(v, f"{path}[{i}]")


# -- numbers ------------------------------------------------------------------------------


@dataclass(frozen=True)
class KnownFigure:
    ticker: str
    metric: str
    period_end: str  # ISO date
    value: Decimal
    accession: str


def parse_money(match: re.Match[str]) -> tuple[Decimal, Decimal]:
    """(amount in USD, half a unit of the last stated digit) for a `$...` match."""
    whole = match.group("num").replace(",", "")
    frac = match.group("frac") or ""
    scale = _SCALE.get((match.group("scale") or "").lower(), Decimal(1))
    amount = Decimal(f"{whole}.{frac}" if frac else whole) * scale
    tolerance = (Decimal(1) / (Decimal(10) ** len(frac))) * scale / 2
    return amount, tolerance


# Numbers not joined to letters, dots or hyphens, so dates (2026-11-25) and ids don't count.
_TOOL_NUMBER = re.compile(r"(?<![\w.\-])-?\d+(?:\.\d+)?(?![\w.\-])")


def _numbers_in(text: str, limit: int = 20000) -> list[Decimal]:
    """Plain numbers in tool output (prices, share counts), as Decimals, capped for speed."""
    out: list[Decimal] = []
    for m in _TOOL_NUMBER.finditer(text):
        try:
            out.append(Decimal(m.group(0)))
        except InvalidOperation:
            continue
        if len(out) >= limit:
            break
    return out


def check_numbers(output: dict[str, Any], known: list[KnownFigure], tool_text: str) -> list[Issue]:
    issues: list[Issue] = []
    for path, fig in figures(output):
        try:
            value = Decimal(str(fig["value_usd"]).replace(",", ""))
        except InvalidOperation:
            issues.append(Issue("numbers", "block", f"value_usd {fig['value_usd']!r} is not a number", path))
            continue
        same_period = [
            k for k in known if k.metric == fig["metric"] and k.period_end == str(fig["period_end"])
        ]
        if not same_period:
            issues.append(
                Issue(
                    "numbers", "block", f"no XBRL {fig['metric']} for period ending {fig['period_end']}", path
                )
            )
        elif not any(k.value == value for k in same_period):
            expected = ", ".join(str(k.value) for k in same_period)
            issues.append(Issue("numbers", "block", f"{fig['metric']} {value} != XBRL {expected}", path))
        elif fig.get("accession") and not any(
            k.value == value and k.accession == fig["accession"] for k in same_period
        ):
            issues.append(
                Issue("numbers", "review", f"accession {fig['accession']} differs from the XBRL source", path)
            )

    values = [k.value for k in known]
    tool_numbers = _numbers_in(tool_text)
    for path, text in strings(output):
        for m in _MONEY.finditer(text):
            amount, tol = parse_money(m)
            if any(abs(amount - v) <= tol for v in values) or any(abs(amount + v) <= tol for v in values):
                continue  # a known figure (losses may be written without the sign)
            if m.group(0).strip() in tool_text:
                continue  # quoted from a tool result (e.g. a filing chunk)
            if any(abs(amount - v) <= tol for v in tool_numbers):
                continue  # a number a tool returned this run (e.g. a closing price), at its stated precision
            issues.append(
                Issue("numbers", "review", f"amount {m.group(0).strip()!r} matches no known figure", path)
            )
    return issues


# -- policy and disclaimer -------------------------------------------------------------------


def check_policy(output: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    for path, raw in strings(output):
        text = _plain(raw)
        hits = [
            m.group(0)
            for pattern in _ADVICE
            for m in pattern.finditer(text)
            if not _NEGATION.search(text[max(0, m.start() - 60) : m.start()])
        ]
        if hits:  # one issue per piece of text, however many patterns it trips
            found = ", ".join(map(repr, hits[:3]))
            issues.append(Issue("policy", "block", f"investment advice: {found}", path))
    return issues


def enforce_disclaimer(output: dict[str, Any], disclaimer: str) -> list[str]:
    """Mutates `output`: makes sure its `disclaimer` field carries the standard text."""
    if "disclaimer" not in output:
        return []
    current = str(output.get("disclaimer") or "").strip()
    if disclaimer.lower() in current.lower():
        return []
    output["disclaimer"] = f"{current} {disclaimer}".strip()
    return ["standard disclaimer added"]


def tool_flags(payloads: list[Any]) -> list[Issue]:
    """Signals the tools themselves raised during the run (e.g. a low-confidence extraction)."""
    blob = json.dumps(payloads, default=str)
    issues = []
    if '"needs_review": true' in blob:
        issues.append(
            Issue("extraction", "review", "the run used a risk-factor extraction flagged for review")
        )
    return issues
