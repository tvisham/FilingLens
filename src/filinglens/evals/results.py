"""Eval result types and the threshold gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

THRESHOLDS_FILE = Path("evals/thresholds.yaml")


@dataclass
class CaseResult:
    case_id: str
    metric: str
    score: float | None
    passed: bool | None  # None = skipped
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    metric: str
    value: float | None  # None = skipped (no cases / no labels)
    threshold: dict[str, float]
    passed: bool | None
    n_cases: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    suite: str
    started_at: str
    finished_at: str
    git_sha: str | None
    models: dict[str, str]
    metrics: list[MetricResult]
    cases: list[CaseResult]
    cost_usd: float = 0.0

    @property
    def passed(self) -> bool:
        """The gate: every metric that ran met its threshold. Skipped metrics don't fail it."""
        return all(m.passed is not False for m in self.metrics)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["passed"] = self.passed
        return d


def load_thresholds(path: Path = THRESHOLDS_FILE) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    out: dict[str, dict[str, float]] = {}
    for metric, rule in raw.items():
        if not isinstance(rule, dict) or not set(rule) <= {"min", "max"} or not rule:
            raise ValueError(f"{path}: {metric}: expected {{min: x}} and/or {{max: y}}")
        out[str(metric)] = {k: float(v) for k, v in rule.items()}
    return out


def judge_metric(
    metric: str, value: float | None, thresholds: dict[str, dict[str, float]], n_cases: int, **details: Any
) -> MetricResult:
    rule = thresholds.get(metric, {})
    if value is None:
        return MetricResult(metric, None, rule, None, n_cases, details)
    ok = ("min" not in rule or value >= rule["min"]) and ("max" not in rule or value <= rule["max"])
    return MetricResult(metric, round(value, 4), rule, ok, n_cases, details)


def share_passed(cases: list[CaseResult]) -> float | None:
    ran = [c for c in cases if c.passed is not None]
    return sum(bool(c.passed) for c in ran) / len(ran) if ran else None
