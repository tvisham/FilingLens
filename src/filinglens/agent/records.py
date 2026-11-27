"""The record of one agent or LLM run, as stored in `agent_runs`. No SDK imports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any


@dataclass
class AgentRunResult:
    scenario: str
    input: dict[str, Any]
    status: str  # ok | invalid_output | disallowed_tool | budget_exceeded | max_turns | error
    output: dict[str, Any] | None = None
    error: str | None = None
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: Decimal | None = None
    latency_ms: int | None = None
    num_turns: int | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    session_id: str | None = None
    repairs: int = 0  # schema-repair retries used (Phase 5)
    guardrails: dict[str, Any] | None = None  # GuardrailResult.to_dict(), set after the run
    # Parsed tool results seen during the run; used by the guardrails, never persisted.
    tool_payloads: list[Any] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("tool_payloads")
        d["cost_usd"] = str(self.cost_usd) if self.cost_usd is not None else None
        return d
