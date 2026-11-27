"""Run agent scenarios with the Claude Agent SDK, using the FilingLens MCP server.

The agent gets no built-in tools (no shell, files or web): `tools=[]`, only the read-only
FilingLens MCP tools are allowed, and `permission_mode="dontAsk"` denies anything else.
User/project Claude Code settings are not loaded, so runs are reproducible.

After a run the output is validated against the scenario's schema, and every chunk id the
agent cites must have appeared in a tool result during the run (an invented citation
fails the run as `unsupported_citation`).
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKError,
    Message,
    ResultMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)
from pydantic import BaseModel, ValidationError

from filinglens.agent.records import AgentRunResult
from filinglens.agent.scenarios import RISK_PROFILE, SCENARIOS, SYSTEM_PROMPT, Scenario
from filinglens.mcp_server.server import SERVER_NAME, TOOL_NAMES
from filinglens.settings import Settings

ALLOWED_TOOLS: tuple[str, ...] = tuple(f"mcp__{SERVER_NAME}__{name}" for name in TOOL_NAMES)
TICKER = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")
SECTOR = re.compile(r"^[a-z][a-z_ ]{1,29}$")
MAX_QUESTION_CHARS = 500

QueryFn = Callable[..., AsyncIterator[Message]]


def build_options(
    settings: Settings, project_root: Path, output: type[BaseModel] | None = None
) -> ClaudeAgentOptions:
    env = {"ANTHROPIC_API_KEY": settings.anthropic_api_key} if settings.anthropic_api_key else {}
    schema = (output or SCENARIOS[RISK_PROFILE].output).model_json_schema()
    return ClaudeAgentOptions(
        model=settings.agent_model,
        effort=settings.agent_effort,
        system_prompt=SYSTEM_PROMPT,
        tools=[],  # no built-in tools
        allowed_tools=list(ALLOWED_TOOLS),
        permission_mode="dontAsk",
        mcp_servers={
            SERVER_NAME: {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-m", "filinglens.mcp_server"],
            }
        },
        strict_mcp_config=True,  # ignore any other MCP servers configured on this machine
        setting_sources=[],  # don't load user/project Claude Code settings
        max_turns=settings.agent_max_turns,
        max_budget_usd=settings.agent_max_budget_usd,
        output_format={"type": "json_schema", "schema": schema},
        cwd=project_root,  # the MCP server reads .env and config/ relative to this
        env=env,
    )


def _tokens(usage: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not usage:
        return None, None
    tokens_in = sum(
        int(usage.get(k) or 0)
        for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    return tokens_in, int(usage.get("output_tokens") or 0)


def _status_from_result(result: ResultMessage) -> str:
    if not result.is_error:
        return "ok"
    return {"error_max_budget_usd": "budget_exceeded", "error_max_turns": "max_turns"}.get(
        result.subtype, "error"
    )


def _chunk_ids_in(value: Any) -> Iterator[int]:
    """Every integer under a key named `chunk_id` or `*chunk_ids`, at any depth."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k == "chunk_id" and isinstance(v, int):
                yield v
            elif k.endswith("chunk_ids") and isinstance(v, list):
                yield from (i for i in v if isinstance(i, int))
            else:
                yield from _chunk_ids_in(v)
    elif isinstance(value, list):
        for v in value:
            yield from _chunk_ids_in(v)


def _tool_result_payloads(block: ToolResultBlock) -> Iterator[Any]:
    parts: list[str] = []
    if isinstance(block.content, str):
        parts.append(block.content)
    elif isinstance(block.content, list):
        parts += [str(p.get("text", "")) for p in block.content if isinstance(p, dict)]
    for text in parts:
        try:
            yield json.loads(text)
        except json.JSONDecodeError:
            continue


def _cited_ids(output: dict[str, Any]) -> set[int]:
    return set(_chunk_ids_in(output))


@dataclass
class _Attempt:
    result: ResultMessage | None = None
    seen_chunk_ids: set[int] = field(default_factory=set)
    parsed_tool_results: int = 0
    error: str | None = None


async def _drive(
    run: AgentRunResult, prompt: str, options: ClaudeAgentOptions, query_fn: QueryFn
) -> _Attempt:
    """Consume one agent stream, recording tool calls and tool results on `run`."""
    attempt = _Attempt()
    try:
        async for message in query_fn(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                run.model = message.model or run.model
                run.tool_calls += [
                    {"name": b.name, "input": b.input} for b in message.content if isinstance(b, ToolUseBlock)
                ]
            elif isinstance(message, UserMessage) and isinstance(message.content, list):
                for block in message.content:
                    if isinstance(block, ToolResultBlock):
                        for payload in _tool_result_payloads(block):
                            attempt.parsed_tool_results += 1
                            attempt.seen_chunk_ids |= set(_chunk_ids_in(payload))
                            run.tool_payloads.append(payload)
            elif isinstance(message, ResultMessage):
                attempt.result = message
    except ClaudeSDKError as exc:
        attempt.error = f"{type(exc).__name__}: {exc}"
    return attempt


def _account(run: AgentRunResult, result: ResultMessage) -> None:
    tokens_in, tokens_out = _tokens(result.usage)
    run.tokens_in = (run.tokens_in or 0) + (tokens_in or 0)
    run.tokens_out = (run.tokens_out or 0) + (tokens_out or 0)
    if result.total_cost_usd is not None:
        run.cost_usd = (run.cost_usd or Decimal(0)) + Decimal(str(result.total_cost_usd))
    run.num_turns = (run.num_turns or 0) + result.num_turns
    run.session_id = result.session_id


def _repair_prompt(error: ValidationError) -> str:
    problems = "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in error.errors()[:15]
    )
    return (
        "Your structured output failed schema validation: "
        f"{problems}. Return the corrected output only. Do not change any figures or citations "
        "except to fix these errors."
    )


async def run_scenario(
    name: str,
    value: str,
    *,
    settings: Settings,
    project_root: Path,
    query_fn: QueryFn = query,
    max_repairs: int = 1,
) -> AgentRunResult:
    """Run one scenario. Never raises for agent failures; they are reported in `status`.

    A schema-invalid output gets `max_repairs` retries in the same session (the agent sees
    the validation errors); citations are then checked against this run's tool results.
    """
    scenario: Scenario = SCENARIOS[name]
    if scenario.input_key == "question":
        value = value.strip()
        control = any(ord(c) < 32 and ord(c) not in (9, 10) for c in value)  # tab and newline allowed
        if not 3 <= len(value) <= MAX_QUESTION_CHARS or control:
            raise ValueError(f"a question must be 3-{MAX_QUESTION_CHARS} printable characters")
    else:
        pattern = TICKER if scenario.input_key == "ticker" else SECTOR
        value = value.strip().upper() if scenario.input_key == "ticker" else value.strip().lower()
        if not pattern.match(value):
            raise ValueError(f"not a {scenario.input_key}: {value!r}")
    run = AgentRunResult(scenario=name, input={scenario.input_key: value}, status="error")
    options = build_options(settings, project_root, scenario.output)
    run.model = settings.agent_model
    started = time.monotonic()
    seen_chunk_ids: set[int] = set()
    parsed_tool_results = 0

    prompt = scenario.prompt(value)
    while True:
        attempt = await _drive(run, prompt, options, query_fn)
        seen_chunk_ids |= attempt.seen_chunk_ids
        parsed_tool_results += attempt.parsed_tool_results
        result = attempt.result
        if result is None:
            run.error = attempt.error or "agent finished without a result message"
            run.status = "error"
            break
        _account(run, result)
        run.status = _status_from_result(result)
        if run.status != "ok":
            run.error = "; ".join(result.errors or []) or result.subtype
            break
        try:
            run.output = scenario.output.model_validate(result.structured_output).model_dump(mode="json")
            run.error = None
            break
        except ValidationError as exc:
            run.status = "invalid_output"
            run.error = f"output failed schema validation: {exc.error_count()} error(s)"
            run.output = result.structured_output if isinstance(result.structured_output, dict) else None
            if run.repairs >= max_repairs or not result.session_id:
                break
            run.repairs += 1
            prompt = _repair_prompt(exc)
            options = replace(options, resume=result.session_id)
    run.latency_ms = round((time.monotonic() - started) * 1000)
    if run.status != "ok":
        return run

    unexpected = sorted({c["name"] for c in run.tool_calls} - set(ALLOWED_TOOLS))
    if unexpected:
        run.status, run.error = "disallowed_tool", f"agent called tools outside the allowlist: {unexpected}"
        return run

    # Citations must come from this run's tool results. Skipped only if no tool result could
    # be read at all (then there is nothing to check against).
    assert run.output is not None
    unsupported = sorted(_cited_ids(run.output) - seen_chunk_ids) if parsed_tool_results else []
    if unsupported:
        run.status = "unsupported_citation"
        run.error = f"cited chunk ids never returned by a tool in this run: {unsupported[:20]}"
    return run


async def run_risk_profile(
    ticker: str,
    *,
    settings: Settings,
    project_root: Path,
    query_fn: QueryFn = query,
) -> AgentRunResult:
    """Scenario (a)."""
    return await run_scenario(
        RISK_PROFILE, ticker, settings=settings, project_root=project_root, query_fn=query_fn
    )
