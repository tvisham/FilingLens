"""The MCP server end to end, in process, through the official SDK client."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest
from mcp import Client
from test_mcp_tools import TODAY, routes

from filinglens.config import Universe
from filinglens.mcp_server import tools
from filinglens.mcp_server.server import TOOL_NAMES, build_server
from helpers import mock_sec_client

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture()
def server(tmp_path: Path) -> Any:
    def factory() -> tools.ToolContext:
        return tools.ToolContext(
            client=mock_sec_client(tmp_path, routes()),
            universe=Universe.model_validate({"sectors": {"x": ["EXMP"]}}),
            today=lambda: TODAY,
        )

    return build_server(factory)


async def test_exposes_exactly_the_read_only_tools(server: Any) -> None:
    async with Client(server) as client:
        listed = (await client.list_tools()).tools
    assert sorted(t.name for t in listed) == sorted(TOOL_NAMES)
    for t in listed:
        assert t.annotations is not None and t.annotations.read_only_hint is True, t.name
        assert t.output_schema is not None, t.name
        assert t.description, t.name


async def test_structured_financials(server: Any) -> None:
    async with Client(server) as client:
        result = await client.call_tool("get_financials", {"ticker": "EXMP", "quarters": 1})
    assert not result.is_error
    q = result.structured_content["revenue"]["quarters"]
    assert q == [
        {
            "fiscal_year": 2026,
            "fiscal_quarter": 3,
            "period_start": "2026-07-01",
            "period_end": "2026-09-30",
            "value_usd": "160.5",
            "concept": "Revenues",
            "accession": "0000000001-26-000030",
            "form": "10-Q",
            "derived": False,
            "derivation": None,
            "source_accessions": ["0000000001-26-000030"],
        }
    ]


async def test_search_with_dates_through_the_protocol(server: Any) -> None:
    async with Client(server) as client:
        result = await client.call_tool(
            "search_filings",
            {
                "ticker": "EXMP",
                "items": ["5.02"],
                "start_date": "2026-01-01",
                "end_date": str(date(2026, 11, 24)),
            },
        )
    assert [f["accession"] for f in result.structured_content["filings"]] == ["0000000001-26-000040"]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"ticker": "PRIVATECO"}, "unknown ticker"),  # readable ToolInputError
        ({"ticker": "drop table;"}, "validation error"),  # rejected by the input schema
    ],
)
async def test_errors_are_readable(server: Any, args: dict[str, Any], expected: str) -> None:
    async with Client(server) as client:
        result = await client.call_tool("lookup_company", args)
    assert result.is_error
    assert expected in result.content[0].text  # type: ignore[union-attr]


async def test_out_of_range_arguments_rejected(server: Any) -> None:
    async with Client(server) as client:
        result = await client.call_tool("get_financials", {"ticker": "EXMP", "quarters": 50})
    assert result.is_error
