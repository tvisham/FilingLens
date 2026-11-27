"""Live: the MCP server as a real stdio subprocess (opt-in: RUN_LIVE=1, SEC_USER_AGENT in .env).

Checks the transport the agent uses, and that the MCP result matches the Phase 1
pipeline exactly for the same company.
"""

from __future__ import annotations

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

from filinglens.edgar.cache import DiskCache
from filinglens.edgar.client import SecClient
from filinglens.mcp_server.server import TOOL_NAMES
from filinglens.pipeline import load_ticker_index, snapshot_company
from filinglens.settings import get_settings

pytestmark = [pytest.mark.live, pytest.mark.anyio]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


async def test_stdio_server_matches_pipeline() -> None:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "filinglens.mcp_server"], cwd=str(ROOT)
    )
    async with Client(params) as client:
        names = sorted(t.name for t in (await client.list_tools()).tools)
        fin = await client.call_tool("get_financials", {"ticker": "AAPL", "quarters": 2})
        idx = await client.call_tool("get_filing_index", {"ticker": "XOM", "forms": ["10-K"]})
    assert names == sorted(TOOL_NAMES)
    assert not fin.is_error and not idx.is_error

    s = get_settings()
    with SecClient(s.sec_user_agent, DiskCache(s.cache_dir)) as c:
        snap = snapshot_company(c, load_ticker_index(c), "AAPL", as_of=date.today())
    assert snap.financials is not None
    expected = [(q.period_end.isoformat(), q.value) for q in snap.financials.revenue.latest_two]
    got = [(q["period_end"], Decimal(q["value_usd"])) for q in fin.structured_content["revenue"]["quarters"]]
    assert got == expected

    # XOM's 10-Ks before the 2026 reorganization come from the predecessor CIK, with its Archive path.
    ten_k_urls = [f["url"] for f in idx.structured_content["filings"]]
    assert ten_k_urls and all("/edgar/data/34088/" in u for u in ten_k_urls)
    print("\nAAPL revenue via MCP stdio:", got, "\nXOM 10-K urls:", *ten_k_urls, sep="\n  ")
