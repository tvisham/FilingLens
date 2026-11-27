"""Live: the MCP server over Streamable HTTP (opt-in: RUN_LIVE=1). Starts a real subprocess."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from mcp import Client

from filinglens.mcp_server.server import TOOL_NAMES

pytestmark = [pytest.mark.live, pytest.mark.anyio]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def test_streamable_http_server() -> None:
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "filinglens.mcp_server", "--http", "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.5)
        async with Client(f"http://127.0.0.1:{port}/mcp") as client:
            names = {t.name for t in (await client.list_tools()).tools}
            result = await client.call_tool("lookup_company", {"ticker": "AAPL"})
        assert names == set(TOOL_NAMES)
        assert not result.is_error and result.structured_content["cik"] == "0000320193"
    finally:
        proc.terminate()
        proc.wait(timeout=10)
