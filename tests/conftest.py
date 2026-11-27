"""Test configuration: live and database tests are opt-in."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_live = pytest.mark.skip(reason="live SEC test: set RUN_LIVE=1")
    skip_db = pytest.mark.skip(reason="database test: set TEST_DATABASE_URL")
    for item in items:
        if "live" in item.keywords and os.environ.get("RUN_LIVE") != "1":
            item.add_marker(skip_live)
        if "db" in item.keywords and not os.environ.get("TEST_DATABASE_URL"):
            item.add_marker(skip_db)
