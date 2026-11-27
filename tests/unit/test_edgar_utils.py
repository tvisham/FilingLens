from __future__ import annotations

import random
from decimal import Decimal
from pathlib import Path

import pytest

from filinglens.config import Universe, load_universe
from filinglens.edgar.cache import DiskCache
from filinglens.edgar.ratelimit import TokenBucket
from filinglens.edgar.retry import RetryPolicy
from filinglens.edgar.tickers import TickerIndex, UnknownTickerError
from filinglens.edgar.useragent import UserAgentError, validate_user_agent
from filinglens.formatting import money


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


# -- rate limiter ------------------------------------------------------------------------


def test_token_bucket_allows_burst_then_throttles() -> None:
    clock = FakeClock()
    bucket = TokenBucket(8, clock=clock, sleep=clock.sleep)
    for _ in range(8):
        bucket.acquire()
    assert clock.sleeps == []
    bucket.acquire()
    assert clock.sleeps and clock.sleeps[0] == pytest.approx(1 / 8)


def test_token_bucket_sustained_rate_never_exceeds_limit() -> None:
    clock = FakeClock()
    bucket = TokenBucket(8, capacity=1, clock=clock, sleep=clock.sleep)
    for _ in range(81):
        bucket.acquire()
    # 81 requests with no burst: at least 80 intervals of 1/8 s.
    assert clock.now >= 10.0 - 1e-9


def test_token_bucket_rejects_bad_rate() -> None:
    with pytest.raises(ValueError):
        TokenBucket(0)


# -- cache -------------------------------------------------------------------------------


def test_cache_roundtrip_and_ttl(tmp_path: Path) -> None:
    clock = FakeClock()
    cache = DiskCache(tmp_path, clock=clock)
    assert cache.get("https://x", 10) is None
    cache.set("https://x", {"a": 1})
    assert cache.get("https://x", 10) == {"a": 1}
    clock.now += 11
    assert cache.get("https://x", 10) is None


def test_cache_ignores_corrupt_file(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.set("k", [1])
    path = next(tmp_path.rglob("*.json"))
    path.write_text("{not json", encoding="utf-8")
    assert cache.get("k", 100) is None


# -- retry -------------------------------------------------------------------------------


def test_retry_honours_retry_after_and_caps_it() -> None:
    policy = RetryPolicy(max_delay=30)
    assert policy.delay(1, "2") == 2.0
    assert policy.delay(1, "999") == 30.0


def test_retry_backoff_is_bounded_exponential() -> None:
    policy = RetryPolicy(base_delay=0.5, max_delay=4, rng=random.Random(0))
    for attempt, ceiling in [(1, 0.5), (2, 1.0), (3, 2.0), (4, 4.0), (8, 4.0)]:
        assert 0 <= policy.delay(attempt) <= ceiling


def test_retryable_statuses() -> None:
    policy = RetryPolicy()
    assert policy.should_retry_status(429)
    assert policy.should_retry_status(503)
    assert not policy.should_retry_status(404)
    assert not policy.should_retry_status(403)


# -- user agent --------------------------------------------------------------------------


def test_user_agent_requires_name_and_email() -> None:
    assert validate_user_agent("  FilingLens Jane jane@example.edu ") == ("FilingLens Jane jane@example.edu")
    with pytest.raises(UserAgentError):
        validate_user_agent("FilingLens")
    with pytest.raises(UserAgentError):
        validate_user_agent("jane@example.edu")


# -- tickers -----------------------------------------------------------------------------


def test_ticker_index_pads_cik_and_normalises_share_classes() -> None:
    index = TickerIndex.from_sec_json(
        {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
        }
    )
    assert index.resolve("aapl").cik == "0000320193"
    assert index.resolve("BRK.B").ticker == "BRK-B"
    with pytest.raises(UnknownTickerError):
        index.resolve("STATEFARM")


# -- config ------------------------------------------------------------------------------


def test_repo_universe_has_25_unique_companies() -> None:
    universe = load_universe(Path(__file__).parents[2] / "config" / "companies.yaml")
    companies = universe.companies()
    assert len(companies) == 25
    assert len({c.ticker for c in companies}) == 25
    assert set(universe.sectors) == {"technology", "financials", "healthcare", "energy", "consumer"}


def test_repo_universe_predecessors() -> None:
    universe = load_universe(Path(__file__).parents[2] / "config" / "companies.yaml")
    xom = next(c for c in universe.companies() if c.ticker == "XOM")
    assert xom.predecessor_ciks == ["0000034088"]
    assert universe.predecessors_for("aapl") == []


def test_universe_pads_predecessor_ciks() -> None:
    universe = Universe.model_validate({"sectors": {"a": ["XOM"]}, "predecessors": {"xom": [34088, "CIK12"]}})
    assert universe.predecessors_for("XOM") == ["0000034088", "0000000012"]


@pytest.mark.parametrize(
    "predecessors",
    [{"MSFT": ["1"]}, {"XOM": ["not-a-cik"]}, {"XOM": ["12345678901"]}],
)
def test_universe_rejects_bad_predecessors(predecessors: dict[str, list[str]]) -> None:
    with pytest.raises(ValueError):
        Universe.model_validate({"sectors": {"a": ["XOM"]}, "predecessors": predecessors})


def test_universe_rejects_duplicate_tickers() -> None:
    with pytest.raises(ValueError, match="more than one sector"):
        Universe.model_validate({"sectors": {"a": ["AAPL"], "b": ["aapl"]}})


# -- formatting --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Decimal("94930000000"), "$94.93B"),
        (Decimal("-1500000"), "-$1.50M"),
        (Decimal("999"), "$999"),
    ],
)
def test_money(value: Decimal, expected: str) -> None:
    assert money(value) == expected
