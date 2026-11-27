"""Ticker -> CIK resolution from SEC's company_tickers.json."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TickerEntry:
    ticker: str
    cik: str  # 10-digit, zero-padded
    title: str


class UnknownTickerError(KeyError):
    pass


def _norm(ticker: str) -> str:
    # SEC spells share classes with "-" (BRK-B); people often type "." (BRK.B).
    return ticker.strip().upper().replace(".", "-")


class TickerIndex:
    def __init__(self, entries: list[TickerEntry]) -> None:
        self._by_ticker = {e.ticker: e for e in entries}

    @classmethod
    def from_sec_json(cls, data: dict[str, Any]) -> TickerIndex:
        """Parse the `{"0": {"cik_str": 320193, "ticker": "AAPL", "title": ...}, ...}` shape."""
        entries = [
            TickerEntry(
                ticker=_norm(str(row["ticker"])),
                cik=str(int(row["cik_str"])).zfill(10),
                title=str(row["title"]),
            )
            for row in data.values()
        ]
        return cls(entries)

    def resolve(self, ticker: str) -> TickerEntry:
        key = _norm(ticker)
        try:
            return self._by_ticker[key]
        except KeyError:
            raise UnknownTickerError(
                f"{ticker!r} is not in SEC's ticker list (private company, delisted, or a typo?)"
            ) from None

    def __len__(self) -> int:
        return len(self._by_ticker)
