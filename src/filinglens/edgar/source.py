"""What the deterministic pipeline needs from EDGAR: the live client or the bulk archives."""

from __future__ import annotations

from typing import Any, Protocol


class EdgarSource(Protocol):
    def company_tickers(self) -> dict[str, Any]: ...

    def submissions(self, cik: int | str) -> dict[str, Any]: ...

    def submissions_page(self, name: str) -> dict[str, Any]: ...

    def company_facts(self, cik: int | str) -> dict[str, Any]: ...
