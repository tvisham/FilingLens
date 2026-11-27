"""Loading the company universe from `config/companies.yaml`."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class CompanySpec(BaseModel):
    ticker: str
    sector: str
    predecessor_ciks: list[str] = Field(default_factory=list)


def _pad_cik(value: str | int) -> str:
    digits = str(value).strip().upper().removeprefix("CIK")
    if not digits.isdigit() or len(digits) > 10:
        raise ValueError(f"not a CIK: {value!r}")
    return digits.zfill(10)


class Universe(BaseModel):
    sectors: dict[str, list[str]]
    # ticker -> earlier registrant CIKs (successor issuers). The only hand-typed CIKs, since
    # SEC's ticker file maps a ticker to its current registrant only.
    predecessors: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("predecessors", mode="before")
    @classmethod
    def _normalise_predecessors(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        return {str(t).strip().upper(): [_pad_cik(c) for c in ciks] for t, ciks in value.items()}

    @model_validator(mode="after")
    def _predecessors_are_listed(self) -> Universe:
        listed = {t for tickers in self.sectors.values() for t in tickers}
        unknown = sorted(set(self.predecessors) - listed)
        if unknown:
            raise ValueError(f"predecessors given for tickers not in any sector: {unknown}")
        return self

    @field_validator("sectors")
    @classmethod
    def _normalise(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        seen: set[str] = set()
        out: dict[str, list[str]] = {}
        for sector, tickers in value.items():
            clean = [t.strip().upper() for t in tickers if t.strip()]
            dupes = seen.intersection(clean)
            if dupes:
                raise ValueError(f"tickers listed in more than one sector: {sorted(dupes)}")
            seen.update(clean)
            out[sector.strip().lower()] = clean
        return out

    def companies(self) -> list[CompanySpec]:
        return [
            CompanySpec(ticker=t, sector=sector, predecessor_ciks=self.predecessors_for(t))
            for sector, tickers in self.sectors.items()
            for t in tickers
        ]

    def predecessors_for(self, ticker: str) -> list[str]:
        return list(self.predecessors.get(ticker.strip().upper(), []))


def load_universe(path: Path) -> Universe:
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Universe.model_validate(raw)
