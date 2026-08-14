"""A1 — Data pipeline STUB module."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from contracts.schemas import Domain


def fetch_wind(source: str, start: date, end: date, cfg: Domain) -> Path:
    """Return fixture wind store path for contract-first development."""
    _ = (source, start, end, cfg)
    return Path("contracts/fixtures/wind_small.zarr")
