"""Cluster D IO module entrypoints."""

from .yb import (
    DEFAULT_YEARS,
    build_overlay,
    fetch_year,
    load_edition,
    parse_leaderboard_csv,
    write_overlay,
)

__all__ = [
    "DEFAULT_YEARS",
    "Domain",
    "FetchSummary",
    "OpenMeteoFetcher",
    "assert_hourly_units",
    "build_overlay",
    "fetch_wind",
    "fetch_year",
    "load_domain",
    "load_edition",
    "parse_leaderboard_csv",
    "write_overlay",
]


def __getattr__(name: str):
    if name in {"Domain", "FetchSummary", "OpenMeteoFetcher", "fetch_wind", "load_domain"}:
        from . import openmeteo

        return getattr(openmeteo, name)
    if name == "assert_hourly_units":
        from .units import assert_hourly_units

        return assert_hourly_units
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
