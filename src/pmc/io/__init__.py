"""Cluster D IO module entrypoints."""

from .openmeteo import Domain, FetchSummary, OpenMeteoFetcher, fetch_wind, load_domain

__all__ = ["Domain", "FetchSummary", "OpenMeteoFetcher", "fetch_wind", "load_domain"]

