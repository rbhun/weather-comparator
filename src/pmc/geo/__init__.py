"""A2 — Geometry and polar STUB geo module."""

from __future__ import annotations

import numpy as np


def _is_land(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    # FIXTURE-GRADE ONLY: crude rectangles standing in for coastlines.
    return (
        ((lat >= 38.7) & (lat <= 41.3) & (lon >= 8.0) & (lon <= 9.8))
        | ((lat >= 41.2) & (lat <= 43.2) & (lon >= 8.3) & (lon <= 9.7))
        | ((lat <= 38.5) & (lon >= 12.0))
        | ((lat >= 43.5) & (lon >= 8.2))
    )


def is_sea(lat, lon) -> np.ndarray:
    """Return True where location is sea in the fixture-grade land mask."""
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    lat_b, lon_b = np.broadcast_arrays(lat_arr, lon_arr)
    return ~_is_land(lat_b, lon_b)


def crosses_land(lat0, lon0, lat1, lon1) -> bool:
    """Return whether a straight segment intersects fixture-grade land boxes."""
    frac = np.linspace(0.0, 1.0, 400)
    lat = float(lat0) + (float(lat1) - float(lat0)) * frac
    lon = float(lon0) + (float(lon1) - float(lon0)) * frac
    return bool(np.any(~is_sea(lat, lon)))
