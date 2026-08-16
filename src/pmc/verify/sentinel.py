"""Sentinel-1 opportunistic wind-speed display (never scored)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np


def empty_sentinel_payload() -> dict[str, Any]:
    return {
        "status": "no_acquisition",
        "acquisition_utc": None,
        "footprint": None,
        "speed_field": None,
        "model_speed_fields": [],
    }


def query_sentinel1_wind(
    *,
    lat_min: float = 37.5,
    lat_max: float = 44.0,
    lon_min: float = 6.5,
    lon_max: float = 14.5,
    lookback_hours: int = 24,
    client: Any | None = None,
) -> dict[str, Any]:
    """Query Copernicus Marine Wind TAC for Sentinel-1 L3 wind in the domain.

    When no client/network is available, returns ``no_acquisition`` cleanly.
    Never raises for empty results. Direction is intentionally omitted — the
    dual-pol inversion uses a model a priori, so direction is not independent.
    """
    if client is None:
        return empty_sentinel_payload()
    try:
        result = client.find_swaths(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            since=datetime.now(timezone.utc) - timedelta(hours=lookback_hours),
        )
    except Exception:  # noqa: BLE001 — opportunistic; empty is success
        return empty_sentinel_payload()
    if not result:
        return empty_sentinel_payload()
    swath = result[0]
    return {
        "status": "available",
        "acquisition_utc": swath.get("acquisition_utc"),
        "footprint": swath.get("footprint"),
        "speed_field": {
            "lat": swath.get("lat"),
            "lon": swath.get("lon"),
            "speed_ms": swath.get("speed_ms"),
            # direction intentionally absent
        },
        "model_speed_fields": swath.get("model_speed_fields") or [],
    }


def synthetic_sentinel_swath() -> dict[str, Any]:
    """Fixture-grade 1 km-ish speed field over Sardinian east coast."""
    lats = np.round(np.arange(39.5, 41.0, 0.05), 4).tolist()
    lons = np.round(np.arange(9.7, 10.3, 0.05), 4).tolist()
    speed = [
        [float(6.0 + 0.4 * (i % 5) + 0.2 * (j % 3)) for j in range(len(lons))]
        for i in range(len(lats))
    ]
    return {
        "status": "available",
        "acquisition_utc": "2026-08-10T05:12:00Z",
        "footprint": {
            "type": "Polygon",
            "coordinates": [
                [
                    [9.7, 39.5],
                    [10.3, 39.5],
                    [10.3, 41.0],
                    [9.7, 41.0],
                    [9.7, 39.5],
                ]
            ],
        },
        "speed_field": {"lat": lats, "lon": lons, "speed_ms": speed},
        "model_speed_fields": [
            {
                "model": "gfs_global",
                "valid_time_utc": "2026-08-10T06:00:00Z",
                "lat": lats,
                "lon": lons,
                "speed_ms": [
                    [float(6.5 + 0.3 * (i % 5)) for j in range(len(lons))]
                    for i in range(len(lats))
                ],
            }
        ],
    }
