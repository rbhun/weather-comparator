"""A3 — Route follower STUB module."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import xarray as xr

from contracts.schemas import FollowResult, Polar, Route, haversine_nm, uv_to_tws_twd


def _route_distance_nm(route: Route) -> float:
    total = 0.0
    for (lat0, lon0), (lat1, lon1) in zip(route.legs[:-1], route.legs[1:]):
        total += float(haversine_nm(lat0, lon0, lat1, lon1))
    return total


def follow(route: Route, wind: xr.Dataset, polar: Polar, start: datetime) -> FollowResult:
    """Return a schema-valid synthetic row for skeleton integration."""
    _ = polar
    dist = _route_distance_nm(route)
    tws_kt, _ = uv_to_tws_twd(wind["u10"].values, wind["v10"].values)
    mean_tws = float(np.nanmean(tws_kt))
    elapsed = dist / 7.2
    hours_below_5 = float(np.nanmean(tws_kt < 5.0) * elapsed)
    hours_upwind = 0.35 * elapsed
    return FollowResult(
        start_time=start,
        route_id=route.id,
        elapsed_hours=float(np.round(elapsed, 2)),
        distance_nm=float(np.round(dist, 2)),
        mean_tws_kt=float(np.round(mean_tws, 2)),
        hours_below_5kt=float(np.round(hours_below_5, 2)),
        hours_upwind=float(np.round(hours_upwind, 2)),
        stalled=False,
        max_stall_hours=0.0,
    )
