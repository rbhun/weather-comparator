"""A3 — Fixed-route follower through historical wind fields."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import xarray as xr

from contracts.schemas import (
    FollowResult,
    Polar,
    Route,
    advance_position,
    angular_difference,
    haversine_nm,
    initial_bearing_deg,
    uv_to_tws_twd,
)

DEFAULT_DT_MINUTES = 10
DEFAULT_STALL_THRESHOLD_KT = 0.5
DEFAULT_STALL_HOURS = 6.0
DEFAULT_MAX_SIM_HOURS = 180.0

def _route_distance_nm(route: Route) -> float:
    total = 0.0
    for (lat0, lon0), (lat1, lon1) in zip(route.legs[:-1], route.legs[1:]):
        total += float(haversine_nm(lat0, lon0, lat1, lon1))
    return total


def _interp_1d(values: np.ndarray, x: float) -> tuple[int, int, float]:
    idx_hi = int(np.searchsorted(values, x, side="right"))
    if idx_hi <= 0:
        return 0, 0, 0.0
    if idx_hi >= values.size:
        i = values.size - 1
        return i, i, 0.0
    idx_lo = idx_hi - 1
    x0 = float(values[idx_lo])
    x1 = float(values[idx_hi])
    if x1 == x0:
        return idx_lo, idx_hi, 0.0
    w = (x - x0) / (x1 - x0)
    return idx_lo, idx_hi, float(np.clip(w, 0.0, 1.0))


def _bilinear_value(
    field_2d: np.ndarray, lat_vals: np.ndarray, lon_vals: np.ndarray, lat: float, lon: float
) -> float:
    lat_c = float(np.clip(lat, lat_vals[0], lat_vals[-1]))
    lon_c = float(np.clip(lon, lon_vals[0], lon_vals[-1]))
    i0, i1, wa = _interp_1d(lat_vals, lat_c)
    j0, j1, wb = _interp_1d(lon_vals, lon_c)

    f00 = float(field_2d[i0, j0])
    f10 = float(field_2d[i1, j0])
    f01 = float(field_2d[i0, j1])
    f11 = float(field_2d[i1, j1])
    corners = np.array([f00, f10, f01, f11], dtype=float)
    valid = np.isfinite(corners)
    if valid.sum() == 0:
        i_center = int(np.clip(np.searchsorted(lat_vals, lat_c), 0, lat_vals.size - 1))
        j_center = int(np.clip(np.searchsorted(lon_vals, lon_c), 0, lon_vals.size - 1))
        max_radius = max(lat_vals.size, lon_vals.size)
        for radius in range(1, max_radius + 1):
            i_lo = max(0, i_center - radius)
            i_hi = min(lat_vals.size, i_center + radius + 1)
            j_lo = max(0, j_center - radius)
            j_hi = min(lon_vals.size, j_center + radius + 1)
            window = field_2d[i_lo:i_hi, j_lo:j_hi]
            finite = window[np.isfinite(window)]
            if finite.size:
                return float(np.mean(finite))
        return float("nan")
    if valid.sum() < 4:
        return float(np.nanmean(corners))
    return float(
        (1.0 - wa) * (1.0 - wb) * f00
        + wa * (1.0 - wb) * f10
        + (1.0 - wa) * wb * f01
        + wa * wb * f11
    )


def _interpolate_uv(wind: xr.Dataset, t: np.datetime64, lat: float, lon: float) -> tuple[float, float]:
    times = wind["time"].values.astype("datetime64[ns]")
    time_ns = times.astype(np.int64)
    t_ns = int(t.astype("datetime64[ns]").astype(np.int64))
    if time_ns.size >= 2:
        dt = int(np.median(np.diff(time_ns)))
        period = int((time_ns[-1] - time_ns[0]) + dt)
        if period > 0:
            t_ns = int(time_ns[0] + ((t_ns - time_ns[0]) % period))
    i0, i1, wt = _interp_1d(time_ns, t_ns)

    lat_vals = wind["lat"].values.astype(float)
    lon_vals = wind["lon"].values.astype(float)
    u0 = _bilinear_value(wind["u10"].values[i0], lat_vals, lon_vals, lat, lon)
    v0 = _bilinear_value(wind["v10"].values[i0], lat_vals, lon_vals, lat, lon)
    if i0 == i1:
        return u0, v0
    u1 = _bilinear_value(wind["u10"].values[i1], lat_vals, lon_vals, lat, lon)
    v1 = _bilinear_value(wind["v10"].values[i1], lat_vals, lon_vals, lat, lon)
    if not np.isfinite([u0, v0, u1, v1]).all():
        u_candidates = [v for v in (u0, u1) if np.isfinite(v)]
        v_candidates = [v for v in (v0, v1) if np.isfinite(v)]
        if not u_candidates or not v_candidates:
            return float("nan"), float("nan")
        return float(np.mean(u_candidates)), float(np.mean(v_candidates))
    return (1.0 - wt) * u0 + wt * u1, (1.0 - wt) * v0 + wt * v1


def follow(
    route: Route,
    wind: xr.Dataset,
    polar: Polar,
    start: datetime,
    *,
    dt_minutes: int = DEFAULT_DT_MINUTES,
    stall_threshold_kt: float = DEFAULT_STALL_THRESHOLD_KT,
    stall_hours_threshold: float = DEFAULT_STALL_HOURS,
    max_sim_hours: float = DEFAULT_MAX_SIM_HOURS,
) -> FollowResult:
    """March a boat along a fixed polyline through the wind field."""
    if len(route.legs) < 2:
        raise ValueError("Route must have at least two waypoints.")

    current_lat, current_lon = route.legs[0]
    waypoint_idx = 1
    dt_hours = dt_minutes / 60.0
    max_steps = max(1, int(max_sim_hours / dt_hours))

    if start.tzinfo is None:
        start_utc = start
    else:
        start_utc = start.astimezone(timezone.utc).replace(tzinfo=None)
    now_utc = np.datetime64(start_utc, "ns")
    elapsed_hours = 0.0
    sailed_distance_nm = 0.0
    tws_weighted_sum = 0.0
    hours_below_5 = 0.0
    hours_upwind = 0.0
    stall_streak_h = 0.0
    max_stall_h = 0.0
    stalled = False

    for _ in range(max_steps):
        if waypoint_idx >= len(route.legs):
            break
        next_lat, next_lon = route.legs[waypoint_idx]
        leg_bearing = initial_bearing_deg(current_lat, current_lon, next_lat, next_lon)
        leg_distance_nm = float(haversine_nm(current_lat, current_lon, next_lat, next_lon))
        if leg_distance_nm < 1e-6:
            current_lat, current_lon = next_lat, next_lon
            waypoint_idx += 1
            continue

        u10, v10 = _interpolate_uv(wind, now_utc, current_lat, current_lon)
        if not np.isfinite([u10, v10]).all():
            tws_kt = 0.0
            twd_deg = 0.0
        else:
            tws_kt, twd_deg = uv_to_tws_twd(u10, v10)
            tws_kt = float(np.asarray(tws_kt))
            twd_deg = float(np.asarray(twd_deg))

        twa_signed = float(angular_difference(twd_deg, leg_bearing))
        abs_twa = abs(twa_signed)
        if tws_kt < stall_threshold_kt:
            bsp_kt = 0.0
            progress_kt = 0.0
        else:
            bsp_kt = float(np.asarray(polar.speed(abs_twa, tws_kt)))
            progress_kt = bsp_kt

            upwind_twa, upwind_bsp = polar.vmg_optimum(tws_kt, upwind=True)
            downwind_twa, downwind_bsp = polar.vmg_optimum(tws_kt, upwind=False)

            if abs_twa < upwind_twa:
                bsp_kt = float(upwind_bsp)
                progress_kt = max(0.0, bsp_kt * np.cos(np.radians(upwind_twa)))
            elif abs_twa > downwind_twa:
                bsp_kt = float(downwind_bsp)
                progress_kt = max(0.0, -bsp_kt * np.cos(np.radians(downwind_twa)))

        if bsp_kt < stall_threshold_kt:
            stall_streak_h += dt_hours
            max_stall_h = max(max_stall_h, stall_streak_h)
            if stall_streak_h > stall_hours_threshold:
                stalled = True
        else:
            stall_streak_h = 0.0

        tws_weighted_sum += tws_kt * dt_hours
        if tws_kt < 5.0:
            hours_below_5 += dt_hours
        if abs_twa < 60.0:
            hours_upwind += dt_hours

        progress_nm = progress_kt * dt_hours
        sailed_nm = bsp_kt * dt_hours
        if progress_nm >= leg_distance_nm:
            current_lat, current_lon = next_lat, next_lon
            waypoint_idx += 1
        elif progress_nm > 0:
            current_lat, current_lon = advance_position(
                current_lat, current_lon, leg_bearing, progress_nm
            )

        elapsed_hours += dt_hours
        sailed_distance_nm += sailed_nm
        now_utc = now_utc + np.timedelta64(int(round(dt_hours * 3600.0)), "s")

    if waypoint_idx < len(route.legs):
        stalled = True

    mean_tws = tws_weighted_sum / elapsed_hours if elapsed_hours > 0 else np.nan
    return FollowResult(
        start_time=start,
        route_id=route.id,
        elapsed_hours=float(np.round(elapsed_hours, 2)),
        distance_nm=float(np.round(sailed_distance_nm, 2)),
        mean_tws_kt=float(np.round(mean_tws, 2)),
        hours_below_5kt=float(np.round(hours_below_5, 2)),
        hours_upwind=float(np.round(hours_upwind, 2)),
        stalled=stalled,
        max_stall_hours=float(np.round(max_stall_h, 2)),
    )
