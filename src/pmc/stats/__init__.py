"""Cluster D statistics and climatology utilities."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

MS_TO_KT = 1.9438445
EARTH_RADIUS_NM = 3440.065


@dataclass(frozen=True)
class TransectSpec:
    """Cross-shore transect definition for coastal enhancement analysis."""

    transect_id: str
    coast_name: str
    coast_lat: float
    coast_lon: float
    offshore_bearing_deg: float
    max_distance_nm: float = 60.0
    bin_size_nm: float = 2.0
    bearing_tolerance_deg: float = 30.0


DEFAULT_TRANSECTS: tuple[TransectSpec, ...] = (
    TransectSpec(
        transect_id="sardinia_ne",
        coast_name="Sardinia NE",
        coast_lat=41.13,
        coast_lon=9.55,
        offshore_bearing_deg=45.0,
    ),
    TransectSpec(
        transect_id="sardinia_e",
        coast_name="Sardinia East",
        coast_lat=40.45,
        coast_lon=9.90,
        offshore_bearing_deg=90.0,
    ),
    TransectSpec(
        transect_id="liguria_w",
        coast_name="Ligurian West",
        coast_lat=43.50,
        coast_lon=7.60,
        offshore_bearing_deg=180.0,
    ),
    TransectSpec(
        transect_id="liguria_c",
        coast_name="Ligurian Central",
        coast_lat=43.75,
        coast_lon=8.90,
        offshore_bearing_deg=180.0,
    ),
)


def climatology(wind: xr.Dataset, months: list[int]) -> xr.Dataset:
    """Compute C6 climatology by UTC hour for the selected months."""

    _assert_wind_dataset(wind)
    month_filter = set(int(m) for m in months)
    if not month_filter:
        raise ValueError("months must contain at least one month number")

    selected = wind.sel(time=wind.time.dt.month.isin(sorted(month_filter)))
    if selected.sizes.get("time", 0) == 0:
        raise ValueError("No samples remain after applying month filter")

    u = selected["u10"]
    v = selected["v10"]
    speed_ms = np.hypot(u, v)
    speed_kt = speed_ms * MS_TO_KT

    grouped_u = u.groupby("time.hour")
    grouped_v = v.groupby("time.hour")
    grouped_speed_ms = speed_ms.groupby("time.hour")
    grouped_speed_kt = speed_kt.groupby("time.hour")

    mean_tws_kt = grouped_speed_kt.mean(dim="time", skipna=True)
    vector_mean_u = grouped_u.mean(dim="time", skipna=True)
    vector_mean_v = grouped_v.mean(dim="time", skipna=True)
    p_below_5kt = (speed_kt < 5.0).groupby("time.hour").mean(dim="time", skipna=True)
    p_below_8kt = (speed_kt < 8.0).groupby("time.hour").mean(dim="time", skipna=True)
    p_above_20kt = (speed_kt > 20.0).groupby("time.hour").mean(dim="time", skipna=True)
    mean_speed_ms = grouped_speed_ms.mean(dim="time", skipna=True)
    directional_const = np.hypot(vector_mean_u, vector_mean_v) / mean_speed_ms
    n_samples = grouped_speed_kt.count(dim="time").astype(np.int32)

    out = xr.Dataset(
        data_vars={
            "mean_tws_kt": mean_tws_kt.astype(np.float32),
            "vector_mean_u": vector_mean_u.astype(np.float32),
            "vector_mean_v": vector_mean_v.astype(np.float32),
            "p_below_5kt": p_below_5kt.astype(np.float32),
            "p_below_8kt": p_below_8kt.astype(np.float32),
            "p_above_20kt": p_above_20kt.astype(np.float32),
            "directional_const": directional_const.astype(np.float32),
            "n_samples": n_samples,
        }
    )

    trusted = out["n_samples"] > 200
    for name in (
        "mean_tws_kt",
        "vector_mean_u",
        "vector_mean_v",
        "p_below_5kt",
        "p_below_8kt",
        "p_above_20kt",
        "directional_const",
    ):
        out[name] = out[name].where(trusted, np.nan)

    years = np.unique(selected.time.dt.year.values)
    out.attrs["source"] = str(selected.attrs.get("source", "unknown"))
    out.attrs["years_used"] = [int(v) for v in years]
    out.attrs["months_used"] = sorted(int(v) for v in month_filter)
    return out


def head_to_head(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build route pairwise win-rate and margin statistics (C7-style rows)."""

    frames: dict[str, pd.DataFrame] = {}
    for route_id, frame in results.items():
        required = {"start_time", "elapsed_hours"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{route_id}: missing columns {sorted(missing)}")
        subset = frame.loc[:, ["start_time", "elapsed_hours"]].copy()
        subset["start_time"] = pd.to_datetime(subset["start_time"], utc=True)
        subset = subset.dropna(subset=["elapsed_hours"])
        frames[route_id] = subset

    rows: list[dict[str, float | int | str]] = []
    for route_a, route_b in itertools.combinations(sorted(frames), 2):
        merged = frames[route_a].merge(
            frames[route_b],
            on="start_time",
            suffixes=("_a", "_b"),
            how="inner",
        )
        if merged.empty:
            continue

        margin_hours = merged["elapsed_hours_b"] - merged["elapsed_hours_a"]
        rows.append(
            {
                "a": route_a,
                "b": route_b,
                "a_wins_pct": round(float((margin_hours > 0).mean() * 100.0), 2),
                "median_margin_hours": round(float(margin_hours.median()), 2),
                "p10_margin_hours": round(float(margin_hours.quantile(0.10)), 2),
                "p90_margin_hours": round(float(margin_hours.quantile(0.90)), 2),
                "n": int(len(margin_hours)),
            }
        )

    return pd.DataFrame(rows).sort_values(["a", "b"]).reset_index(drop=True)


def model_skill(
    rows: pd.DataFrame,
    *,
    model_col: str = "model",
    lead_col: str = "lead_days",
    u_pred_col: str = "u10_pred",
    v_pred_col: str = "v10_pred",
    u_ref_col: str = "u10_ref",
    v_ref_col: str = "v10_ref",
) -> pd.DataFrame:
    """Compute model skill statistics stratified by lead and observed wind bin."""

    required = {model_col, lead_col, u_pred_col, v_pred_col, u_ref_col, v_ref_col}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"model_skill missing columns: {sorted(missing)}")

    data = rows.loc[:, list(required)].copy()
    for col in (u_pred_col, v_pred_col, u_ref_col, v_ref_col):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna()
    if data.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "lead_days",
                "wind_bin",
                "vec_rmse_kt",
                "speed_bias_kt",
                "dir_mae_deg",
                "n_samples",
                "reference_biased",
            ]
        )

    obs_speed_kt = np.hypot(data[u_ref_col], data[v_ref_col]) * MS_TO_KT
    pred_speed_kt = np.hypot(data[u_pred_col], data[v_pred_col]) * MS_TO_KT
    obs_dir = _uv_to_twd_deg(data[u_ref_col].to_numpy(), data[v_ref_col].to_numpy())
    pred_dir = _uv_to_twd_deg(data[u_pred_col].to_numpy(), data[v_pred_col].to_numpy())

    data["obs_speed_kt"] = obs_speed_kt
    data["pred_speed_kt"] = pred_speed_kt
    data["obs_dir_deg"] = obs_dir
    data["pred_dir_deg"] = pred_dir
    data["wind_bin"] = pd.cut(
        obs_speed_kt,
        bins=[0.0, 6.0, 12.0, 20.0, np.inf],
        labels=["0-6kt", "6-12kt", "12-20kt", "20kt+"],
        right=False,
    )

    output_rows: list[dict[str, float | str | bool]] = []
    group_cols = [model_col, lead_col, "wind_bin"]
    grouped = data.groupby(group_cols, dropna=True, observed=True)
    for (model, lead_days, wind_bin), frame in grouped:
        du = frame[u_pred_col] - frame[u_ref_col]
        dv = frame[v_pred_col] - frame[v_ref_col]
        vec_err_kt = np.hypot(du, dv) * MS_TO_KT

        dir_diff = _angular_difference_deg(
            frame["pred_dir_deg"].to_numpy(),
            frame["obs_dir_deg"].to_numpy(),
        )
        output_rows.append(
            {
                "model": str(model),
                "lead_days": int(lead_days),
                "wind_bin": str(wind_bin),
                "vec_rmse_kt": round(float(np.sqrt(np.mean(np.square(vec_err_kt)))), 2),
                "speed_bias_kt": round(float((frame["pred_speed_kt"] - frame["obs_speed_kt"]).mean()), 2),
                "dir_mae_deg": round(float(np.mean(np.abs(dir_diff))), 2),
                "n_samples": int(len(frame)),
                "reference_biased": _is_reference_biased_model(str(model)),
            }
        )

    return pd.DataFrame(output_rows).sort_values(["model", "lead_days", "wind_bin"]).reset_index(drop=True)


def cross_shore_transects(
    clim: xr.Dataset,
    transects: Iterable[TransectSpec] = DEFAULT_TRANSECTS,
) -> pd.DataFrame:
    """Compute mean wind by distance offshore and UTC hour for selected coasts."""

    if "mean_tws_kt" not in clim:
        raise ValueError("clim must contain mean_tws_kt")
    if "hour" not in clim.dims:
        raise ValueError("clim must include an hour dimension")

    lat_vals = np.asarray(clim["lat"].values, dtype=float)
    lon_vals = np.asarray(clim["lon"].values, dtype=float)
    hours = np.asarray(clim["hour"].values, dtype=int)
    mean_tws = np.asarray(clim["mean_tws_kt"].values, dtype=float)

    mesh_lat, mesh_lon = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    rows: list[dict[str, float | int | str]] = []

    for spec in transects:
        distance_nm = _haversine_nm(spec.coast_lat, spec.coast_lon, mesh_lat, mesh_lon)
        bearing_deg = _initial_bearing_deg(spec.coast_lat, spec.coast_lon, mesh_lat, mesh_lon)
        off_axis = np.abs(_angular_difference_deg(bearing_deg, spec.offshore_bearing_deg))

        mask = (
            (distance_nm >= 0.0)
            & (distance_nm <= spec.max_distance_nm)
            & (off_axis <= spec.bearing_tolerance_deg)
        )
        if not np.any(mask):
            continue

        bin_edges = np.arange(0.0, spec.max_distance_nm + spec.bin_size_nm, spec.bin_size_nm)
        if bin_edges[-1] < spec.max_distance_nm:
            bin_edges = np.append(bin_edges, spec.max_distance_nm)

        selected_dist = distance_nm[mask]
        selected_bins = np.digitize(selected_dist, bin_edges, right=False) - 1
        selected_bins = np.clip(selected_bins, 0, len(bin_edges) - 2)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        for hour_idx, hour_value in enumerate(hours):
            hour_grid = mean_tws[hour_idx, :, :]
            hour_values = hour_grid[mask]
            valid = np.isfinite(hour_values)
            if not np.any(valid):
                continue

            for bin_idx, center in enumerate(bin_centers):
                in_bin = valid & (selected_bins == bin_idx)
                if not np.any(in_bin):
                    continue
                rows.append(
                    {
                        "transect_id": spec.transect_id,
                        "coast_name": spec.coast_name,
                        "hour": int(hour_value),
                        "distance_offshore_nm": round(float(center), 2),
                        "mean_tws_kt": round(float(np.nanmean(hour_values[in_bin])), 2),
                        "n_cells": int(np.count_nonzero(in_bin)),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "transect_id",
                "coast_name",
                "hour",
                "distance_offshore_nm",
                "mean_tws_kt",
                "n_cells",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["transect_id", "hour", "distance_offshore_nm"]
    ).reset_index(drop=True)


def estimate_coastal_enhancement_range(
    transects: pd.DataFrame,
    *,
    drop_threshold_kt: float = 0.8,
) -> pd.DataFrame:
    """Estimate where coastal enhancement decays offshore by hour/transect."""

    required = {"transect_id", "coast_name", "hour", "distance_offshore_nm", "mean_tws_kt"}
    missing = required - set(transects.columns)
    if missing:
        raise ValueError(f"transects missing columns: {sorted(missing)}")

    rows: list[dict[str, float | int | str]] = []
    grouped = transects.groupby(["transect_id", "coast_name", "hour"], dropna=False, observed=True)
    for (transect_id, coast_name, hour), frame in grouped:
        frame = frame.sort_values("distance_offshore_nm")
        if frame.empty:
            continue
        near_coast = float(frame["mean_tws_kt"].iloc[0])
        threshold = near_coast - drop_threshold_kt
        farther = frame[frame["mean_tws_kt"] <= threshold]
        if farther.empty:
            decay_distance = float(frame["distance_offshore_nm"].max())
        else:
            decay_distance = float(farther["distance_offshore_nm"].iloc[0])
        rows.append(
            {
                "transect_id": str(transect_id),
                "coast_name": str(coast_name),
                "hour": int(hour),
                "enhancement_decay_nm": round(decay_distance, 2),
                "near_coast_tws_kt": round(near_coast, 2),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "transect_id",
                "coast_name",
                "hour",
                "enhancement_decay_nm",
                "near_coast_tws_kt",
            ]
        )

    return pd.DataFrame(rows).sort_values(["transect_id", "hour"]).reset_index(drop=True)


def _assert_wind_dataset(wind: xr.Dataset) -> None:
    needed_dims = {"time", "lat", "lon"}
    if not needed_dims.issubset(set(wind.dims)):
        raise ValueError(f"wind dataset must include dims {sorted(needed_dims)}")
    for var in ("u10", "v10"):
        if var not in wind:
            raise ValueError(f"wind dataset missing variable '{var}'")


def _uv_to_twd_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.mod(np.degrees(np.arctan2(-u, -v)), 360.0)


def _angular_difference_deg(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def _is_reference_biased_model(model: str) -> bool:
    lowered = model.lower()
    return "ecmwf" in lowered or "aifs" in lowered


def _haversine_nm(
    lat1_deg: float,
    lon1_deg: float,
    lat2_deg: np.ndarray,
    lon2_deg: np.ndarray,
) -> np.ndarray:
    lat1 = np.deg2rad(lat1_deg)
    lon1 = np.deg2rad(lon1_deg)
    lat2 = np.deg2rad(lat2_deg)
    lon2 = np.deg2rad(lon2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_NM * c


def _initial_bearing_deg(
    lat1_deg: float,
    lon1_deg: float,
    lat2_deg: np.ndarray,
    lon2_deg: np.ndarray,
) -> np.ndarray:
    lat1 = math.radians(lat1_deg)
    lon1 = math.radians(lon1_deg)
    lat2 = np.deg2rad(lat2_deg)
    lon2 = np.deg2rad(lon2_deg)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return np.mod(bearing + 360.0, 360.0)

