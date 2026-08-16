"""Collocation: QC buckets, coastal/light-air separation, thinning."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from contracts.schemas import VERIFY_LIGHT_AIR_MS, haversine_nm

from .conventions import lead_bucket_for_hours, region_for_point, speed_bucket_for_obs_ms
from .store import VerifyConfig, cell_id_for


# Crude coastline proxy points for land-distance (nm→km). Good enough for
# fixture-grade coastal bucketing; production should use geo coastline.
_COAST_PROXIES = np.array(
    [
        [38.2, 13.1],
        [39.2, 9.6],
        [40.1, 9.7],
        [40.9, 9.6],
        [41.2, 9.4],
        [41.5, 9.2],
        [42.0, 8.7],
        [42.5, 8.8],
        [43.3, 7.8],
        [43.7, 7.4],
        [44.0, 8.9],
        [41.9, 8.6],
        [39.9, 8.4],
    ],
    dtype=float,
)


def land_distance_km(lat: float, lon: float) -> float:
    d_nm = haversine_nm(lat, lon, _COAST_PROXIES[:, 0], _COAST_PROXIES[:, 1])
    return float(np.min(d_nm) * 1.852)


def thin_to_grid(cells: pd.DataFrame, grid_deg: float, max_points: int) -> pd.DataFrame:
    if cells.empty:
        return cells
    frame = cells.copy()
    if "obs_speed_ms" not in frame.columns:
        frame["obs_speed_ms"] = np.hypot(
            frame["obs_u10"].to_numpy(dtype=float),
            frame["obs_v10"].to_numpy(dtype=float),
        )
    if "source_file_hash" not in frame.columns:
        frame["source_file_hash"] = ""
    frame["_glat"] = np.round(frame["lat"].to_numpy(dtype=float) / grid_deg) * grid_deg
    frame["_glon"] = np.round(frame["lon"].to_numpy(dtype=float) / grid_deg) * grid_deg

    rows: list[dict] = []
    for (_, _, instrument), grp in frame.groupby(["_glat", "_glon", "instrument"]):
        rows.append(
            {
                "lat": float(grp["_glat"].iloc[0]),
                "lon": float(grp["_glon"].iloc[0]),
                "time": pd.Timestamp(grp["time"].mean()),
                "obs_u10": float(grp["obs_u10"].mean()),
                "obs_v10": float(grp["obs_v10"].mean()),
                "obs_speed_ms": float(grp["obs_speed_ms"].mean()),
                "n_raw": int(len(grp)),
                "instrument": str(instrument),
                "obs_class": str(grp["obs_class"].iloc[0]),
                "source_file_hash": str(grp["source_file_hash"].iloc[0]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("n_raw", ascending=False).head(max_points).reset_index(drop=True)


def assign_buckets(cells: pd.DataFrame, cfg: VerifyConfig) -> pd.DataFrame:
    """Label each cell headline / coastal / light_air (QC already applied at ingest)."""
    if cells.empty:
        return cells
    rows = []
    regions = cfg.regions or {}
    for rec in cells.to_dict(orient="records"):
        lat = float(rec["lat"])
        lon = float(rec["lon"])
        spd = float(rec.get("obs_speed_ms", np.hypot(rec["obs_u10"], rec["obs_v10"])))
        dist = land_distance_km(lat, lon)
        if dist < cfg.land_distance_km:
            label = "coastal"
        elif spd < cfg.light_air_ms:
            label = "light_air"
        else:
            label = "headline"
        rows.append(
            {
                **rec,
                "land_dist_km": np.float32(dist),
                "bucket_label": label,
                "speed_bucket": speed_bucket_for_obs_ms(spd),
                "region": region_for_point(lat, lon, regions),
                "cell_id": cell_id_for(
                    lat, lon, pd.Timestamp(rec["time"]), str(rec["instrument"])
                ),
            }
        )
    return pd.DataFrame(rows)


def collocate_with_forecasts(
    cells: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    pass_id: str,
    cfg: VerifyConfig,
) -> pd.DataFrame:
    """Join observation cells to model forecasts at matching lat/lon/time.

    ``forecasts`` columns: model, run_init, valid_time, lat, lon, model_u10, model_v10
    """
    if cells.empty or forecasts.empty:
        return pd.DataFrame()

    tol = pd.Timedelta(minutes=cfg.time_tolerance_minutes)
    out_rows: list[dict[str, Any]] = []
    for cell in cells.to_dict(orient="records"):
        obs_t = pd.Timestamp(cell["time"])
        # Nearest forecast in space+time per model×run_init
        dlat = np.abs(forecasts["lat"].to_numpy(dtype=float) - float(cell["lat"]))
        dlon = np.abs(forecasts["lon"].to_numpy(dtype=float) - float(cell["lon"]))
        near = forecasts[(dlat <= cfg.thin_grid_deg) & (dlon <= cfg.thin_grid_deg)].copy()
        if near.empty:
            continue
        near["dt"] = (pd.to_datetime(near["valid_time"]) - obs_t).abs()
        near = near[near["dt"] <= tol]
        if near.empty:
            continue
        for (model, run_init), grp in near.groupby(["model", "run_init"], sort=False):
            row = grp.sort_values("dt").iloc[0]
            lead_h = (
                pd.Timestamp(row["valid_time"]) - pd.Timestamp(run_init)
            ).total_seconds() / 3600.0
            bucket = lead_bucket_for_hours(lead_h)
            if bucket is None:
                continue
            obs_u = float(cell["obs_u10"])
            obs_v = float(cell["obs_v10"])
            if cfg.equivalent_neutral_correction:
                # Scatterometer EN wind is typically ~0.2 m/s above true 10 m.
                spd = np.hypot(obs_u, obs_v)
                if spd > 0:
                    scale = max(spd - cfg.equivalent_neutral_delta_ms, 0.0) / spd
                    obs_u *= scale
                    obs_v *= scale
            out_rows.append(
                {
                    "pass_id": pass_id,
                    "obs_class": cell["obs_class"],
                    "instrument": cell["instrument"],
                    "source_file_hash": cell.get("source_file_hash", ""),
                    "cell_id": cell["cell_id"],
                    "model": str(model),
                    "run_init": pd.Timestamp(run_init).to_datetime64(),
                    "valid_time": pd.Timestamp(cell["time"]).to_datetime64(),
                    "lead_hours": np.float32(lead_h),
                    "lat": np.float32(cell["lat"]),
                    "lon": np.float32(cell["lon"]),
                    "obs_u10": np.float32(obs_u),
                    "obs_v10": np.float32(obs_v),
                    "model_u10": np.float32(row["model_u10"]),
                    "model_v10": np.float32(row["model_v10"]),
                    "lead_bucket": bucket,
                    "speed_bucket": cell["speed_bucket"],
                    "region": cell["region"],
                    "bucket_label": cell["bucket_label"],
                    "land_dist_km": np.float32(cell["land_dist_km"]),
                }
            )
    return pd.DataFrame(out_rows)
