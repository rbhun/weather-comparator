"""Land-station MSLP scoring and thermal/gradient timing diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from contracts.schemas import MS_TO_KT, uv_to_tws_twd

from .conventions import lead_bucket_for_hours


def load_stations(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(raw.get("stations") or [])


def score_mslp(
    obs: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    min_n: int = 5,
) -> list[dict[str, Any]]:
    """Bias and RMSE of MSLP per model × station × lead bucket.

    ``obs``: station_id, time, mslp_hpa
    ``forecasts``: station_id, model, run_init, valid_time, mslp_hpa
    """
    if obs.empty or forecasts.empty:
        return []
    rows: list[dict[str, Any]] = []
    for station_id, ogrp in obs.groupby("station_id"):
        for model, fgrp in forecasts.groupby("model"):
            merged = []
            for _, o in ogrp.iterrows():
                dt = (pd.to_datetime(fgrp["valid_time"]) - pd.Timestamp(o["time"])).abs()
                near = fgrp[dt <= pd.Timedelta(minutes=30)]
                if near.empty:
                    continue
                best = near.iloc[dt[dt <= pd.Timedelta(minutes=30)].argmin()]
                lead_h = (
                    pd.Timestamp(best["valid_time"]) - pd.Timestamp(best["run_init"])
                ).total_seconds() / 3600.0
                bucket = lead_bucket_for_hours(lead_h)
                if bucket is None:
                    continue
                merged.append(
                    {
                        "lead_bucket": bucket,
                        "bias": float(best["mslp_hpa"] - o["mslp_hpa"]),
                        "err2": float(best["mslp_hpa"] - o["mslp_hpa"]) ** 2,
                    }
                )
            if not merged:
                continue
            mdf = pd.DataFrame(merged)
            for bucket, bgrp in mdf.groupby("lead_bucket"):
                n = int(len(bgrp))
                rows.append(
                    {
                        "station_id": str(station_id),
                        "model": str(model),
                        "lead_bucket": str(bucket),
                        "bias_hpa": round(float(bgrp["bias"].mean()), 2),
                        "rmse_hpa": round(float(np.sqrt(bgrp["err2"].mean())), 2),
                        "n": n,
                        "rankable": n >= min_n,
                    }
                )
    return rows


def detect_onset_time(
    times: pd.DatetimeIndex | np.ndarray,
    u10: np.ndarray,
    v10: np.ndarray,
    *,
    speed_kt: float,
    dir_min: float,
    dir_max: float,
) -> pd.Timestamp | None:
    """First time speed crosses threshold with direction in sector (meteorological FROM)."""
    tws_kt, twd = uv_to_tws_twd(u10, v10)
    times_arr = pd.to_datetime(times)
    for t, spd, direction in zip(times_arr, tws_kt, twd):
        if spd < speed_kt:
            continue
        if dir_min <= dir_max:
            in_sector = dir_min <= direction <= dir_max
        else:
            # wrap e.g. 300..30
            in_sector = direction >= dir_min or direction <= dir_max
        if in_sector:
            return pd.Timestamp(t)
    return None


def onset_lags(
    obs_series: pd.DataFrame,
    model_series: pd.DataFrame,
    *,
    speed_kt: float,
    dir_min: float,
    dir_max: float,
    kind: str = "thermal",
) -> list[dict[str, Any]]:
    """Model-minus-observed onset lag in minutes per station per day."""
    rows: list[dict[str, Any]] = []
    if obs_series.empty:
        return rows
    obs_series = obs_series.copy()
    obs_series["day"] = pd.to_datetime(obs_series["time"]).dt.floor("D")
    for (station_id, day), ogrp in obs_series.groupby(["station_id", "day"]):
        ogrp = ogrp.sort_values("time")
        obs_onset = detect_onset_time(
            ogrp["time"].to_numpy(),
            ogrp["obs_u10"].to_numpy(dtype=float),
            ogrp["obs_v10"].to_numpy(dtype=float),
            speed_kt=speed_kt,
            dir_min=dir_min,
            dir_max=dir_max,
        )
        if obs_onset is None:
            continue
        day_models = model_series[
            (model_series["station_id"] == station_id)
            & (pd.to_datetime(model_series["valid_time"]).dt.floor("D") == day)
        ]
        for model, mgrp in day_models.groupby("model"):
            mgrp = mgrp.sort_values("valid_time")
            model_onset = detect_onset_time(
                mgrp["valid_time"].to_numpy(),
                mgrp["model_u10"].to_numpy(dtype=float),
                mgrp["model_v10"].to_numpy(dtype=float),
                speed_kt=speed_kt,
                dir_min=dir_min,
                dir_max=dir_max,
            )
            if model_onset is None:
                continue
            lag = int((model_onset - obs_onset).total_seconds() / 60.0)
            rows.append(
                {
                    "station_id": str(station_id),
                    "day_utc": pd.Timestamp(day).strftime("%Y-%m-%d"),
                    "model": str(model),
                    "obs_onset_utc": obs_onset.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "model_onset_utc": model_onset.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "lag_minutes": lag,
                    "kind": kind,
                }
            )
    return rows
