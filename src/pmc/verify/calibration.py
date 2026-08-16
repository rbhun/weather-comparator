"""Expedition calibration from scatterometer 48–72 h bucket only."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from contracts.schemas import angular_difference, uv_to_tws_twd

from .metrics import bootstrap_ci


def expedition_calibration_from_pairs(
    pairs: pd.DataFrame,
    *,
    n_boot: int = 500,
    seed: int = 20260816,
    min_n: int = 30,
) -> list[dict[str, Any]]:
    """Derive TWS scale (%) and TWD twist (°) per model × instrument.

    Expedition convention: 100% = unchanged TWS scale.
    Only scatterometer headline pairs in lead_bucket 48–72 are eligible.
    Instruments are never pooled (footprint sets different error floors).
    """
    if pairs is None or pairs.empty:
        return []

    forbidden = pairs["obs_class"].isin(["land_station", "sentinel1"])
    if forbidden.any():
        # Hard structural guard — callers must filter before invoke.
        raise AssertionError(
            "land_station/sentinel1 pairs must never reach expedition_calibration_from_pairs"
        )

    eligible = pairs[
        (pairs["obs_class"] == "scatterometer")
        & (pairs["lead_bucket"] == "48-72")
        & (pairs["bucket_label"] == "headline")
    ]
    rows: list[dict[str, Any]] = []
    for (model, instrument), grp in eligible.groupby(["model", "instrument"]):
        obs_u = grp["obs_u10"].to_numpy(dtype=float)
        obs_v = grp["obs_v10"].to_numpy(dtype=float)
        model_u = grp["model_u10"].to_numpy(dtype=float)
        model_v = grp["model_v10"].to_numpy(dtype=float)
        n = int(len(grp))
        obs_spd = np.hypot(obs_u, obs_v)
        model_spd = np.hypot(model_u, model_v)
        # TWS scale: mean(model/obs)*100, where obs>0
        mask = obs_spd > 0.5
        if np.any(mask):
            scale = float(np.mean(model_spd[mask] / obs_spd[mask]) * 100.0)
        else:
            scale = 100.0
        _, obs_dir = uv_to_tws_twd(obs_u, obs_v)
        _, model_dir = uv_to_tws_twd(model_u, model_v)
        twist = float(np.mean(angular_difference(model_dir, obs_dir))) if n else 0.0

        # Bootstrap CIs via resampling scale/twist
        rng = np.random.default_rng(seed)
        scales = np.empty(n_boot)
        twists = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, max(n, 1), size=max(n, 1))
            if n == 0:
                scales[i] = 100.0
                twists[i] = 0.0
                continue
            ou, ov = obs_u[idx], obs_v[idx]
            mu, mv = model_u[idx], model_v[idx]
            os_ = np.hypot(ou, ov)
            ms_ = np.hypot(mu, mv)
            msk = os_ > 0.5
            scales[i] = float(np.mean(ms_[msk] / os_[msk]) * 100.0) if np.any(msk) else 100.0
            _, od = uv_to_tws_twd(ou, ov)
            _, md = uv_to_tws_twd(mu, mv)
            twists[i] = float(np.mean(angular_difference(md, od)))
        rows.append(
            {
                "model": str(model),
                "instrument": str(instrument),
                "tws_scale_pct": round(scale, 2),
                "twd_twist_deg": round(twist, 2),
                "n": n,
                "tws_scale_ci95": [
                    round(float(np.nanpercentile(scales, 2.5)), 2),
                    round(float(np.nanpercentile(scales, 97.5)), 2),
                ],
                "twd_twist_ci95": [
                    round(float(np.nanpercentile(twists, 2.5)), 2),
                    round(float(np.nanpercentile(twists, 97.5)), 2),
                ],
                "source": f"scatterometer:{instrument}:48-72",
                "rankable": n >= min_n,
            }
        )
    return rows


def assert_no_land_wind_in_calibration(
    calibration_rows: list[dict[str, Any]],
    pairs: pd.DataFrame | None = None,
) -> None:
    """Acceptance guard: land-station wind cannot reach calibration output."""
    for row in calibration_rows:
        source = str(row.get("source", ""))
        if not source.startswith("scatterometer:"):
            raise AssertionError(f"Non-scatterometer calibration source: {source}")
        if "land" in source or "sentinel" in source or "metar" in source:
            raise AssertionError(f"Forbidden calibration source: {source}")
    if pairs is not None and not pairs.empty:
        if (pairs["obs_class"] != "scatterometer").any():
            raise AssertionError("Calibration pairs must be scatterometer-only")
