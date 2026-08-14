"""A5 — Statistics module (real head-to-head, fixture-grade climatology)."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import xarray as xr

from contracts.schemas import MS_TO_KT, directional_constancy


def climatology(wind: xr.Dataset, months: list[int]) -> xr.Dataset:
    """Compute C6-style hourly climatology from a wind cube."""
    month_mask = wind["time"].dt.month.isin(months)
    selected = wind.sel(time=month_mask)
    tws = np.hypot(selected["u10"], selected["v10"]) * MS_TO_KT

    grouped_tws = tws.groupby("time.hour")

    n_samples = selected["u10"].groupby("time.hour").count(dim="time").astype(np.int32)
    mean_tws = grouped_tws.mean(dim="time", skipna=True).astype(np.float32)
    vector_mean_u = (
        selected["u10"].groupby("time.hour").mean(dim="time", skipna=True).astype(np.float32)
    )
    vector_mean_v = (
        selected["v10"].groupby("time.hour").mean(dim="time", skipna=True).astype(np.float32)
    )
    p_below_5 = ((tws < 5.0).groupby("time.hour").mean(dim="time", skipna=True)).astype(
        np.float32
    )
    p_below_8 = ((tws < 8.0).groupby("time.hour").mean(dim="time", skipna=True)).astype(
        np.float32
    )
    p_above_20 = (
        (tws > 20.0).groupby("time.hour").mean(dim="time", skipna=True)
    ).astype(np.float32)

    const_rows = []
    for hr in range(24):
        u_hr = selected["u10"].where(selected["time"].dt.hour == hr, drop=True).values
        v_hr = selected["v10"].where(selected["time"].dt.hour == hr, drop=True).values
        const_rows.append(directional_constancy(u_hr, v_hr))
    const = xr.DataArray(
        np.stack(const_rows, axis=0),
        coords={"hour": np.arange(24), "lat": selected["lat"], "lon": selected["lon"]},
        dims=("hour", "lat", "lon"),
    )

    climo = xr.Dataset(
        data_vars={
            "mean_tws_kt": mean_tws,
            "vector_mean_u": vector_mean_u,
            "vector_mean_v": vector_mean_v,
            "p_below_5kt": p_below_5,
            "p_below_8kt": p_below_8,
            "p_above_20kt": p_above_20,
            "directional_const": const.astype(np.float32),
            "n_samples": n_samples,
        },
        coords={"hour": mean_tws["hour"], "lat": wind["lat"], "lon": wind["lon"]},
        attrs={
            "source": wind.attrs.get("source", "unknown"),
            "years_used": sorted(set(selected["time"].dt.year.values.tolist())),
            "months_used": sorted(set(months)),
        },
    )

    low_samples = climo["n_samples"] < 200
    for var in (
        "mean_tws_kt",
        "vector_mean_u",
        "vector_mean_v",
        "p_below_5kt",
        "p_below_8kt",
        "p_above_20kt",
        "directional_const",
    ):
        climo[var] = climo[var].where(~low_samples)
    return climo


def head_to_head(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute pairwise win-rate and margin percentiles on matched starts."""
    rows: list[dict[str, object]] = []
    for a, b in combinations(sorted(results.keys()), 2):
        left = results[a][["start_time", "elapsed_hours"]].rename(
            columns={"elapsed_hours": "elapsed_a"}
        )
        right = results[b][["start_time", "elapsed_hours"]].rename(
            columns={"elapsed_hours": "elapsed_b"}
        )
        merged = left.merge(right, on="start_time", how="inner").dropna()
        if merged.empty:
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "a_wins_pct": np.nan,
                    "median_margin_hours": np.nan,
                    "p10_margin_hours": np.nan,
                    "p90_margin_hours": np.nan,
                    "n": 0,
                }
            )
            continue

        margins = merged["elapsed_b"] - merged["elapsed_a"]  # positive => A faster
        rows.append(
            {
                "a": a,
                "b": b,
                "a_wins_pct": float(np.round((margins > 0.0).mean() * 100.0, 2)),
                "median_margin_hours": float(np.round(np.median(margins), 2)),
                "p10_margin_hours": float(np.round(np.percentile(margins, 10), 2)),
                "p90_margin_hours": float(np.round(np.percentile(margins, 90), 2)),
                "n": int(margins.size),
            }
        )
    return pd.DataFrame(rows)
