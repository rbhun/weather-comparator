"""Tests for YB-track vs analysis-wind residual check."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from contracts.schemas import Polar  # noqa: E402
from pmc.io.yb import Boat  # noqa: E402
from pmc.polar import load_polar  # noqa: E402
from pmc.stats.yb_wind_check import (  # noqa: E402
    WindCheckConfig,
    annotate_samples_with_wind,
    assign_region,
    format_summary_markdown,
    normalise_residuals_per_boat,
    run_wind_check,
    summarise_residuals,
    track_motion_samples,
)


def _toy_polar() -> Polar:
    return load_polar(ROOT / "contracts" / "fixtures" / "polar_52ft.pol")


def _toy_boat() -> Boat:
    # Northbound track at ~6 kt over 2 hours, 20-min fixes.
    start = datetime(2024, 8, 18, 12, 0, tzinfo=timezone.utc)
    times: list[str] = []
    lats: list[float] = []
    lons: list[float] = []
    lat = 41.0
    lon = 10.0
    for i in range(7):
        t = start.timestamp() + i * 20 * 60
        times.append(datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        lats.append(lat)
        lons.append(lon)
        # 6 kt north ≈ 0.0333 deg lat per 20 min (6 nm/h * 1/3 h / 60)
        lat += 6.0 * (20.0 / 60.0) / 60.0
    return Boat(
        year=2024,
        name="TEST BOAT",
        status="FINISHED",
        finished=True,
        absolute_rank=1,
        elapsed_s=7200,
        elapsed_label="2h 00m",
        start_utc=times[0],
        finish_utc=times[-1],
        lon=lons,
        lat=lats,
        times=times,
    )


def _toy_wind(*, tws_kt: float = 12.0, twd_deg: float = 90.0) -> xr.Dataset:
    """Uniform wind field covering the toy track."""

    from contracts.schemas import tws_twd_to_uv

    times = pd.date_range("2024-08-18", periods=48, freq="h", tz="UTC").tz_localize(None)
    lats = np.arange(40.5, 42.0, 0.1)
    lons = np.arange(9.5, 11.0, 0.1)
    u, v = tws_twd_to_uv(tws_kt, twd_deg)
    u10 = np.full((times.size, lats.size, lons.size), float(u), dtype=np.float32)
    v10 = np.full((times.size, lats.size, lons.size), float(v), dtype=np.float32)
    return xr.Dataset(
        data_vars={
            "u10": (("time", "lat", "lon"), u10),
            "v10": (("time", "lat", "lon"), v10),
        },
        coords={"time": times, "lat": lats.astype(np.float32), "lon": lons.astype(np.float32)},
        attrs={"source": "toy"},
    )


def test_track_motion_samples_sog_cog() -> None:
    boat = _toy_boat()
    samples = track_motion_samples(boat, cfg=WindCheckConfig(interval_min=20))
    assert len(samples) >= 5
    assert samples["sog_kt"].median() == pytest.approx(6.0, abs=0.3)
    assert samples["cog_deg"].median() == pytest.approx(0.0, abs=2.0)


def test_positive_residual_when_analysis_too_light() -> None:
    """Fleet SOG ~6 kt; analysis 4 kt beam reach → polar predicts slower → +residual."""

    boat = _toy_boat()
    # Beam wind at 4 kt: polar beam ~3.2 kt predicted vs 6 kt SOG.
    wind = _toy_wind(tws_kt=4.0, twd_deg=90.0)
    polar = _toy_polar()
    samples, summaries = run_wind_check([boat], wind, polar, cfg=WindCheckConfig(interval_min=20))
    # Skip offshore distance (coastline) by injecting a constant for summarise path
    samples = samples.copy()
    samples["distance_offshore_nm"] = 15.0
    samples["offshore_bin"] = "10-20nm"
    samples["near_coast"] = False
    samples["tws_bin"] = pd.cut(
        samples["analysis_tws_kt"],
        bins=[0, 6, 12, 20, np.inf],
        labels=["0-6kt", "6-12kt", "12-20kt", "20kt+"],
        right=False,
    )
    assert samples["residual_kt"].median() > 1.0
    by_tws = summarise_residuals(samples)["by_tws"]
    assert not by_tws.empty
    assert int(by_tws["n"].sum()) == len(samples.dropna(subset=["residual_kt"]))


def test_annotate_sets_twa_from_cog_and_twd() -> None:
    boat = _toy_boat()
    motion = track_motion_samples(boat)
    # Westerly wind, northbound track → TWA ~90
    wind = _toy_wind(tws_kt=10.0, twd_deg=270.0)
    polar = _toy_polar()
    # Avoid coastline dependency in this unit test
    annotated = annotate_samples_with_wind(
        motion, wind, polar, compute_offshore=False
    )
    annotated["distance_offshore_nm"] = 12.0
    assert annotated["analysis_tws_kt"].median() == pytest.approx(10.0, abs=0.2)
    assert abs(float(annotated["twa_deg"].median())) == pytest.approx(90.0, abs=5.0)


def test_per_boat_normalisation_zeros_boat_median() -> None:
    frame = pd.DataFrame(
        {
            "year": [2024, 2024, 2025, 2025],
            "boat": ["A", "A", "A", "A"],
            "residual_kt": [1.0, 3.0, 10.0, 14.0],
        }
    )
    out = normalise_residuals_per_boat(frame)
    assert out.groupby(["year", "boat"])["residual_norm_kt"].median().abs().max() < 1e-9


def test_assign_region_boxes() -> None:
    assert assign_region(41.13, 9.8) == "sardinia_east"
    assert assign_region(38.8, 12.0) == "tyrrhenian"
    assert assign_region(43.5, 8.0) == "ligurian"

    frame = pd.DataFrame(
        {
            "offshore_bin": ["0-5nm", "0-5nm", "40nm+"],
            "hour_local": [14, 14, 2],
            "tws_bin": ["6-12kt", "6-12kt", "0-6kt"],
            "residual_kt": [2.0, 3.0, -0.5],
            "sog_kt": [8.0, 9.0, 4.0],
            "predicted_sog_kt": [6.0, 6.0, 4.5],
            "analysis_tws_kt": [8.0, 8.0, 4.0],
            "near_coast": [True, True, False],
            "afternoon_local": [True, True, False],
        }
    )
    summaries = summarise_residuals(frame)
    text = format_summary_markdown(summaries, meta={"n_samples": 3})
    assert "residual_median_kt" in text or "By distance offshore" in text
    assert "n=" in text or "| n |" in text or "n " in text
    assert not summaries["by_offshore"].empty
    assert int(summaries["by_offshore"].loc[
        summaries["by_offshore"]["offshore_bin"].astype(str) == "0-5nm", "n"
    ].iloc[0]) == 2
