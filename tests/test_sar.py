"""Acceptance tests for the Sentinel-1 SAR lee-shadow falsification module."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from contracts.schemas import (  # noqa: E402
    validate_sar_shadow_payload,
    validate_sar_store,
)
from pmc.sar.analyse import analyse_shadow_test  # noqa: E402
from pmc.sar.fetch import load_sar_config, open_sar_store  # noqa: E402
from scripts.build_sar_fixture import build_sar_fixture  # noqa: E402

FIXTURE = ROOT / "contracts" / "fixtures" / "sar_scenes_small.zarr"


@pytest.fixture(scope="module")
def sar_fixture() -> xr.Dataset:
    return open_sar_store(FIXTURE)


@pytest.fixture
def cfg() -> dict:
    c = load_sar_config()
    c = copy.deepcopy(c)
    c["bootstrap_samples"] = 400
    c["min_scenes_threshold"] = 10
    return c


def test_c9_fixture_validates(sar_fixture: xr.Dataset) -> None:
    validate_sar_store(sar_fixture)


def test_c9_rejects_direction_variables(sar_fixture: xr.Dataset) -> None:
    bad = sar_fixture.copy()
    bad["u10"] = bad["wind_speed_ms"] * 0.0
    with pytest.raises(ValueError, match="must not contain"):
        validate_sar_store(bad)


def test_scene_missing_band_is_excluded(sar_fixture: xr.Dataset, cfg: dict) -> None:
    """Acceptance 1: scene lacking either required band is discarded."""
    result = analyse_shadow_test(sar_fixture, cfg)
    # Fixture marks scenes 0 and 7 incomplete → 24 raw, 22 retained.
    assert result["n_scenes_raw"] == 24
    assert result["n_scenes_retained"] == 22
    retained_scenes = {row["time_utc"] for row in result["three_way_table"]}
    assert len(retained_scenes) == 22


def test_control_near_zero(sar_fixture: xr.Dataset, cfg: dict) -> None:
    """Acceptance 2 (happy path): control differential indistinguishable from zero."""
    result = analyse_shadow_test(sar_fixture, cfg)
    assert result["pipeline_valid"] is True
    assert result["control_indistinguishable_from_zero"] is True
    ctrl = result["control"]["paired_differential_kt"]
    assert ctrl["n"] > 0
    lo, hi = ctrl["ci95"]
    assert lo <= 0.0 <= hi


def test_control_failure_suppresses_sardinia(cfg: dict) -> None:
    """Acceptance 2 (failure path): non-zero control invalidates and suppresses."""
    biased = build_sar_fixture(control_bias_kt=4.0, n_scenes=20, include_incomplete=False)
    # Force a band differential in control by biasing inshore vs offshore via
    # virtual-coast geometry: rebuild with spatially structured control bias.
    # Simpler: mutate control-region speeds so 0–5 nm is systematically weaker.
    lat = biased["lat"].values
    lon = biased["lon"].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    dist_ctrl = (lon2d - 11.8) * 60.0 * np.cos(np.radians(lat2d))
    ctrl = (
        (lat2d >= 39.5)
        & (lat2d <= 41.0)
        & (lon2d >= 11.8)
        & (lon2d <= 13.2)
    )
    speed = biased["wind_speed_ms"].values.copy()
    for s in range(speed.shape[0]):
        inshore = ctrl & (dist_ctrl >= 0.5) & (dist_ctrl < 5.0)
        offshore = ctrl & (dist_ctrl >= 7.5) & (dist_ctrl < 10.0)
        speed[s] = np.where(inshore, speed[s] - 2.0, speed[s])  # m/s ≈ 4 kt
        speed[s] = np.where(offshore, speed[s] + 0.0, speed[s])
    biased["wind_speed_ms"].values[:] = speed

    cfg["min_scenes_threshold"] = 5
    result = analyse_shadow_test(biased, cfg)
    assert result["pipeline_valid"] is False
    assert result["verdict"] == "insufficient sample"
    assert result["paired_differentials_kt"]["sar"]["mean"] is None


def test_buffer_sensitivity_at_least_three(sar_fixture: xr.Dataset, cfg: dict) -> None:
    """Acceptance 3."""
    result = analyse_shadow_test(sar_fixture, cfg)
    assert len(result["buffer_sensitivity"]) >= 3
    assert {round(b["buffer_km"], 2) for b in result["buffer_sensitivity"]} >= {1.0, 1.5, 2.0}


def test_insufficient_sample_forces_verdict(sar_fixture: xr.Dataset, cfg: dict) -> None:
    """Acceptance 4: below threshold → insufficient sample, no mean plotted."""
    cfg["min_scenes_threshold"] = 1000
    result = analyse_shadow_test(sar_fixture, cfg)
    assert result["verdict"] == "insufficient sample"
    assert result["paired_differentials_kt"]["sar"]["mean"] is None
    assert result["paired_differentials_kt"]["sar"]["samples"] == []
    validate_sar_shadow_payload(result)


def test_supported_verdict_on_injected_lee(sar_fixture: xr.Dataset, cfg: dict) -> None:
    result = analyse_shadow_test(
        sar_fixture,
        cfg,
        precomputed_model_diffs={
            "arome": {s: -3.2 for s in range(24)},
            "era5": {s: -0.5 for s in range(24)},
        },
    )
    assert result["verdict"] == "supported"
    mean = result["paired_differentials_kt"]["sar"]["mean"]
    assert mean is not None
    assert -5.0 < mean < -2.0
