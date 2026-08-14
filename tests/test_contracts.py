from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from contracts.schemas import (  # noqa: E402
    Polar,
    Route,
    directional_constancy,
    haversine_nm,
    load_course,
    load_routes,
    tws_twd_to_uv,
    uv_to_tws_twd,
    validate_wind_store,
)
from pmc.polar import load_polar  # noqa: E402


def test_northerly_has_negative_v() -> None:
    u, v = tws_twd_to_uv(10.0, 0.0)
    assert v < 0.0
    assert abs(u) < 1e-12


def test_westerly_has_positive_u() -> None:
    u, v = tws_twd_to_uv(10.0, 270.0)
    assert u > 0.0
    assert abs(v) < 1e-12


def test_uv_round_trip_is_exact() -> None:
    tws = np.array([3.2, 7.5, 14.7, 22.1])
    twd = np.array([0.0, 44.0, 181.0, 359.0])
    u, v = tws_twd_to_uv(tws, twd)
    tws2, twd2 = uv_to_tws_twd(u, v)
    assert np.allclose(tws, tws2, atol=1e-10)
    delta = ((twd - twd2 + 180.0) % 360.0) - 180.0
    assert np.allclose(delta, 0.0, atol=1e-10)


def test_vector_average_wraparound_justification_for_uv_storage() -> None:
    # This is the key reason to store wind as u/v: directional wrap at 0/360.
    tws = np.array([10.0, 10.0])
    twd = np.array([350.0, 10.0])
    u, v = tws_twd_to_uv(tws, twd)
    mean_u = float(np.mean(u))
    mean_v = float(np.mean(v))
    _, mean_twd = uv_to_tws_twd(mean_u, mean_v)
    dist_to_north = min(abs(float(mean_twd)), abs(360.0 - float(mean_twd)))
    assert dist_to_north <= 1.0


def test_directional_constancy_bounds() -> None:
    steady_u = np.full((24, 10), 4.0)
    steady_v = np.full((24, 10), -2.0)
    random_u = np.random.default_rng(7).normal(0.0, 4.0, size=(24, 10))
    random_v = np.random.default_rng(8).normal(0.0, 4.0, size=(24, 10))
    const_steady = directional_constancy(steady_u, steady_v)
    const_random = directional_constancy(random_u, random_v)
    assert np.allclose(const_steady, 1.0, atol=1e-6)
    assert float(np.nanmean(const_random)) < 0.35


def test_palermo_monaco_distance_reasonable() -> None:
    nm = float(haversine_nm(38.20, 13.32, 43.73, 7.42))
    assert 380.0 <= nm <= 430.0


def test_every_configured_route_passes_gate() -> None:
    course = load_course(ROOT / "config" / "course.yaml")
    routes = load_routes(ROOT / "config/routes.yaml")
    for route in routes:
        route.assert_passes_gate(course.gate[0], course.gate[1], course.gate_tolerance_nm)


def test_parametric_routes_expand_to_expected_forks_and_offsets() -> None:
    routes = load_routes(ROOT / "config/routes.yaml")
    assert len(routes) == 8
    offsets = {
        next(tag for tag in route.tags if tag.startswith("leg1_offset_nm="))
        for route in routes
    }
    forks = {
        next(tag for tag in route.tags if tag.startswith("leg2_fork="))
        for route in routes
    }
    assert offsets == {
        "leg1_offset_nm=-40",
        "leg1_offset_nm=-20",
        "leg1_offset_nm=+0",
        "leg1_offset_nm=+20",
    }
    assert forks == {"leg2_fork=east_of_corsica", "leg2_fork=bonifacio_west_corsica"}


def test_route_missing_gate_raises() -> None:
    course = load_course(ROOT / "config" / "course.yaml")
    route = Route(
        id="bad",
        label="bad",
        description="missing gate",
        legs=((38.2, 13.32), (40.2, 12.0), (43.73, 7.42)),
    )
    try:
        route.assert_passes_gate(course.gate[0], course.gate[1], course.gate_tolerance_nm)
        raise AssertionError("Expected assert_passes_gate to raise for missing gate.")
    except ValueError:
        pass


def test_fixture_wind_validates_and_has_land_nan() -> None:
    ds = xr.open_zarr(ROOT / "contracts/fixtures/wind_small.zarr", consolidated=True)
    validate_wind_store(ds)
    u = ds["u10"].values
    v = ds["v10"].values
    assert np.any(np.isnan(u))
    assert np.array_equal(np.isnan(u), np.isnan(v))


def test_fixture_wind_has_real_diurnal_cycle() -> None:
    ds = xr.open_zarr(ROOT / "contracts/fixtures/wind_small.zarr", consolidated=True)
    tws = np.hypot(ds["u10"], ds["v10"]) * 1.9438445
    near_coast = tws.sel(lon=8.9, method="nearest")
    offshore = tws.sel(lon=12.8, method="nearest")
    amp_coast = float(near_coast.groupby("time.hour").mean().max() - near_coast.groupby("time.hour").mean().min())
    amp_offshore = float(offshore.groupby("time.hour").mean().max() - offshore.groupby("time.hour").mean().min())
    assert amp_coast > 1.2
    assert amp_coast > amp_offshore


def test_polar_clamps_instead_of_extrapolates() -> None:
    polar = load_polar(ROOT / "contracts/fixtures/polar_52ft.pol")
    at_min = polar.speed(60.0, float(polar.tws_kt.min()))
    below = polar.speed(60.0, float(polar.tws_kt.min() - 10.0))
    at_max = polar.speed(60.0, float(polar.tws_kt.max()))
    above = polar.speed(60.0, float(polar.tws_kt.max() + 10.0))
    assert np.allclose(at_min, below)
    assert np.allclose(at_max, above)


def test_polar_symmetric_about_zero_twa() -> None:
    polar = load_polar(ROOT / "contracts/fixtures/polar_52ft.pol")
    tws = np.array([6.0, 12.0, 18.0])
    plus = polar.speed(np.array([30.0, 75.0, 140.0]), tws)
    minus = polar.speed(np.array([-30.0, -75.0, -140.0]), tws)
    assert np.allclose(plus, minus)


def test_upwind_vmg_angle_in_expected_range() -> None:
    polar = load_polar(ROOT / "contracts/fixtures/polar_52ft.pol")
    angle, _ = polar.vmg_optimum(12.0, upwind=True)
    assert 35.0 <= angle <= 55.0
