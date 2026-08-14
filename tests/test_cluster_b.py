from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from contracts.schemas import Route, load_course, load_routes  # noqa: E402
from pmc.follow import follow  # noqa: E402
from pmc.geo import crosses_land  # noqa: E402
from pmc.polar import load_polar  # noqa: E402


def test_gate_to_monaco_crosses_corsica() -> None:
    course = load_course(ROOT / "config/course.yaml")
    assert crosses_land(course.gate[0], course.gate[1], course.finish[0], course.finish[1])


def test_east_of_corsica_track_segments_do_not_cross_land() -> None:
    east_entry = (41.60, 9.85)
    via_east = (42.00, 9.95)
    via_north_east = (43.20, 9.70)
    north_clear = (43.35, 9.20)
    assert not crosses_land(east_entry[0], east_entry[1], via_east[0], via_east[1])
    assert not crosses_land(via_east[0], via_east[1], via_north_east[0], via_north_east[1])
    assert not crosses_land(via_north_east[0], via_north_east[1], north_clear[0], north_clear[1])


def test_follow_varies_with_start_time_when_wind_changes() -> None:
    route = Route(
        id="vary-test",
        label="vary-test",
        description="start-time sensitivity",
        legs=((38.2, 13.32), (39.2, 12.8)),
        tags=(),
    )
    time = np.array(
        [
            np.datetime64("2026-08-18T00:00:00"),
            np.datetime64("2026-08-18T01:00:00"),
            np.datetime64("2026-08-18T02:00:00"),
            np.datetime64("2026-08-18T03:00:00"),
        ]
    )
    lat = np.array([38.0, 39.4], dtype=np.float32)
    lon = np.array([12.6, 13.4], dtype=np.float32)
    # Alternating strong/weak headwind to force different elapsed outcomes.
    u10 = np.zeros((4, 2, 2), dtype=np.float32)
    v10 = np.array(
        [
            [[-9.0, -9.0], [-9.0, -9.0]],
            [[-1.0, -1.0], [-1.0, -1.0]],
            [[-9.0, -9.0], [-9.0, -9.0]],
            [[-1.0, -1.0], [-1.0, -1.0]],
        ],
        dtype=np.float32,
    )
    wind = xr.Dataset(
        data_vars={"u10": (("time", "lat", "lon"), u10), "v10": (("time", "lat", "lon"), v10)},
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={
            "source": "test",
            "fetched_utc": "2026-08-18T00:00:00Z",
            "api_version": "test",
            "omissions": [],
        },
    )
    polar = load_polar(ROOT / "contracts/fixtures/polar_52ft.pol")

    start0 = load_course(ROOT / "config/course.yaml").start_time_utc.replace(
        hour=0, minute=0
    )
    r0 = follow(route, wind, polar, start0)
    r1 = follow(route, wind, polar, start0 + timedelta(hours=1))
    assert abs(r0.elapsed_hours - r1.elapsed_hours) > 0.1


def test_follow_stall_detection_in_calm_field() -> None:
    route = Route(
        id="stall-test",
        label="stall-test",
        description="stall behavior check",
        legs=((38.2, 13.32), (38.7, 13.32)),
        tags=(),
    )
    time = np.array(
        [
            np.datetime64("2026-08-18T00:00:00"),
            np.datetime64("2026-08-18T01:00:00"),
        ]
    )
    lat = np.array([38.0, 38.8], dtype=np.float32)
    lon = np.array([13.0, 13.6], dtype=np.float32)
    calm = np.zeros((2, 2, 2), dtype=np.float32)
    wind = xr.Dataset(
        data_vars={"u10": (("time", "lat", "lon"), calm), "v10": (("time", "lat", "lon"), calm)},
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={
            "source": "test",
            "fetched_utc": "2026-08-18T00:00:00Z",
            "api_version": "test",
            "omissions": [],
        },
    )
    polar = load_polar(ROOT / "contracts/fixtures/polar_52ft.pol")
    result = follow(route, wind, polar, load_course(ROOT / "config/course.yaml").start_time_utc)
    assert result.stalled
    assert result.max_stall_hours > 6.0
    assert result.elapsed_hours >= 6.0
