"""Run an end-to-end fixture-only pipeline and print stage pass/fail."""

from __future__ import annotations

import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from contracts.schemas import (  # noqa: E402
    haversine_nm,
    load_course,
    load_domain,
    load_routes,
    tws_twd_to_uv,
    uv_to_tws_twd,
    validate_climatology,
    validate_wind_store,
)
from pmc import follow as follow_mod  # noqa: E402
from pmc import geo as geo_mod  # noqa: E402
from pmc import io as io_mod  # noqa: E402
from pmc import polar as polar_mod  # noqa: E402
from pmc import report as report_mod  # noqa: E402
from pmc import stats as stats_mod  # noqa: E402


def run_stage(name: str, fn):
    try:
        result = fn()
        print(f"[PASS] {name}")
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {name}: {exc}")
        raise


def main() -> int:
    course = run_stage("load config/course.yaml", lambda: load_course(ROOT / "config/course.yaml"))
    domain = run_stage("load config/domain.yaml", lambda: load_domain(ROOT / "config/domain.yaml"))
    routes = run_stage("load config/routes.yaml", lambda: load_routes(ROOT / "config/routes.yaml"))

    def _assert_routes():
        for route in routes:
            route.assert_passes_gate(course.gate[0], course.gate[1], course.gate_tolerance_nm)
        return len(routes)

    n_routes = run_stage("assert all routes pass gate", _assert_routes)
    print(f"  routes checked: {n_routes}")

    wind_path = io_mod.fetch_wind(
        source="analysis",
        start=course.start_time_utc.date(),
        end=course.start_time_utc.date(),
        cfg=domain,
    )
    wind_path_abs = ROOT / wind_path
    wind = run_stage("open fixture wind zarr", lambda: xr.open_zarr(wind_path_abs, consolidated=True))
    run_stage("validate C1 wind store", lambda: validate_wind_store(wind))

    def _uv_round_trip():
        tws = np.array([4.0, 8.0, 12.0, 20.0])
        twd = np.array([0.0, 90.0, 225.0, 350.0])
        u, v = tws_twd_to_uv(tws, twd)
        tws2, twd2 = uv_to_tws_twd(u, v)
        if not np.allclose(tws, tws2, atol=1e-10):
            raise ValueError("TWS round trip mismatch.")
        if not np.allclose(((twd - twd2 + 180) % 360) - 180, 0.0, atol=1e-10):
            raise ValueError("TWD round trip mismatch.")

    run_stage("uv/twd round-trip", _uv_round_trip)

    def _geometry_checks():
        palermo = course.start
        monaco = course.finish
        nm = float(haversine_nm(palermo[0], palermo[1], monaco[0], monaco[1]))
        if nm < 380.0 or nm > 430.0:
            raise ValueError(f"Unexpected Palermo→Monaco distance: {nm:.2f} nm")
        if not geo_mod.crosses_land(course.gate[0], course.gate[1], monaco[0], monaco[1]):
            raise ValueError("Gate→Monaco line should cross land in fixture-grade mask.")

    run_stage("geometry sanity checks", _geometry_checks)

    polar = run_stage(
        "load fixture polar",
        lambda: polar_mod.load_polar(ROOT / "contracts/fixtures/polar_52ft.pol"),
    )
    run_stage("validate polar", lambda: polar.validate())

    def _benchmark_polar():
        rng = np.random.default_rng(42)
        twa = rng.uniform(-180.0, 180.0, size=1_000_000)
        tws = rng.uniform(1.0, 30.0, size=1_000_000)
        t0 = time.perf_counter()
        out = polar.speed(twa, tws)
        dt = time.perf_counter() - t0
        if np.any(out < 0.0):
            raise ValueError("Polar interpolation returned negative speed.")
        print(f"  polar.speed(1e6) runtime: {dt:.3f}s")
        return dt

    dt_interp = run_stage("benchmark polar interpolation", _benchmark_polar)
    if dt_interp > 1.0:
        print("[WARN] polar interpolation benchmark exceeded 1s target.")

    def _follow_all_routes() -> dict[str, pd.DataFrame]:
        starts = [course.start_time_utc + timedelta(days=i) for i in range(12)]
        result_frames: dict[str, pd.DataFrame] = {}
        for route in routes:
            rows = []
            for start in starts:
                rows.append(follow_mod.follow(route, wind, polar, start).as_row())
            result_frames[route.id] = pd.DataFrame(rows)
        return result_frames

    results = run_stage("run follower for 12 start days", _follow_all_routes)
    all_elapsed = np.concatenate(
        [frame["elapsed_hours"].dropna().to_numpy(dtype=float) for frame in results.values()]
    )
    median_elapsed = float(np.nanmedian(all_elapsed))
    print(f"  median elapsed hours: {median_elapsed:.2f}")
    if median_elapsed < 40.0 or median_elapsed > 110.0:
        print("[WARN] Median elapsed time outside 40-110h sanity band.")

    h2h = run_stage("compute head-to-head", lambda: stats_mod.head_to_head(results))
    climo = run_stage("compute climatology", lambda: stats_mod.climatology(wind, [8]))
    run_stage("validate climatology", lambda: validate_climatology(climo))

    def _emit():
        route_summaries = []
        for route in routes:
            samples = results[route.id]["elapsed_hours"].dropna().to_numpy(dtype=float)
            p10, p50, p90 = np.percentile(samples, [10, 50, 90])
            route_summaries.append(
                {
                    "id": route.id,
                    "label": route.label,
                    "legs": [[lat, lon] for lat, lon in route.legs],
                    "elapsed_hours": {
                        "p10": float(p10),
                        "p50": float(p50),
                        "p90": float(p90),
                        "samples": samples.tolist(),
                    },
                    "stall_rate": float(results[route.id]["stalled"].mean()),
                }
            )
        payload_path = ROOT / "dashboard" / "data.json"
        meta = {
            "generated_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "display_timezone": course.display_timezone,
            "course": {
                "start": [course.start[0], course.start[1]],
                "gate": [course.gate[0], course.gate[1]],
                "finish": [course.finish[0], course.finish[1]],
            },
            "polar_name": polar.name,
            "polar_is_validated": False,
            "warnings": [
                "Polar below 8 kt is unvalidated; use outputs for relative comparison.",
                "Follower and geometry are fixture-grade stubs in P0.",
            ],
        }
        skill_rows = [
            {
                "model": "ecmwf_ifs",
                "lead_days": 3,
                "wind_bin": "0-6kt",
                "vec_rmse_kt": 2.3,
                "speed_bias_kt": 0.5,
                "dir_mae_deg": 28.0,
                "reference_biased": True,
            },
            {
                "model": "gfs_global",
                "lead_days": 3,
                "wind_bin": "0-6kt",
                "vec_rmse_kt": 3.0,
                "speed_bias_kt": 0.6,
                "dir_mae_deg": 35.0,
                "reference_biased": False,
            },
        ]
        return report_mod.emit(
            climatology_ds=climo,
            routes_summary=route_summaries,
            head_to_head_df=h2h,
            skill_rows=skill_rows,
            meta=meta,
            output_path=payload_path,
        )

    output = run_stage("emit dashboard payload", _emit)
    print(f"  emitted: {output.relative_to(ROOT)}")

    stubs = [
        "pmc.io.fetch_wind (returns fixture path)",
        "pmc.geo.is_sea/crosses_land (fixture-grade rectangles)",
        "pmc.follow.follow (synthetic row)",
        "pmc.route.optimise (NotImplementedError)",
    ]
    print("[INFO] Remaining stubs:")
    for stub in stubs:
        print(f"  - {stub}")
    print("[PASS] walking skeleton completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        raise SystemExit(1)
