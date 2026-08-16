"""Generate C9 verify fixtures (instruments + current_weather payload)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from contracts.schemas import (  # noqa: E402
    INSTRUMENT_DIR_CONVENTION,
    MS_TO_KT,
    tws_twd_to_uv,
    validate_current_weather,
)
from pmc.verify.conventions import wind_to_uv_ms  # noqa: E402
from pmc.verify.pipeline import build_current_weather_payload  # noqa: E402
from pmc.verify.sentinel import synthetic_sentinel_swath  # noqa: E402
from pmc.verify.store import VerifyConfig, VerifyStore  # noqa: E402
from pmc.verify.collocate import assign_buckets, collocate_with_forecasts, thin_to_grid  # noqa: E402
from pmc.verify.ingest import ingest_observations  # noqa: E402

OUT = ROOT / "contracts" / "fixtures" / "verify"
INSTR = OUT / "instruments"


def _mistral_cell() -> dict:
    """Strong NW mistral outflow — unambiguous synoptic situation."""
    # Meteorological FROM = 315° (NW), speed 18 m/s
    return {
        "lat": 41.5,
        "lon": 8.0,
        "time": "2026-08-10T12:00:00Z",
        "wind_speed_ms": 18.0,
        "wind_dir_meteo_from": 315.0,
    }


def write_instrument_fixtures() -> None:
    INSTR.mkdir(parents=True, exist_ok=True)
    cell = _mistral_cell()
    u_ref, v_ref = tws_twd_to_uv(cell["wind_speed_ms"] * MS_TO_KT, cell["wind_dir_meteo_from"])
    ref_u = float(u_ref)
    ref_v = float(v_ref)

    for instrument, convention in INSTRUMENT_DIR_CONVENTION.items():
        if convention == "meteorological":
            product_dir = cell["wind_dir_meteo_from"]
        else:
            # oceanographic toward = from + 180
            product_dir = (cell["wind_dir_meteo_from"] + 180.0) % 360.0

        ds = xr.Dataset(
            data_vars={
                "lat": ("cell", np.array([cell["lat"]], dtype=np.float32)),
                "lon": ("cell", np.array([cell["lon"]], dtype=np.float32)),
                "time": (
                    "cell",
                    np.array([np.datetime64(cell["time"][:-1])], dtype="datetime64[ns]"),
                ),
                "wind_speed": ("cell", np.array([cell["wind_speed_ms"]], dtype=np.float32)),
                "wind_dir": ("cell", np.array([product_dir], dtype=np.float32)),
                "quality_flag": ("cell", np.array([0], dtype=np.int8)),
                "land_flag": ("cell", np.array([0], dtype=np.int8)),
                "ice_flag": ("cell", np.array([0], dtype=np.int8)),
                "rain_flag": ("cell", np.array([0], dtype=np.int8)),
            },
            attrs={
                "instrument": instrument,
                "direction_convention": convention,
                "reference_u10_ms": ref_u,
                "reference_v10_ms": ref_v,
                "synoptic": "mistral_outflow_nw",
            },
        )
        path = INSTR / f"{instrument}_mistral.nc"
        ds.to_netcdf(path)

        # Sidecar JSON with independently computed reference for unit tests
        side = {
            "instrument": instrument,
            "convention": convention,
            "product_wind_dir_deg": product_dir,
            "wind_speed_ms": cell["wind_speed_ms"],
            "reference_u10_ms": ref_u,
            "reference_v10_ms": ref_v,
            "lat": cell["lat"],
            "lon": cell["lon"],
            "time": cell["time"],
            "note": "Independent reference: meteo FROM 315° at 18 m/s (mistral).",
        }
        (INSTR / f"{instrument}_mistral.ref.json").write_text(
            json.dumps(side, indent=2) + "\n", encoding="utf-8"
        )

        # Sanity: ingest must recover reference
        got_u, got_v = wind_to_uv_ms(
            cell["wind_speed_ms"], product_dir, instrument=instrument
        )
        if abs(float(got_u) - ref_u) > 1e-6 or abs(float(got_v) - ref_v) > 1e-6:
            raise AssertionError(f"Fixture self-check failed for {instrument}")


def write_synthetic_pass_fixture() -> None:
    """Build a store + current_weather.json with known constant model offset."""
    store_dir = OUT / "store"
    if store_dir.exists():
        for p in store_dir.glob("*"):
            p.unlink()
    store_dir.mkdir(parents=True, exist_ok=True)

    cfg = VerifyConfig(
        store_dir=store_dir,
        regions={
            "tyrrhenian_open": {"lat": [38.5, 41.0], "lon": [10.5, 14.0]},
            "sardinia_east_corridor": {"lat": [39.0, 41.2], "lon": [9.6, 10.8]},
            "bonifacio_approach": {"lat": [40.8, 41.6], "lon": [8.5, 9.8]},
            "ligurian": {"lat": [42.5, 44.0], "lon": [6.5, 10.0]},
        },
        min_rank_n=30,
        bootstrap_samples=200,
    )

    # 40 offshore cells on distinct 0.25° grid cells (open Tyrrhenian / west Corsica)
    n = 40
    # Grid so thinning keeps all 40
    coords = []
    for i in range(n):
        lat = 40.0 + 0.25 * (i % 8)
        lon = 11.0 + 0.25 * (i // 8)
        coords.append((lat, lon))
    lats = np.array([c[0] for c in coords], dtype=float)
    lons = np.array([c[1] for c in coords], dtype=float)
    speed = np.full(n, 12.0)  # m/s
    # meteo from 315
    u_obs, v_obs = tws_twd_to_uv(speed * MS_TO_KT, np.full(n, 315.0))
    times = pd.to_datetime(["2026-08-10T12:00:00Z"] * n).tz_localize(None)

    cells = pd.DataFrame(
        {
            "lat": lats.astype(np.float32),
            "lon": lons.astype(np.float32),
            "time": times,
            "obs_u10": u_obs.astype(np.float32),
            "obs_v10": v_obs.astype(np.float32),
            "obs_speed_ms": speed.astype(np.float32),
            "instrument": "ascat_metop_b",
            "obs_class": "scatterometer",
            "source_file_hash": "fixturehash001",
        }
    )
    # Constant model offset: +1.5 m/s speed along same direction, +8° twist
    # Scale speed by (12+1.5)/12 and rotate direction by +8°
    model_speed = speed + 1.5
    model_dir = (315.0 + 8.0) % 360.0
    u_mod, v_mod = tws_twd_to_uv(model_speed * MS_TO_KT, np.full(n, model_dir))

    run_init = pd.Timestamp("2026-08-08T00:00:00")
    valid = pd.Timestamp("2026-08-10T12:00:00")  # lead = 60 h → 48-72
    forecasts = pd.DataFrame(
        {
            "model": ["gfs_global"] * n,
            "run_init": [run_init] * n,
            "valid_time": [valid] * n,
            "lat": lats.astype(np.float32),
            "lon": lons.astype(np.float32),
            "model_u10": u_mod.astype(np.float32),
            "model_v10": v_mod.astype(np.float32),
        }
    )
    # Also add a low-n instrument pass for rankable=false demo
    small_n = 10
    cells_small = cells.iloc[:small_n].copy()
    cells_small["instrument"] = "hscat_hy2b"
    cells_small["source_file_hash"] = "fixturehash002"
    forecasts_small = forecasts.iloc[:small_n].copy()
    forecasts_small["model"] = "ecmwf_ifs"

    from pmc.verify.pipeline import run_verification_pass

    # Write temp netcdf via JSON ingest path instead
    obs_json = OUT / "synthetic_ascat.json"
    obs_json.write_text(
        json.dumps(
            {
                "obs_class": "scatterometer",
                "instrument": "ascat_metop_b",
                "cells": [
                    {
                        "lat": float(lat),
                        "lon": float(lon),
                        "time": "2026-08-10T12:00:00Z",
                        "wind_speed": float(spd),
                        "wind_dir": 315.0,
                        "land_flag": 0,
                        "ice_flag": 0,
                        "rain_flag": 0,
                        "quality_flag": 0,
                    }
                    for lat, lon, spd in zip(lats, lons, speed)
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    obs_json_small = OUT / "synthetic_hscat_lown.json"
    obs_json_small.write_text(
        json.dumps(
            {
                "obs_class": "scatterometer",
                "instrument": "hscat_hy2b",
                "cells": [
                    {
                        "lat": float(lat),
                        "lon": float(lon),
                        "time": "2026-08-10T12:00:00Z",
                        "wind_speed": float(spd),
                        "wind_dir": 135.0,  # oceanographic toward = 315 from
                        "land_flag": 0,
                        "ice_flag": 0,
                        "rain_flag": 0,
                        "quality_flag": 0,
                    }
                    for lat, lon, spd in zip(lats[:small_n], lons[:small_n], speed[:small_n])
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Combine forecasts for both models
    forecasts_all = pd.concat(
        [
            forecasts,
            forecasts.assign(model="ecmwf_ifs"),
            forecasts_small,
        ],
        ignore_index=True,
    )

    summary = run_verification_pass(
        "pass-2026-08-10T12:00:00Z",
        cfg,
        observation_files=[obs_json, obs_json_small],
        forecasts=forecasts_all,
    )
    print("pass:", summary)

    # Land station fixture (timing + MSLP) — not for calibration
    land_json = OUT / "synthetic_land.json"
    land_json.write_text(
        json.dumps(
            {
                "obs_class": "land_station",
                "cells": [
                    {
                        "station_id": "LIET",
                        "lat": 39.919,
                        "lon": 9.683,
                        "time": "2026-08-10T10:00:00Z",
                        "mslp_hpa": 1014.0,
                        "obs_u10": 0.0,
                        "obs_v10": -3.0,
                    },
                    {
                        "station_id": "LFKF",
                        "lat": 41.502,
                        "lon": 9.098,
                        "time": "2026-08-10T10:00:00Z",
                        "mslp_hpa": 1012.5,
                        "obs_u10": 2.0,
                        "obs_v10": -4.0,
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    station_obs = pd.DataFrame(
        [
            {
                "station_id": "LIET",
                "time": pd.Timestamp("2026-08-10T10:00:00"),
                "mslp_hpa": 1014.0,
                "obs_u10": 0.0,
                "obs_v10": -3.0,
            },
            {
                "station_id": "LFKF",
                "time": pd.Timestamp("2026-08-10T10:00:00"),
                "mslp_hpa": 1012.5,
                "obs_u10": 2.0,
                "obs_v10": -4.0,
            },
        ]
    )
    station_forecasts = pd.DataFrame(
        [
            {
                "station_id": "LIET",
                "model": "gfs_global",
                "run_init": pd.Timestamp("2026-08-08T00:00:00"),
                "valid_time": pd.Timestamp("2026-08-10T10:00:00"),
                "mslp_hpa": 1015.0,
            },
            {
                "station_id": "LFKF",
                "model": "gfs_global",
                "run_init": pd.Timestamp("2026-08-08T00:00:00"),
                "valid_time": pd.Timestamp("2026-08-10T10:00:00"),
                "mslp_hpa": 1013.0,
            },
        ]
    )

    payload = build_current_weather_payload(
        store_dir,
        cfg=cfg,
        stations_yaml=ROOT / "config/stations.yaml",
        station_obs=station_obs,
        station_forecasts=station_forecasts,
        sentinel=synthetic_sentinel_swath(),
        generated_utc="2026-08-10T18:00:00Z",
    )
    validate_current_weather(payload)
    out_path = OUT / "current_weather.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")

    # Known offset notes for tests
    (OUT / "EXPECTED_OFFSET.json").write_text(
        json.dumps(
            {
                "speed_bias_ms": 1.5,
                "twd_twist_deg": 8.0,
                "tws_scale_pct": round((13.5 / 12.0) * 100.0, 2),
                "model": "gfs_global",
                "instrument": "ascat_metop_b",
                "lead_bucket": "48-72",
                "n": 40,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    write_instrument_fixtures()
    write_synthetic_pass_fixture()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
