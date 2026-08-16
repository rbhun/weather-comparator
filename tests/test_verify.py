"""Acceptance tests for C9 live observation verification."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from contracts.schemas import (  # noqa: E402
    INSTRUMENT_DIR_CONVENTION,
    validate_current_weather,
)
from pmc.verify.calibration import (  # noqa: E402
    assert_no_land_wind_in_calibration,
    expedition_calibration_from_pairs,
)
from pmc.verify.conventions import wind_to_uv_ms  # noqa: E402
from pmc.verify.ingest import detect_format, ingest_observations  # noqa: E402
from pmc.verify.pipeline import build_current_weather_payload, run_verification_pass  # noqa: E402
from pmc.verify.store import VerifyConfig, VerifyStore  # noqa: E402

FIXTURES = ROOT / "contracts" / "fixtures" / "verify"
INSTR = FIXTURES / "instruments"


@pytest.mark.parametrize("instrument", sorted(INSTRUMENT_DIR_CONVENTION))
def test_per_instrument_direction_convention_matches_independent_reference(instrument: str) -> None:
    nc = INSTR / f"{instrument}_mistral.nc"
    ref = json.loads((INSTR / f"{instrument}_mistral.ref.json").read_text(encoding="utf-8"))
    assert nc.exists(), f"Missing fixture {nc}"
    report = ingest_observations(nc)
    assert report.n_cells == 1
    assert report.instrument == instrument
    row = report.cells.iloc[0]
    assert abs(float(row["obs_u10"]) - float(ref["reference_u10_ms"])) < 1e-5
    assert abs(float(row["obs_v10"]) - float(ref["reference_v10_ms"])) < 1e-5

    # Also assert direct converter against product wind_dir from the NetCDF
    ds = xr.open_dataset(nc)
    try:
        u, v = wind_to_uv_ms(
            float(ds["wind_speed"].values[0]),
            float(ds["wind_dir"].values[0]),
            instrument=instrument,
        )
    finally:
        ds.close()
    assert abs(float(u) - float(ref["reference_u10_ms"])) < 1e-5
    assert abs(float(v) - float(ref["reference_v10_ms"])) < 1e-5


def test_unknown_instrument_refuses_to_guess() -> None:
    with pytest.raises(ValueError, match="Refusing to guess"):
        wind_to_uv_ms(10.0, 180.0, instrument="not_a_real_sensor")


def test_reingest_is_idempotent(tmp_path: Path) -> None:
    store_dir = tmp_path / "verify"
    cfg = VerifyConfig(store_dir=store_dir, bootstrap_samples=50, min_rank_n=30)
    obs = FIXTURES / "synthetic_ascat.json"
    # Build matching forecasts from the obs cells
    report = ingest_observations(obs)
    cells = report.cells
    run_init = pd.Timestamp("2026-08-08T00:00:00")
    valid = pd.Timestamp("2026-08-10T12:00:00")
    forecasts = pd.DataFrame(
        {
            "model": ["gfs_global"] * len(cells),
            "run_init": [run_init] * len(cells),
            "valid_time": [valid] * len(cells),
            "lat": cells["lat"].to_numpy(),
            "lon": cells["lon"].to_numpy(),
            "model_u10": cells["obs_u10"].to_numpy() + 0.1,
            "model_v10": cells["obs_v10"].to_numpy(),
        }
    )
    s1 = run_verification_pass(
        "pass-idem-1",
        cfg,
        observation_files=[obs],
        forecasts=forecasts,
    )
    assert s1.scatterometer_new_rows > 0
    store = VerifyStore(store_dir)
    n_after_first = len(store.load_collocated())
    s2 = run_verification_pass(
        "pass-idem-1",
        cfg,
        observation_files=[obs],
        forecasts=forecasts,
    )
    assert s2.scatterometer_new_rows == 0
    assert len(store.load_collocated()) == n_after_first


def test_synthetic_constant_offset_recovers_bias_and_twist() -> None:
    expected = json.loads((FIXTURES / "EXPECTED_OFFSET.json").read_text(encoding="utf-8"))
    payload = json.loads((FIXTURES / "current_weather.json").read_text(encoding="utf-8"))
    validate_current_weather(payload)

    rows = [
        r
        for r in payload["scorecard"]
        if r["instrument"] == expected["instrument"]
        and r["model"] == expected["model"]
        and r["lead_bucket"] == "48-72"
        and r["region"] == "all"
    ]
    assert rows, "Expected aggregate scorecard row missing"
    row = rows[0]
    assert abs(row["speed_bias_ms"] - expected["speed_bias_ms"]) < 1e-6
    assert row["n"] >= 30
    assert row["rankable"] is True

    cal = [
        c
        for c in payload["expedition_calibration"]
        if c["model"] == expected["model"] and c.get("instrument") == expected["instrument"]
    ]
    assert cal
    assert abs(cal[0]["twd_twist_deg"] - expected["twd_twist_deg"]) < 1e-6
    assert abs(cal[0]["tws_scale_pct"] - expected["tws_scale_pct"]) < 1e-6
    assert cal[0]["source"].startswith("scatterometer:")


def test_low_n_scorecard_refuses_to_rank() -> None:
    payload = json.loads((FIXTURES / "current_weather.json").read_text(encoding="utf-8"))
    low = [
        r
        for r in payload["scorecard"]
        if r["instrument"] == "hscat_hy2b" and r["region"] == "all"
    ]
    assert low
    for row in low:
        assert row["n"] < payload["meta"]["min_rank_n"]
        assert row["rankable"] is False


def test_land_station_wind_cannot_reach_expedition_calibration() -> None:
    # Construct land-station pairs and assert calibration rejects them
    land = pd.DataFrame(
        {
            "obs_class": ["land_station"],
            "instrument": ["metar_synop"],
            "lead_bucket": ["48-72"],
            "bucket_label": ["headline"],
            "model": ["gfs_global"],
            "obs_u10": [1.0],
            "obs_v10": [0.0],
            "model_u10": [2.0],
            "model_v10": [0.0],
        }
    )
    with pytest.raises(AssertionError, match="land_station"):
        expedition_calibration_from_pairs(land)

    # Mixed frame also fails
    mixed = pd.concat(
        [
            land,
            pd.DataFrame(
                {
                    "obs_class": ["scatterometer"],
                    "instrument": ["ascat_metop_b"],
                    "lead_bucket": ["48-72"],
                    "bucket_label": ["headline"],
                    "model": ["gfs_global"],
                    "obs_u10": [5.0],
                    "obs_v10": [-5.0],
                    "model_u10": [5.5],
                    "model_v10": [-5.0],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(AssertionError, match="land_station"):
        expedition_calibration_from_pairs(mixed)

    payload = json.loads((FIXTURES / "current_weather.json").read_text(encoding="utf-8"))
    assert_no_land_wind_in_calibration(payload["expedition_calibration"])
    for row in payload["expedition_calibration"]:
        assert "land" not in row["source"]
        assert "metar" not in row["source"]
        assert "sentinel" not in row["source"]


def test_ingest_dry_run_and_format_detect() -> None:
    path = FIXTURES / "synthetic_ascat.json"
    assert detect_format(path) == "json"
    report = ingest_observations(path, dry_run=True)
    assert report.dry_run is True
    assert report.n_cells > 0
    assert report.obs_class == "scatterometer"


def test_fixture_current_weather_validates() -> None:
    payload = json.loads((FIXTURES / "current_weather.json").read_text(encoding="utf-8"))
    validate_current_weather(payload)
    assert payload["meta"]["default_lead_bucket"] == "48-72"
    assert "0-12" in payload["meta"]["circularity_lead_buckets"]
    assert payload["bucket_counts"]["headline"] >= 0


def test_build_payload_from_fixture_store(tmp_path: Path) -> None:
    # Copy fixture store if present, else rebuild via pass
    src_store = FIXTURES / "store"
    if not (src_store / "collocated.parquet").exists():
        pytest.skip("fixture store missing")
    import shutil

    dest = tmp_path / "verify"
    shutil.copytree(src_store, dest)
    cfg = VerifyConfig(store_dir=dest, bootstrap_samples=50)
    payload = build_current_weather_payload(
        dest,
        cfg=cfg,
        stations_yaml=ROOT / "config/stations.yaml",
        generated_utc="2026-08-10T18:00:00Z",
    )
    validate_current_weather(payload)
