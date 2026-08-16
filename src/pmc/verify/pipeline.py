"""Verification pass orchestration and dashboard payload builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contracts.schemas import MODELS_ASSIMILATING_SCATTEROMETER, validate_current_weather

from .calibration import (
    assert_no_land_wind_in_calibration,
    expedition_calibration_from_pairs,
)
from .collocate import assign_buckets, collocate_with_forecasts, thin_to_grid
from .ingest import IngestReport, ingest_observations
from .metrics import bootstrap_ci, direction_mae_deg, speed_bias_ms, vector_rmse_ms
from .sentinel import empty_sentinel_payload, query_sentinel1_wind
from .stations import load_stations, onset_lags, score_mslp
from .store import VerifyConfig, VerifyStore, load_verify_config


@dataclass
class PassSummary:
    pass_id: str
    scatterometer_new_rows: int = 0
    land_station_new_rows: int = 0
    sentinel_status: str = "no_acquisition"
    bucket_counts: dict[str, int] = field(
        default_factory=lambda: {
            "headline": 0,
            "coastal": 0,
            "light_air": 0,
            "qc_reject": 0,
        }
    )
    messages: list[str] = field(default_factory=list)
    noop_classes: list[str] = field(default_factory=list)


def run_verification_pass(
    pass_id: str,
    cfg: VerifyConfig,
    *,
    observation_files: list[Path] | None = None,
    forecasts: pd.DataFrame | None = None,
    station_forecasts: pd.DataFrame | None = None,
    station_model_wind: pd.DataFrame | None = None,
    sentinel_client: Any | None = None,
    stations_yaml: Path | None = None,
) -> PassSummary:
    """Run one 6-hourly verification pass.

    No-op per class when no new obs — never an error, never write empty records.
    """
    store = VerifyStore(cfg.store_dir)
    summary = PassSummary(pass_id=pass_id)
    observation_files = observation_files or []

    scatt_cells: list[pd.DataFrame] = []
    land_cells: list[pd.DataFrame] = []
    qc_reject = 0

    for path in observation_files:
        report = ingest_observations(path, dry_run=False)
        qc_reject += report.n_rejected_qc
        if report.cells is None or report.n_cells == 0:
            summary.messages.extend(report.messages)
            continue
        cells = report.cells.copy()
        cells["source_file_hash"] = report.source_file_hash
        if report.obs_class == "scatterometer":
            scatt_cells.append(cells)
        elif report.obs_class == "land_station":
            land_cells.append(cells)
        else:
            summary.messages.append(f"Skipping unsupported class from {path}")

    summary.bucket_counts["qc_reject"] = qc_reject

    # --- Scatterometer ---
    if not scatt_cells or forecasts is None or forecasts.empty:
        summary.noop_classes.append("scatterometer")
        summary.messages.append("Scatterometer: no-op (no new obs or forecasts).")
    else:
        raw = pd.concat(scatt_cells, ignore_index=True)
        if "obs_speed_ms" not in raw.columns:
            raw["obs_speed_ms"] = np.hypot(raw["obs_u10"], raw["obs_v10"])
        thinned = thin_to_grid(raw, cfg.thin_grid_deg, cfg.max_points_per_pass)
        bucketed = assign_buckets(thinned, cfg)
        for label, n in bucketed["bucket_label"].value_counts().items():
            summary.bucket_counts[str(label)] = int(
                summary.bucket_counts.get(str(label), 0) + n
            )
        collocated = collocate_with_forecasts(
            bucketed, forecasts, pass_id=pass_id, cfg=cfg
        )
        if collocated.empty:
            summary.noop_classes.append("scatterometer")
            summary.messages.append("Scatterometer: no collocated pairs.")
        else:
            n_new = store.append_collocated(collocated)
            summary.scatterometer_new_rows = n_new
            if n_new == 0:
                summary.messages.append("Scatterometer: re-ingest idempotent (0 new rows).")
            else:
                summary.messages.append(f"Scatterometer: inserted {n_new} collocated rows.")

    # --- Land stations ---
    if not land_cells:
        summary.noop_classes.append("land_station")
        summary.messages.append("Land stations: no-op.")
    else:
        land = pd.concat(land_cells, ignore_index=True)
        # Persist MSLP-oriented rows as collocated with NaN model wind; model MSLP
        # join happens below when station_forecasts provided.
        if station_forecasts is not None and not station_forecasts.empty:
            # Store synthetic collocated pressure residuals as pairs with u/v NaN
            # so they can never feed wind calibration.
            pressure_rows = []
            for rec in land.to_dict(orient="records"):
                pressure_rows.append(
                    {
                        "pass_id": pass_id,
                        "obs_class": "land_station",
                        "instrument": "metar_synop",
                        "source_file_hash": rec.get("source_file_hash", ""),
                        "cell_id": f"stn-{rec.get('station_id')}-{rec.get('time')}",
                        "model": "_pressure_placeholder",
                        "run_init": pd.Timestamp(rec["time"]).to_datetime64(),
                        "valid_time": pd.Timestamp(rec["time"]).to_datetime64(),
                        "lead_hours": np.float32(0.0),
                        "lat": np.float32(rec.get("lat", np.nan)),
                        "lon": np.float32(rec.get("lon", np.nan)),
                        "obs_u10": np.float32(np.nan),
                        "obs_v10": np.float32(np.nan),
                        "model_u10": np.float32(np.nan),
                        "model_v10": np.float32(np.nan),
                        "lead_bucket": "0-12",
                        "speed_bucket": "sub_3ms",
                        "region": "land",
                        "bucket_label": "qc_reject",  # excluded from wind headlines
                        "land_dist_km": np.float32(0.0),
                        "mslp_hpa": rec.get("mslp_hpa"),
                        "station_id": rec.get("station_id"),
                    }
                )
            n_new = store.append_collocated(pd.DataFrame(pressure_rows))
            summary.land_station_new_rows = n_new
        else:
            summary.noop_classes.append("land_station")
            summary.messages.append("Land stations: obs present but no station forecasts.")

    # --- Sentinel-1 ---
    sentinel = query_sentinel1_wind(client=sentinel_client)
    summary.sentinel_status = str(sentinel.get("status", "no_acquisition"))
    if summary.sentinel_status == "no_acquisition":
        summary.noop_classes.append("sentinel1")
        summary.messages.append("Sentinel-1: no acquisition.")
    else:
        summary.messages.append("Sentinel-1: swath available (display only, not scored).")

    # Refresh derived scores from full store (scatterometer headline only).
    all_pairs = store.load_collocated()
    scores = _compute_score_rows(all_pairs, cfg)
    store.replace_scores(scores)
    return summary


def _compute_score_rows(pairs: pd.DataFrame, cfg: VerifyConfig) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    scatt = pairs[
        (pairs["obs_class"] == "scatterometer") & (pairs["bucket_label"] == "headline")
    ]
    if scatt.empty:
        return pd.DataFrame()
    rows = []

    def _emit(grp: pd.DataFrame, key: dict) -> None:
        obs_u = grp["obs_u10"].to_numpy(dtype=float)
        obs_v = grp["obs_v10"].to_numpy(dtype=float)
        model_u = grp["model_u10"].to_numpy(dtype=float)
        model_v = grp["model_v10"].to_numpy(dtype=float)
        n = int(len(grp))
        rmse = vector_rmse_ms(obs_u, obs_v, model_u, model_v)
        bias = speed_bias_ms(obs_u, obs_v, model_u, model_v)
        dmae = direction_mae_deg(obs_u, obs_v, model_u, model_v)
        rmse_ci = bootstrap_ci(
            obs_u,
            obs_v,
            model_u,
            model_v,
            statistic="vec_rmse",
            n_boot=cfg.bootstrap_samples,
            seed=cfg.bootstrap_seed,
        )
        bias_ci = bootstrap_ci(
            obs_u,
            obs_v,
            model_u,
            model_v,
            statistic="speed_bias",
            n_boot=cfg.bootstrap_samples,
            seed=cfg.bootstrap_seed,
        )
        assimilating = str(key["model"]) in MODELS_ASSIMILATING_SCATTEROMETER
        circular = assimilating and key["lead_bucket"] == "0-12"
        rows.append(
            {
                **key,
                "vec_rmse_ms": round(float(rmse), 4),
                "vec_rmse_ci95_lo": round(float(rmse_ci[0]), 4),
                "vec_rmse_ci95_hi": round(float(rmse_ci[1]), 4),
                "speed_bias_ms": round(float(bias), 4),
                "speed_bias_ci95_lo": round(float(bias_ci[0]), 4),
                "speed_bias_ci95_hi": round(float(bias_ci[1]), 4),
                "dir_mae_deg": round(float(dmae), 4) if np.isfinite(dmae) else None,
                "n": n,
                "rankable": n >= cfg.min_rank_n,
                "circularity_contaminated": circular,
            }
        )

    # Stratified rows
    group_cols = [
        "pass_id",
        "instrument",
        "model",
        "run_init",
        "lead_bucket",
        "region",
        "speed_bucket",
    ]
    for keys, grp in scatt.groupby(group_cols, dropna=False):
        key = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        _emit(grp, key)

    # Headline aggregates (region=all, speed_bucket=all) for default scorecard
    agg_cols = ["pass_id", "instrument", "model", "run_init", "lead_bucket"]
    for keys, grp in scatt.groupby(agg_cols, dropna=False):
        key = dict(zip(agg_cols, keys if isinstance(keys, tuple) else (keys,)))
        key["region"] = "all"
        key["speed_bucket"] = "all"
        _emit(grp, key)

    return pd.DataFrame(rows)


def build_current_weather_payload(
    store_dir: Path,
    *,
    cfg: VerifyConfig | None = None,
    stations_yaml: Path | None = None,
    station_obs: pd.DataFrame | None = None,
    station_forecasts: pd.DataFrame | None = None,
    station_model_wind: pd.DataFrame | None = None,
    sentinel: dict[str, Any] | None = None,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Build the C9 ``current_weather`` dashboard section from the store."""
    cfg = cfg or VerifyConfig(store_dir=store_dir)
    store = VerifyStore(store_dir)
    pairs = store.load_collocated()
    scores = store.load_scores()
    if scores.empty and not pairs.empty:
        scores = _compute_score_rows(pairs, cfg)

    scorecard = []
    for rec in scores.to_dict(orient="records") if not scores.empty else []:
        scorecard.append(
            {
                "pass_id": rec["pass_id"],
                "instrument": rec["instrument"],
                "model": rec["model"],
                "lead_bucket": rec["lead_bucket"],
                "region": rec.get("region", "all"),
                "speed_bucket": rec.get("speed_bucket", "all"),
                "vec_rmse_ms": rec["vec_rmse_ms"],
                "vec_rmse_ci95": [rec["vec_rmse_ci95_lo"], rec["vec_rmse_ci95_hi"]],
                "speed_bias_ms": rec["speed_bias_ms"],
                "speed_bias_ci95": [
                    rec["speed_bias_ci95_lo"],
                    rec["speed_bias_ci95_hi"],
                ],
                "dir_mae_deg": rec["dir_mae_deg"],
                "n": int(rec["n"]),
                "rankable": bool(rec["rankable"]),
                "circularity_contaminated": bool(rec["circularity_contaminated"]),
            }
        )

    # Trend: latest lead-bucket 48-72 vec RMSE per model across passes
    trend = []
    if not scores.empty:
        t = scores[scores["lead_bucket"] == "48-72"]
        for _, rec in t.iterrows():
            trend.append(
                {
                    "pass_id": rec["pass_id"],
                    "pass_time_utc": _pass_time_utc(str(rec["pass_id"])),
                    "model": rec["model"],
                    "instrument": rec["instrument"],
                    "lead_bucket": "48-72",
                    "vec_rmse_ms": rec["vec_rmse_ms"],
                    "n": int(rec["n"]),
                }
            )

    residuals = []
    if not pairs.empty:
        scatt = pairs[
            (pairs["obs_class"] == "scatterometer")
            & (pairs["bucket_label"] == "headline")
            & (pairs["lead_bucket"] == "48-72")
        ]
        # Cap residual points for payload size
        sample = scatt.head(500)
        for rec in sample.to_dict(orient="records"):
            residuals.append(
                {
                    "pass_id": rec["pass_id"],
                    "model": rec["model"],
                    "lat": float(rec["lat"]),
                    "lon": float(rec["lon"]),
                    "du_ms": float(rec["obs_u10"] - rec["model_u10"]),
                    "dv_ms": float(rec["obs_v10"] - rec["model_v10"]),
                    "obs_time_utc": pd.Timestamp(rec["valid_time"])
                    .tz_localize("UTC")
                    .strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )

    # Expedition calibration — scatterometer 48-72 only
    scatt_pairs = (
        pairs[pairs["obs_class"] == "scatterometer"]
        if not pairs.empty
        else pd.DataFrame()
    )
    calibration = expedition_calibration_from_pairs(
        scatt_pairs,
        n_boot=cfg.bootstrap_samples,
        seed=cfg.bootstrap_seed,
        min_n=cfg.min_rank_n,
    )
    assert_no_land_wind_in_calibration(calibration, scatt_pairs if not scatt_pairs.empty else None)

    stations = []
    if stations_yaml and stations_yaml.exists():
        stations = [
            {
                "id": s["id"],
                "name": s["name"],
                "lat": s["lat"],
                "lon": s["lon"],
                "high_value": bool(s.get("high_value", False)),
            }
            for s in load_stations(stations_yaml)
        ]

    mslp_scores: list[dict[str, Any]] = []
    onset: list[dict[str, Any]] = []
    if station_obs is not None and station_forecasts is not None:
        mslp_scores = score_mslp(station_obs, station_forecasts)
    if station_obs is not None and station_model_wind is not None:
        onset = onset_lags(
            station_obs,
            station_model_wind,
            speed_kt=6.0,
            dir_min=90.0,
            dir_max=220.0,
            kind="thermal",
        )

    bucket_counts = {"headline": 0, "coastal": 0, "light_air": 0, "qc_reject": 0}
    if not pairs.empty and "bucket_label" in pairs.columns:
        for label, n in pairs["bucket_label"].value_counts().items():
            if label in bucket_counts:
                bucket_counts[str(label)] = int(n)

    gen = generated_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "meta": {
            "generated_utc": gen,
            "equivalent_neutral_correction": cfg.equivalent_neutral_correction,
            "min_rank_n": cfg.min_rank_n,
            "default_lead_bucket": cfg.default_lead_bucket,
            "circularity_lead_buckets": ["0-12"],
            "models_assimilating_scatterometer": list(cfg.assimilating),
            "models_non_assimilating": list(cfg.non_assimilating),
            "warnings": [
                "0–12 h scatterometer scores are assimilation-contaminated for "
                "IFS, ICON, AROME and ARPEGE — they measure assimilation, not skill.",
                "48–72 h is the decision-relevant lead bucket (default view).",
                "Expedition calibration uses scatterometer 48–72 h only; "
                "land stations and Sentinel-1 cannot contribute.",
                (
                    "Equivalent-neutral correction: "
                    + ("applied." if cfg.equivalent_neutral_correction else "not applied.")
                ),
            ],
        },
        "scorecard": scorecard,
        "trend": trend,
        "residuals": residuals,
        "pressure": {
            "stations": stations,
            "mslp_scores": mslp_scores,
            "onset_lags": onset,
        },
        "sentinel": sentinel or empty_sentinel_payload(),
        "expedition_calibration": [
            {k: v for k, v in row.items() if k != "rankable"} for row in calibration
        ],
        "bucket_counts": bucket_counts,
    }
    validate_current_weather(payload)
    return payload


def _pass_time_utc(pass_id: str) -> str:
    # pass_id forms: "pass-2026-08-10T12:00:00Z" or similar
    if "T" in pass_id:
        frag = pass_id.split("pass-", 1)[-1]
        if frag.endswith("Z"):
            return frag
    return "1970-01-01T00:00:00Z"
