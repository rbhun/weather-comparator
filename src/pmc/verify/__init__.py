"""Live observation verification for the Current weather dashboard tab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .calibration import expedition_calibration_from_pairs, assert_no_land_wind_in_calibration
from .ingest import IngestReport, detect_format, ingest_observations
from .metrics import bootstrap_ci, direction_mae_deg, speed_bias_ms, vector_rmse_ms
from .pipeline import PassSummary, VerifyConfig, build_current_weather_payload, run_verification_pass
from .store import VerifyStore, load_verify_config

__all__ = [
    "IngestReport",
    "PassSummary",
    "VerifyConfig",
    "VerifyStore",
    "assert_no_land_wind_in_calibration",
    "bootstrap_ci",
    "build_current_weather_payload",
    "detect_format",
    "direction_mae_deg",
    "expedition_calibration_from_pairs",
    "ingest_observations",
    "load_verify_config",
    "run_verification_pass",
    "speed_bias_ms",
    "vector_rmse_ms",
]


def empty_current_weather_payload() -> dict[str, Any]:
    """Minimal valid C9 payload used when no passes have run yet."""
    from contracts.schemas import (
        MODELS_ASSIMILATING_SCATTEROMETER,
        MODELS_NON_ASSIMILATING_SCATTEROMETER,
        VERIFY_MIN_RANK_N,
    )

    return {
        "meta": {
            "generated_utc": "1970-01-01T00:00:00Z",
            "equivalent_neutral_correction": False,
            "min_rank_n": VERIFY_MIN_RANK_N,
            "default_lead_bucket": "48-72",
            "circularity_lead_buckets": ["0-12"],
            "models_assimilating_scatterometer": list(MODELS_ASSIMILATING_SCATTEROMETER),
            "models_non_assimilating": list(MODELS_NON_ASSIMILATING_SCATTEROMETER),
            "warnings": [
                "No live verification passes yet.",
                "0–12 h scatterometer scores are assimilation-contaminated for IFS/ICON/AROME/ARPEGE.",
            ],
        },
        "scorecard": [],
        "trend": [],
        "residuals": [],
        "pressure": {"stations": [], "mslp_scores": [], "onset_lags": []},
        "sentinel": {
            "status": "no_acquisition",
            "acquisition_utc": None,
            "footprint": None,
            "speed_field": None,
            "model_speed_fields": [],
        },
        "expedition_calibration": [],
        "bucket_counts": {
            "headline": 0,
            "coastal": 0,
            "light_air": 0,
            "qc_reject": 0,
        },
    }
