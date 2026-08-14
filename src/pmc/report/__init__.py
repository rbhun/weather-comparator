"""A6 — Dashboard/report module (real payload emitter)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from contracts.schemas import validate_dashboard_payload


def _round_value(value: Any) -> Any:
    if isinstance(value, float):
        return float(np.round(value, 2))
    if isinstance(value, np.floating):
        return float(np.round(float(value), 2))
    if isinstance(value, list):
        return [_round_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _round_value(v) for k, v in value.items()}
    return value


def emit(
    *,
    climatology_ds: xr.Dataset,
    routes_summary: list[dict[str, Any]],
    head_to_head_df: pd.DataFrame,
    skill_rows: list[dict[str, Any]],
    meta: dict[str, Any],
    output_path: Path,
) -> Path:
    """Emit a C7 dashboard payload JSON file."""
    by_hour = []
    for hr in climatology_ds["hour"].values.astype(int).tolist():
        by_hour.append(
            {
                "hour": int(hr),
                "p_below_5kt": _round_value(
                    climatology_ds["p_below_5kt"].sel(hour=hr).values.tolist()
                ),
                "mean_tws_kt": _round_value(
                    climatology_ds["mean_tws_kt"].sel(hour=hr).values.tolist()
                ),
            }
        )

    payload = {
        "meta": _round_value(meta),
        "climatology": {
            "grid": {
                "lat": _round_value(climatology_ds["lat"].values.tolist()),
                "lon": _round_value(climatology_ds["lon"].values.tolist()),
            },
            "by_hour": by_hour,
        },
        "routes": _round_value(routes_summary),
        "head_to_head": _round_value(head_to_head_df.to_dict(orient="records")),
        "skill": _round_value(skill_rows),
    }
    validate_dashboard_payload(payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return output_path
