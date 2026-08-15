"""A6 — Dashboard/report module."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from contracts.schemas import validate_dashboard_payload

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

MAX_DASHBOARD_PAYLOAD_BYTES = 20 * 1024 * 1024
OPTIONAL_CLIMO_VARS = (
    "p_below_8kt",
    "p_above_20kt",
    "vector_mean_u",
    "vector_mean_v",
    "directional_const",
    "n_samples",
)


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value_f = float(value)
        if not np.isfinite(value_f):
            return None
        return value_f
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        ns = value.astype("datetime64[ns]").astype("int64")
        dt = datetime.fromtimestamp(float(ns) / 1_000_000_000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    return _normalize_scalar(value)


def _round_value(value: Any) -> Any:
    if isinstance(value, float):
        return float(np.round(value, 2))
    if isinstance(value, list):
        return [_round_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _round_value(v) for k, v in value.items()}
    return value


def _build_climatology_by_hour(climatology_ds: "xr.Dataset") -> list[dict[str, Any]]:
    by_hour: list[dict[str, Any]] = []
    for hr in climatology_ds["hour"].values.astype(int).tolist():
        row = {
            "hour": int(hr),
            "p_below_5kt": climatology_ds["p_below_5kt"].sel(hour=hr).values,
            "mean_tws_kt": climatology_ds["mean_tws_kt"].sel(hour=hr).values,
        }
        for var in OPTIONAL_CLIMO_VARS:
            if var in climatology_ds:
                row[var] = climatology_ds[var].sel(hour=hr).values
        by_hour.append(row)
    return by_hour


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _round_value(_to_jsonable(payload))
    validate_dashboard_payload(normalized)
    return normalized


def emit(
    *,
    climatology_ds: "xr.Dataset",
    routes_summary: list[dict[str, Any]],
    head_to_head_df: "pd.DataFrame",
    skill_rows: list[dict[str, Any]],
    meta: dict[str, Any],
    output_path: Path,
    extra_sections: dict[str, Any] | None = None,
) -> Path:
    """Emit a C7 dashboard payload JSON file."""
    payload: dict[str, Any] = {
        "meta": meta,
        "climatology": {
            "grid": {
                "lat": climatology_ds["lat"].values,
                "lon": climatology_ds["lon"].values,
            },
            "by_hour": _build_climatology_by_hour(climatology_ds),
        },
        "routes": routes_summary,
        "head_to_head": head_to_head_df.to_dict(orient="records"),
        "skill": skill_rows,
    }
    if extra_sections:
        for key, value in extra_sections.items():
            if key in payload:
                raise ValueError(f"extra_sections must not override core key: {key}")
            payload[key] = value

    normalized = _normalize_payload(payload)
    encoded = json.dumps(
        normalized,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    if len(encoded.encode("utf-8")) > MAX_DASHBOARD_PAYLOAD_BYTES:
        raise ValueError(
            "Dashboard payload exceeds 20 MB contract guidance. "
            "Trim samples or reduce precision."
        )

    return _write_payload_files(output_path, encoded)


def _write_payload_files(output_path: Path, encoded: str) -> Path:
    """Write data.json plus a data.js sibling so file:// pages can load without fetch."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(encoded, encoding="utf-8")
    js_path = output_path.with_name(f"{output_path.stem}.js")
    js_path.write_text(f"window.DASHBOARD_PAYLOAD = {encoded.rstrip()};\n", encoding="utf-8")
    return output_path


def validate_and_write_payload(input_path: Path, output_path: Path) -> Path:
    """Validate and normalize an existing payload JSON before writing it out."""
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    normalized = _normalize_payload(payload)
    encoded = json.dumps(
        normalized,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    return _write_payload_files(output_path, encoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate/normalize an existing dashboard JSON payload and write output. "
            "Useful for preparing dashboard/data.json from fixture or generated data."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSON payload path (C7 shape).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path (normalized, rounded, validated).",
    )
    args = parser.parse_args(argv)

    out_path = validate_and_write_payload(args.input, args.output)
    print(f"Wrote dashboard payload: {out_path}")
    print(f"Wrote file:// companion: {out_path.with_name(out_path.stem + '.js')}")
    print(f"Validated at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    return 0
