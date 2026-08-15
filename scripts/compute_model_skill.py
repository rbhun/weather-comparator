#!/usr/bin/env python3
"""Compute C7 model-skill rows from Open-Meteo previous runs vs IFS analysis.

Samples sea points along the race corridor, compares each forecast model at
lead days 1–7 against ECMWF IFS analysis, stratified by observed wind bin.
Writes JSON suitable for the dashboard `skill` array and optionally patches
dashboard/data.json + data.js.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pmc import stats as stats_mod  # noqa: E402

PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = "wind_u_component_10m,wind_v_component_10m"

# Corridor sample: start → Ustica → Sardinia E → gate → Bonifacio → Corsica E → Liguria → finish
SAMPLE_POINTS: list[tuple[float, float]] = [
    (38.20, 13.32),
    (38.60, 13.00),
    (39.40, 11.80),
    (40.20, 11.20),
    (40.80, 10.40),
    (41.13, 9.55),
    (41.40, 9.80),
    (42.20, 10.00),
    (42.80, 9.20),
    (43.40, 8.20),
    (43.73, 7.42),
]

DEFAULT_MODELS = (
    "ecmwf_ifs",
    "ecmwf_aifs025",
    "gfs_global",
    "icon_global",
    "gem_global",
    "arpege_europe",
)


def _api_key() -> str | None:
    key = os.getenv("OPENMETEO_API_KEY", "").strip()
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENMETEO_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
                return key or None
    return None


def _month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        chunk_start = max(cur, start)
        chunk_end = min(nxt - timedelta(days=1), end)
        if chunk_start <= chunk_end:
            chunks.append((chunk_start, chunk_end))
        cur = nxt
    return chunks


class RequestCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, endpoint: str, params: dict[str, Any]) -> Path:
        canonical = json.dumps(
            {"endpoint": endpoint, "params": params},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def get(self, endpoint: str, params: dict[str, Any]) -> Any | None:
        path = self.path_for(endpoint, params)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, endpoint: str, params: dict[str, Any], payload: Any) -> None:
        path = self.path_for(endpoint, params)
        path.write_text(json.dumps(payload), encoding="utf-8")


def _fetch_json(
    endpoint: str,
    params: dict[str, Any],
    cache: RequestCache,
    *,
    timeout: int = 90,
) -> Any:
    cached = cache.get(endpoint, params)
    if cached is not None:
        return cached

    query = dict(params)
    key = _api_key()
    if key:
        query["apikey"] = key
    url = f"{endpoint}?{urllib.parse.urlencode(query)}"
    last_err: Exception | None = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pmc-skill/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.load(resp)
            cache.put(endpoint, params, payload)
            return payload
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code in {429, 500, 502, 503, 504}:
                time.sleep(2**attempt + 0.5)
                continue
            raise
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2**attempt + 0.5)
    raise RuntimeError(f"Failed fetching {endpoint}: {last_err}")


def _as_point_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise TypeError(f"Unexpected payload type: {type(payload)!r}")


def _rows_from_payload(
    payload: Any,
    *,
    model: str,
    lead_days: int | None,
    role: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    points = _as_point_list(payload)
    for idx, point in enumerate(points):
        hourly = point.get("hourly") or {}
        times = hourly.get("time") or []
        u_vals = hourly.get("wind_u_component_10m") or []
        v_vals = hourly.get("wind_v_component_10m") or []
        if not times:
            continue
        # Join on requested corridor points, not native model grid coords —
        # each model snaps to a slightly different lat/lon.
        if idx < len(SAMPLE_POINTS):
            lat, lon = SAMPLE_POINTS[idx]
        else:
            lat = float(point.get("latitude"))
            lon = float(point.get("longitude"))
        for ts, u, v in zip(times, u_vals, v_vals):
            if u is None or v is None:
                continue
            row: dict[str, Any] = {
                "time": ts,
                "lat": lat,
                "lon": lon,
                "model": model,
                f"u10_{role}": float(u),
                f"v10_{role}": float(v),
            }
            if lead_days is not None:
                row["lead_days"] = int(lead_days)
            rows.append(row)
    return rows


def _point_params(start: date, end: date, model: str, previous_day: int | None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "latitude": ",".join(f"{lat:.4f}" for lat, _ in SAMPLE_POINTS),
        "longitude": ",".join(f"{lon:.4f}" for _, lon in SAMPLE_POINTS),
        "hourly": HOURLY,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "models": model,
        "timezone": "UTC",
    }
    if previous_day is not None:
        params["previous_day"] = int(previous_day)
    return params


def fetch_pairs(
    *,
    start: date,
    end: date,
    models: tuple[str, ...],
    leads: tuple[int, ...],
    cache: RequestCache,
    max_workers: int = 6,
) -> pd.DataFrame:
    chunks = _month_chunks(start, end)
    print(f"[skill] chunks={len(chunks)} points={len(SAMPLE_POINTS)} models={len(models)} leads={leads}")

    ref_frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in chunks:
        params = _point_params(chunk_start, chunk_end, "ecmwf_ifs", None)
        payload = _fetch_json(ARCHIVE, params, cache)
        rows = _rows_from_payload(payload, model="ecmwf_ifs", lead_days=None, role="ref")
        if rows:
            ref_frames.append(pd.DataFrame(rows))
        print(f"[skill] archive {chunk_start}..{chunk_end}: {len(rows)} rows")

    if not ref_frames:
        raise SystemExit("No reference (analysis) rows fetched")
    ref = pd.concat(ref_frames, ignore_index=True)
    ref["time"] = pd.to_datetime(ref["time"], utc=True)
    ref = ref.drop_duplicates(subset=["time", "lat", "lon"], keep="last")

    jobs: list[tuple[str, int, date, date]] = []
    for model in models:
        for lead in leads:
            for chunk_start, chunk_end in chunks:
                jobs.append((model, lead, chunk_start, chunk_end))

    pred_rows: list[dict[str, Any]] = []
    done = 0

    def _one(job: tuple[str, int, date, date]) -> list[dict[str, Any]]:
        model, lead, chunk_start, chunk_end = job
        params = _point_params(chunk_start, chunk_end, model, lead)
        payload = _fetch_json(PREVIOUS_RUNS, params, cache)
        return _rows_from_payload(payload, model=model, lead_days=lead, role="pred")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_one, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                rows = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[skill] FAIL {job}: {exc}")
                continue
            pred_rows.extend(rows)
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"[skill] previous_runs {done}/{len(jobs)}")

    if not pred_rows:
        raise SystemExit("No previous-run rows fetched")

    pred = pd.DataFrame(pred_rows)
    pred["time"] = pd.to_datetime(pred["time"], utc=True)

    merged = pred.merge(
        ref[["time", "lat", "lon", "u10_ref", "v10_ref"]],
        on=["time", "lat", "lon"],
        how="inner",
    )
    print(
        f"[skill] paired rows={len(merged)} models={sorted(merged['model'].unique().tolist())}"
    )
    return merged


def compute_skill_rows(pairs: pd.DataFrame) -> list[dict[str, Any]]:
    skill = stats_mod.model_skill(pairs)
    rows = skill.to_dict(orient="records")
    for row in rows:
        row["lead_days"] = int(row["lead_days"])
        row["n_samples"] = int(row.get("n_samples", 0))
        row["reference_biased"] = bool(row["reference_biased"])
        for key in ("vec_rmse_kt", "speed_bias_kt", "dir_mae_deg"):
            row[key] = float(row[key])
    return rows


def patch_dashboard(skill_rows: list[dict[str, Any]], meta_extra: dict[str, Any]) -> None:
    data_path = ROOT / "dashboard" / "data.json"
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    payload["skill"] = skill_rows
    meta = payload.setdefault("meta", {})
    meta["skill"] = meta_extra
    warning = (
        "Model skill scores ECMWF IFS / AIFS against IFS analysis "
        "(reference-biased). Independent models are comparable to each other; "
        "biased rows are not."
    )
    warnings = list(meta.get("warnings") or [])
    if warning not in warnings:
        warnings.append(warning)
    meta["warnings"] = warnings
    encoded = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    data_path.write_text(encoded, encoding="utf-8")
    js_path = ROOT / "dashboard" / "data.js"
    js_path.write_text(f"window.DASHBOARD_PAYLOAD = {encoded.rstrip()};\n", encoding="utf-8")
    print(f"[skill] patched {data_path} and {js_path} ({len(skill_rows)} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-07-31")
    parser.add_argument("--leads", default="1,2,3,4,5,6,7")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "skill" / "skill_rows.json",
    )
    parser.add_argument("--patch-dashboard", action="store_true")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    leads = tuple(int(x) for x in args.leads.split(",") if x.strip())
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    cache = RequestCache(ROOT / "data" / "cache" / "model_skill")

    pairs = fetch_pairs(
        start=start,
        end=end,
        models=models,
        leads=leads,
        cache=cache,
        max_workers=args.workers,
    )
    skill_rows = compute_skill_rows(pairs)
    meta_extra = {
        "reference": "ecmwf_ifs_analysis",
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "n_points": len(SAMPLE_POINTS),
        "n_paired": int(len(pairs)),
        "models": list(models),
        "leads": list(leads),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            "Sparse corridor sample vs IFS analysis. Stratified by observed wind bin. "
            "ECMWF/AIFS rows are reference_biased."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"meta": meta_extra, "skill": skill_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[skill] wrote {args.output} rows={len(skill_rows)}")

    if args.patch_dashboard:
        patch_dashboard(skill_rows, meta_extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
