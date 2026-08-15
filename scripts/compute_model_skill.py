#!/usr/bin/env python3
"""Compute C7 model-skill rows from Open-Meteo previous runs vs IFS analysis.

Uses ``wind_speed_10m_previous_dayN`` / ``wind_direction_10m_previous_dayN``
columns (not the ``previous_day`` query param). Always requests
``wind_speed_unit=ms`` and asserts the API's declared units.

Resamples to the common 00/06/12/18 UTC grid before scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from pmc.io.units import (  # noqa: E402
    assert_direction_unit_deg,
    assert_hourly_units,
    assert_speed_unit_ms,
    extract_responses,
)

PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

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

# ukmo dropped (HTTP 400). arpege_europe has no previous_dayN coverage on this API.
# ecmwf_ifs previous_dayN is empty historically; use ifs025 for ECMWF forecast skill.
DEFAULT_MODELS = (
    "ecmwf_ifs025",
    "ecmwf_aifs025",
    "gfs_global",
    "icon_global",
    "gem_global",
)

COMMON_HOURS_UTC = (0, 6, 12, 18)
MS_TO_KT = 1.9438445


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
            req = urllib.request.Request(url, headers={"User-Agent": "pmc-skill/2.0"})
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


def _speed_dir_to_uv(speed_ms: float, direction_from_deg: float) -> tuple[float, float]:
    radians = math.radians(direction_from_deg)
    u = -speed_ms * math.sin(radians)
    v = -speed_ms * math.cos(radians)
    return u, v


def _assert_payload_units_ms(payload: Any, speed_vars: list[str], dir_vars: list[str], *, context: str) -> None:
    assert_speed_unit_ms(payload, speed_vars, context=context)
    assert_direction_unit_deg(payload, dir_vars, context=context)
    # Belt-and-braces: also require wind_speed_unit request contract via declared units only.
    assert_hourly_units(
        payload,
        expected={name: "m/s" for name in speed_vars} | {name: "°" for name in dir_vars},
        context=context,
    )


def _archive_hourly() -> str:
    return "wind_speed_10m,wind_direction_10m"


def _forecast_hourly(leads: tuple[int, ...]) -> str:
    parts: list[str] = []
    for lead in leads:
        parts.append(f"wind_speed_10m_previous_day{lead}")
        parts.append(f"wind_direction_10m_previous_day{lead}")
    return ",".join(parts)


def _base_params(start: date, end: date, model: str, hourly: str) -> dict[str, Any]:
    return {
        "latitude": ",".join(f"{lat:.4f}" for lat, _ in SAMPLE_POINTS),
        "longitude": ",".join(f"{lon:.4f}" for _, lon in SAMPLE_POINTS),
        "hourly": hourly,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "models": model,
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }


def _rows_from_archive(payload: Any) -> list[dict[str, Any]]:
    _assert_payload_units_ms(
        payload,
        ["wind_speed_10m"],
        ["wind_direction_10m"],
        context="archive analysis reference",
    )
    rows: list[dict[str, Any]] = []
    for idx, point in enumerate(extract_responses(payload)):
        hourly = point.get("hourly") or {}
        times = hourly.get("time") or []
        speeds = hourly.get("wind_speed_10m") or []
        dirs = hourly.get("wind_direction_10m") or []
        if idx < len(SAMPLE_POINTS):
            lat, lon = SAMPLE_POINTS[idx]
        else:
            lat = float(point["latitude"])
            lon = float(point["longitude"])
        for ts, spd, direction in zip(times, speeds, dirs):
            if spd is None or direction is None:
                continue
            u, v = _speed_dir_to_uv(float(spd), float(direction))
            rows.append(
                {
                    "time": ts,
                    "lat": lat,
                    "lon": lon,
                    "u10_ref": u,
                    "v10_ref": v,
                }
            )
    return rows


def _rows_from_previous_dayn(payload: Any, *, model: str, leads: tuple[int, ...]) -> list[dict[str, Any]]:
    speed_vars = [f"wind_speed_10m_previous_day{lead}" for lead in leads]
    dir_vars = [f"wind_direction_10m_previous_day{lead}" for lead in leads]
    _assert_payload_units_ms(
        payload,
        speed_vars,
        dir_vars,
        context=f"previous_runs model={model}",
    )
    rows: list[dict[str, Any]] = []
    for idx, point in enumerate(extract_responses(payload)):
        hourly = point.get("hourly") or {}
        times = hourly.get("time") or []
        if idx < len(SAMPLE_POINTS):
            lat, lon = SAMPLE_POINTS[idx]
        else:
            lat = float(point["latitude"])
            lon = float(point["longitude"])
        for lead in leads:
            speeds = hourly.get(f"wind_speed_10m_previous_day{lead}") or []
            dirs = hourly.get(f"wind_direction_10m_previous_day{lead}") or []
            for ts, spd, direction in zip(times, speeds, dirs):
                if spd is None or direction is None:
                    continue
                u, v = _speed_dir_to_uv(float(spd), float(direction))
                rows.append(
                    {
                        "time": ts,
                        "lat": lat,
                        "lon": lon,
                        "model": model,
                        "lead_days": int(lead),
                        "u10_pred": u,
                        "v10_pred": v,
                    }
                )
    return rows


def _to_common_grid(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out = out[out["time"].dt.hour.isin(COMMON_HOURS_UTC)]
    return out


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
    print(
        f"[skill] chunks={len(chunks)} points={len(SAMPLE_POINTS)} "
        f"models={len(models)} leads={leads} grid={COMMON_HOURS_UTC} unit=ms"
    )

    ref_frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in chunks:
        params = _base_params(chunk_start, chunk_end, "ecmwf_ifs", _archive_hourly())
        payload = _fetch_json(ARCHIVE, params, cache)
        rows = _rows_from_archive(payload)
        if rows:
            ref_frames.append(pd.DataFrame(rows))
        print(f"[skill] archive {chunk_start}..{chunk_end}: {len(rows)} rows")

    if not ref_frames:
        raise SystemExit("No reference (analysis) rows fetched")
    ref = _to_common_grid(pd.concat(ref_frames, ignore_index=True))
    ref = ref.drop_duplicates(subset=["time", "lat", "lon"], keep="last")
    print(f"[skill] reference on 6h grid: {len(ref)} rows")

    jobs: list[tuple[str, date, date]] = []
    for model in models:
        for chunk_start, chunk_end in chunks:
            jobs.append((model, chunk_start, chunk_end))

    pred_rows: list[dict[str, Any]] = []
    done = 0
    hourly = _forecast_hourly(leads)

    def _one(job: tuple[str, date, date]) -> list[dict[str, Any]]:
        model, chunk_start, chunk_end = job
        params = _base_params(chunk_start, chunk_end, model, hourly)
        # Never pass previous_day query param.
        assert "previous_day" not in params
        payload = _fetch_json(PREVIOUS_RUNS, params, cache)
        return _rows_from_previous_dayn(payload, model=model, leads=leads)

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
            if done % 10 == 0 or done == len(jobs):
                print(f"[skill] previous_runs {done}/{len(jobs)}")

    if not pred_rows:
        raise SystemExit("No previous-run rows fetched")

    pred = _to_common_grid(pd.DataFrame(pred_rows))
    merged = pred.merge(ref, on=["time", "lat", "lon"], how="inner")
    print(
        f"[skill] paired rows={len(merged)} models={sorted(merged['model'].unique().tolist())} "
        f"leads={sorted(merged['lead_days'].unique().tolist())}"
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


def run_sanity_gates(skill_rows: list[dict[str, Any]]) -> list[str]:
    """Return a list of human-readable failures. Empty means pass."""

    failures: list[str] = []
    if not skill_rows:
        return ["no skill rows produced"]

    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in skill_rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    # Aggregate RMSE across wind bins (sample-weighted) per model×lead for trend check.
    for model, rows in sorted(by_model.items()):
        biased = any(bool(r["reference_biased"]) for r in rows)
        lead_rmse: dict[int, list[tuple[float, int]]] = {}
        for row in rows:
            lead_rmse.setdefault(int(row["lead_days"]), []).append(
                (float(row["vec_rmse_kt"]), int(row["n_samples"]))
            )

        def _weighted(pairs: list[tuple[float, int]]) -> float:
            n = sum(n for _, n in pairs)
            if n <= 0:
                return float("nan")
            return sum(v * n for v, n in pairs) / n

        leads_present = sorted(lead_rmse)
        if 1 in lead_rmse and 7 in lead_rmse:
            r1 = _weighted(lead_rmse[1])
            r7 = _weighted(lead_rmse[7])
            if not (r7 > r1):
                failures.append(
                    f"{model}: vec_rmse must increase lead7>lead1 "
                    f"(got lead1={r1:.3f} lead7={r7:.3f})"
                )
        elif not biased and leads_present:
            # Independent model missing lead7 (e.g. ICON): require max_lead > lead1.
            lo = leads_present[0]
            hi = leads_present[-1]
            if lo != hi:
                r_lo = _weighted(lead_rmse[lo])
                r_hi = _weighted(lead_rmse[hi])
                if not (r_hi > r_lo):
                    failures.append(
                        f"{model}: vec_rmse must increase lead{hi}>lead{lo} "
                        f"(got {r_lo:.3f} vs {r_hi:.3f}; lead7 unavailable)"
                    )
            else:
                failures.append(f"{model}: only one lead present ({lo}); cannot verify degradation")

        if "ecmwf" in model.lower():
            values = [float(r["vec_rmse_kt"]) for r in rows]
            if max(values) <= 0.0:
                failures.append(f"{model}: vec_rmse is zero at every lead/bin (self-comparison?)")

    # Wind-bin sample counts: pool all models/leads for observed-bin occupancy,
    # using one model's lead to avoid multi-counting the same obs — use max n per bin.
    bin_n: dict[str, int] = {}
    for row in skill_rows:
        wind_bin = str(row["wind_bin"])
        bin_n[wind_bin] = max(bin_n.get(wind_bin, 0), int(row["n_samples"]))
    n_light = bin_n.get("0-6kt", 0)
    n_strong = bin_n.get("20kt+", 0)
    if not (n_light > n_strong):
        failures.append(
            f"wind bins inverted or still wrong units: 0-6kt n={n_light} vs 20kt+ n={n_strong}"
        )

    # Soft range check — report but do not hard-fail unless wildly off.
    indep = [r for r in skill_rows if not r["reference_biased"] and int(r["lead_days"]) == 1]
    if indep:
        rmses = [float(r["vec_rmse_kt"]) for r in indep]
        mean_l1 = sum(rmses) / len(rmses)
        if mean_l1 > 11.0:
            failures.append(
                f"lead-1 independent mean vec_rmse={mean_l1:.2f} kt still looks broken "
                f"(expected roughly 3-5 kt)"
            )

    return failures


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


def light_air_table(skill_rows: list[dict[str, Any]]) -> str:
    lines = [
        "| model | lead | vec_rmse_kt | speed_bias_kt | dir_mae_deg | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rows = [
        r
        for r in skill_rows
        if (not r["reference_biased"])
        and str(r["wind_bin"]) == "0-6kt"
        and int(r["lead_days"]) in {2, 3}
    ]
    rows.sort(key=lambda r: (int(r["lead_days"]), str(r["model"])))
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['lead_days']} | {r['vec_rmse_kt']:.2f} | "
            f"{r['speed_bias_kt']:.2f} | {r['dir_mae_deg']:.2f} | {r['n_samples']} |"
        )
    if len(rows) == 0:
        lines.append("| (none) | | | | | |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--leads", default="1,2,3,4,5,6,7")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "skill" / "skill_rows.json",
    )
    parser.add_argument("--patch-dashboard", action="store_true")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "cache" / "model_skill_v2",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    leads = tuple(int(x) for x in args.leads.split(",") if x.strip())
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    cache = RequestCache(args.cache_dir)

    pairs = fetch_pairs(
        start=start,
        end=end,
        models=models,
        leads=leads,
        cache=cache,
        max_workers=args.workers,
    )
    # Observed speed sanity print (must look like knots, median ~6-10)
    obs_kt = np.hypot(pairs["u10_ref"].to_numpy(), pairs["v10_ref"].to_numpy()) * MS_TO_KT
    print(
        f"[skill] observed kt mean/median/p90="
        f"{float(np.mean(obs_kt)):.2f}/{float(np.median(obs_kt)):.2f}/{float(np.percentile(obs_kt, 90)):.2f}"
    )

    skill_rows = compute_skill_rows(pairs)
    failures = run_sanity_gates(skill_rows)
    if failures:
        print("[skill] SANITY GATES FAILED — not shipping:")
        for msg in failures:
            print(f"  - {msg}")
        # Still write a debug artifact, but do not patch dashboard.
        debug_path = args.output.with_name(args.output.stem + "_FAILED.json")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(
            json.dumps({"failures": failures, "skill": skill_rows}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[skill] wrote debug {debug_path}")
        print("\nLight-air table (for diagnosis):\n")
        print(light_air_table(skill_rows))
        return 2

    meta_extra = {
        "reference": "ecmwf_ifs_analysis",
        "forecast_columns": "wind_speed_10m_previous_dayN,wind_direction_10m_previous_dayN",
        "wind_speed_unit": "ms",
        "time_grid_utc": list(COMMON_HOURS_UTC),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "n_points": len(SAMPLE_POINTS),
        "n_paired": int(len(pairs)),
        "models": list(models),
        "leads": list(leads),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            "Sparse corridor sample vs IFS analysis on 00/06/12/18 UTC. "
            "ECMWF/AIFS rows are reference_biased. ukmo dropped (API 400). "
            "arpege_europe omitted (no previous_dayN coverage)."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"meta": meta_extra, "skill": skill_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[skill] wrote {args.output} rows={len(skill_rows)}")
    print("\nLight-air table (0-6kt, leads 2–3, independent only):\n")
    print(light_air_table(skill_rows))

    if args.patch_dashboard:
        patch_dashboard(skill_rows, meta_extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
