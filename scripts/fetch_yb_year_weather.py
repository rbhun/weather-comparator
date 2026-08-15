#!/usr/bin/env python3
"""Fetch race-window wind fields for each YB edition and attach them to the overlay."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pmc.io.yb import write_overlay  # noqa: E402

LATS = np.round(np.arange(37.5, 44.0 + 0.001, 0.25), 4)
LONS = np.round(np.arange(6.5, 14.5 + 0.001, 0.25), 4)
ARCHIVE = "https://customer-archive-api.open-meteo.com/v1/archive"
MODELS = ("ecmwf_ifs", "era5")


def _load_key() -> str:
    key = os.getenv("OPENMETEO_API_KEY", "").strip()
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENMETEO_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
                if key:
                    os.environ["OPENMETEO_API_KEY"] = key
                    return key
    raise SystemExit("OPENMETEO_API_KEY is required for year weather")


def _race_window(edition: dict) -> tuple[dt.date, dt.date]:
    start = dt.datetime.fromisoformat(edition["start_utc"].replace("Z", "+00:00")).date()
    end_raw = edition.get("end_utc") or edition["start_utc"]
    end = dt.datetime.fromisoformat(end_raw.replace("Z", "+00:00")).date()
    if end < start:
        end = start
    # include the day after last finish so late arrivals are covered
    end = min(end + dt.timedelta(days=1), start + dt.timedelta(days=7))
    return start, end


def _fetch_batch(
    session: requests.Session,
    key: str,
    model: str,
    start: dt.date,
    end: dt.date,
    points: list[tuple[int, int, float, float]],
) -> dict:
    params = {
        "latitude": ",".join(f"{p[2]:.4f}" for p in points),
        "longitude": ",".join(f"{p[3]:.4f}" for p in points),
        "hourly": "wind_speed_10m",
        "wind_speed_unit": "kn",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "models": model,
        "timezone": "UTC",
        "apikey": key,
    }
    last = None
    for attempt in range(5):
        resp = session.get(ARCHIVE, params=params, timeout=90)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in {429, 500, 502, 503, 504}:
            time.sleep(2**attempt + 1)
            last = resp
            continue
        raise RuntimeError(f"Open-Meteo {resp.status_code}: {resp.text[:240]}")
    raise RuntimeError(f"Open-Meteo failed after retries: {last.status_code if last else 'no response'}")


def fetch_year_field(session: requests.Session, key: str, start: dt.date, end: dt.date) -> dict:
    points = [
        (i, j, float(lat), float(lon))
        for i, lat in enumerate(LATS)
        for j, lon in enumerate(LONS)
    ]
    mean = np.full((len(LATS), len(LONS)), np.nan, dtype=np.float32)
    calm = np.full((len(LATS), len(LONS)), np.nan, dtype=np.float32)
    used_model = None
    chunk = 80
    for model in MODELS:
        try:
            for offset in range(0, len(points), chunk):
                batch = points[offset : offset + chunk]
                payload = _fetch_batch(session, key, model, start, end, batch)
                rows = payload if isinstance(payload, list) else [payload]
                if len(rows) != len(batch):
                    raise RuntimeError(f"expected {len(batch)} series, got {len(rows)}")
                for (i, j, _lat, _lon), row in zip(batch, rows):
                    hourly = (row or {}).get("hourly") or {}
                    speeds = hourly.get("wind_speed_10m") or []
                    vals = [float(v) for v in speeds if v is not None]
                    if not vals:
                        continue
                    arr = np.asarray(vals, dtype=np.float32)
                    mean[i, j] = float(arr.mean())
                    calm[i, j] = float(np.mean(arr < 5.0))
                time.sleep(0.35)
            used_model = model
            break
        except Exception as exc:
            print(f"  model {model} failed: {exc}", flush=True)
            mean[:] = np.nan
            calm[:] = np.nan
            continue
    if used_model is None:
        return {"available": False, "note": "Open-Meteo archive request failed for this year."}

    def _grid(arr: np.ndarray) -> list[list[float | None]]:
        out = []
        for row in arr:
            out.append([None if not np.isfinite(v) else round(float(v), 2) for v in row])
        return out

    return {
        "available": True,
        "model": used_model,
        "resolution_deg": 0.25,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "lat": [float(v) for v in LATS],
        "lon": [float(v) for v in LONS],
        "mean_tws_kt": _grid(mean),
        "p_below_5kt": _grid(calm),
        "note": f"Race-window mean 10 m wind from Open-Meteo {used_model} ({start} to {end}).",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="dashboard/yb_results.json")
    parser.add_argument("--js", default="dashboard/yb_results.js")
    args = parser.parse_args()

    overlay = json.loads(Path(args.json).read_text(encoding="utf-8"))
    key = _load_key()
    session = requests.Session()
    for edition in overlay["editions"]:
        start, end = _race_window(edition)
        print(f"[weather] {edition['year']} {start} -> {end}", flush=True)
        edition["weather"] = fetch_year_field(session, key, start, end)
        n = sum(
            1
            for row in edition["weather"].get("mean_tws_kt") or []
            for v in row
            if v is not None
        )
        print(f"  cells={n} model={edition['weather'].get('model')}", flush=True)
    write_overlay(overlay, Path(args.json), Path(args.js))
    print(f"[weather] updated {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
