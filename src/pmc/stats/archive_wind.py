"""Fetch IFS analysis u/v for YB sample points via Open-Meteo archive API."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import xarray as xr

from pmc.io.units import assert_hourly_units, extract_responses

ARCHIVE_CUSTOMER = "https://customer-archive-api.open-meteo.com/v1/archive"
ARCHIVE_PUBLIC = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_CACHE = Path("data/cache/yb_wind_check_archive")
BATCH_SIZE = 40


def _api_key() -> str | None:
    key = os.getenv("OPENMETEO_API_KEY", "").strip()
    if key:
        return key
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENMETEO_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
                return key or None
    return None


def _http_get_json(url: str, timeout: int = 120) -> Any:
    req = Request(url, headers={"User-Agent": "pmc-yb-wind-check/0.1"})
    last: Exception | None = None
    for attempt in range(5):
        try:
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if isinstance(exc, HTTPError) and exc.code in {400, 404}:
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="replace")[:240]
                except Exception:
                    pass
                raise RuntimeError(f"Open-Meteo HTTP {exc.code}: {body}") from exc
            time.sleep(1.5 * (2**attempt))
    raise RuntimeError(f"Open-Meteo archive failed: {last}") from last


def _cache_path(cache_root: Path, model: str, start: date, end: date, points: Sequence[tuple[float, float]]) -> Path:
    digest = hashlib.sha1(
        (
            f"{model}|{start.isoformat()}|{end.isoformat()}|"
            + ";".join(f"{lat:.2f},{lon:.2f}" for lat, lon in points)
        ).encode()
    ).hexdigest()
    return cache_root / f"batch_{digest}.json"


def fetch_batch_series(
    points: Sequence[tuple[float, float]],
    start: date,
    end: date,
    *,
    cache_root: Path = DEFAULT_CACHE,
    model: str = "ecmwf_ifs",
) -> list[pd.DataFrame]:
    """Hourly u/v (m/s) for a batch of lat/lon points; disk-cached."""

    if not points:
        return []
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(cache_root, model, start, end, points)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        params = {
            "latitude": ",".join(f"{lat:.4f}" for lat, _ in points),
            "longitude": ",".join(f"{lon:.4f}" for _, lon in points),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": "wind_u_component_10m,wind_v_component_10m",
            "wind_speed_unit": "ms",
            "models": model,
            "timezone": "UTC",
        }
        key_val = _api_key()
        base = ARCHIVE_CUSTOMER if key_val else ARCHIVE_PUBLIC
        if key_val:
            params["apikey"] = key_val
        url = f"{base}?{urlencode(params)}"
        payload = _http_get_json(url)
        assert_hourly_units(
            payload,
            expected={
                "wind_u_component_10m": "m/s",
                "wind_v_component_10m": "m/s",
            },
            context=f"yb_wind_check batch n={len(points)}",
        )
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        time.sleep(0.25)

    responses = extract_responses(payload)
    if len(responses) != len(points):
        # Single-point responses sometimes unwrap to one dict.
        if len(points) == 1 and len(responses) == 1:
            pass
        else:
            raise RuntimeError(
                f"Expected {len(points)} archive responses, got {len(responses)}"
            )

    frames: list[pd.DataFrame] = []
    for (lat, lon), response in zip(points, responses):
        hourly = response.get("hourly") or {}
        times = hourly.get("time") or []
        u = hourly.get("wind_u_component_10m") or []
        v = hourly.get("wind_v_component_10m") or []
        if not times or len(times) != len(u) or len(times) != len(v):
            raise RuntimeError(f"Malformed hourly series for {lat},{lon}")
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(times, utc=True).tz_localize(None),
                "u10": np.asarray(u, dtype=float),
                "v10": np.asarray(v, dtype=float),
            }
        )
        frame["lat"] = float(lat)
        frame["lon"] = float(lon)
        frames.append(frame)
    return frames


def build_wind_dataset_for_samples(
    samples: pd.DataFrame,
    *,
    resolution: float = 0.1,
    cache_root: Path = DEFAULT_CACHE,
    model: str = "ecmwf_ifs",
    pad_days: int = 0,
    batch_size: int = BATCH_SIZE,
) -> xr.Dataset:
    """Build a C1-like Dataset covering sample cells/times via batched archive pulls."""

    if samples.empty:
        raise ValueError("No samples to fetch wind for")

    frame = samples.copy()
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
    frame["lat_r"] = (np.round(frame["lat"] / resolution) * resolution).astype(float)
    frame["lon_r"] = (np.round(frame["lon"] / resolution) * resolution).astype(float)
    frame["year"] = frame["time_utc"].dt.year

    series_list: list[pd.DataFrame] = []
    for year, year_frame in frame.groupby("year"):
        t_min = year_frame["time_utc"].min().tz_convert("UTC") - pd.Timedelta(days=pad_days)
        t_max = year_frame["time_utc"].max().tz_convert("UTC") + pd.Timedelta(days=pad_days)
        start = t_min.date()
        end = t_max.date()
        if end < start:
            end = start
        # Cap absurd windows
        if (end - start).days > 14:
            end = start + timedelta(days=14)

        cells = (
            year_frame.groupby(["lat_r", "lon_r"], as_index=False)
            .size()
            .loc[:, ["lat_r", "lon_r"]]
        )
        points = [(float(r.lat_r), float(r.lon_r)) for r in cells.itertuples(index=False)]
        print(
            f"[archive] year={year} window={start}→{end} cells={len(points)}",
            flush=True,
        )
        for offset in range(0, len(points), batch_size):
            batch = points[offset : offset + batch_size]
            print(
                f"[archive]   batch {offset // batch_size + 1}/"
                f"{(len(points) + batch_size - 1) // batch_size} n={len(batch)}",
                flush=True,
            )
            series_list.extend(
                fetch_batch_series(
                    batch,
                    start,
                    end,
                    cache_root=cache_root,
                    model=model,
                )
            )

    all_series = pd.concat(series_list, ignore_index=True)
    all_series["lat"] = all_series["lat"].astype(float).round(4)
    all_series["lon"] = all_series["lon"].astype(float).round(4)
    lats = np.array(sorted(all_series["lat"].unique()), dtype=np.float64)
    lons = np.array(sorted(all_series["lon"].unique()), dtype=np.float64)
    times = np.array(sorted(all_series["time"].unique()), dtype="datetime64[ns]")
    lat_index = {round(float(v), 4): i for i, v in enumerate(lats)}
    lon_index = {round(float(v), 4): i for i, v in enumerate(lons)}
    time_index = {np.datetime64(t, "ns"): i for i, t in enumerate(times)}

    u10 = np.full((times.size, lats.size, lons.size), np.nan, dtype=np.float32)
    v10 = np.full((times.size, lats.size, lons.size), np.nan, dtype=np.float32)
    for row in all_series.itertuples(index=False):
        ti = time_index[np.datetime64(row.time, "ns")]
        i = lat_index[round(float(row.lat), 4)]
        j = lon_index[round(float(row.lon), 4)]
        u10[ti, i, j] = row.u10
        v10[ti, i, j] = row.v10

    return xr.Dataset(
        data_vars={
            "u10": (("time", "lat", "lon"), u10),
            "v10": (("time", "lat", "lon"), v10),
        },
        coords={
            "time": times,
            "lat": lats.astype(np.float32),
            "lon": lons.astype(np.float32),
        },
        attrs={
            "source": f"ifs_analysis_archive:{model}",
            "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "api_version": "open-meteo-archive",
            "omissions": [],
        },
    )
