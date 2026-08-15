"""Fetch IFS analysis u/v for YB sample points via Open-Meteo archive API."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
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
MS_TO_KT = 1.9438445


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


def _http_get_json(url: str, timeout: int = 90) -> dict[str, Any]:
    req = Request(url, headers={"User-Agent": "pmc-yb-wind-check/0.1"})
    last: Exception | None = None
    for attempt in range(5):
        try:
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if isinstance(exc, HTTPError) and exc.code in {400, 404}:
                raise
            time.sleep(1.5 * (2**attempt))
    raise RuntimeError(f"Open-Meteo archive failed: {last}") from last


def fetch_point_series(
    lat: float,
    lon: float,
    start: date,
    end: date,
    *,
    cache_root: Path = DEFAULT_CACHE,
    model: str = "ecmwf_ifs",
) -> pd.DataFrame:
    """Hourly u/v (m/s) for one grid point; disk-cached by request params."""

    cache_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(
        f"{model}|{lat:.2f}|{lon:.2f}|{start.isoformat()}|{end.isoformat()}".encode()
    ).hexdigest()
    cache_path = cache_root / f"{key}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        params = {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
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
            context=f"yb_wind_check {lat},{lon}",
        )
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        time.sleep(0.15)

    responses = extract_responses(payload)
    if not responses:
        raise RuntimeError(f"No hourly payload for {lat},{lon}")
    hourly = responses[0].get("hourly") or {}
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
    return frame


def build_wind_dataset_for_samples(
    samples: pd.DataFrame,
    *,
    resolution: float = 0.1,
    cache_root: Path = DEFAULT_CACHE,
    model: str = "ecmwf_ifs",
    pad_days: int = 0,
) -> xr.Dataset:
    """Build a C1-like zarr-compatible Dataset covering sample cells/times."""

    if samples.empty:
        raise ValueError("No samples to fetch wind for")

    frame = samples.copy()
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
    frame["lat_r"] = (np.round(frame["lat"] / resolution) * resolution).astype(float)
    frame["lon_r"] = (np.round(frame["lon"] / resolution) * resolution).astype(float)

    cells = frame.groupby(["lat_r", "lon_r"], as_index=False).agg(
        t_min=("time_utc", "min"),
        t_max=("time_utc", "max"),
    )
    series_list: list[pd.DataFrame] = []
    for row in cells.itertuples(index=False):
        start = (row.t_min.tz_convert("UTC") - pd.Timedelta(days=pad_days)).date()
        end = (row.t_max.tz_convert("UTC") + pd.Timedelta(days=pad_days)).date()
        series_list.append(
            fetch_point_series(
                float(row.lat_r),
                float(row.lon_r),
                start,
                end,
                cache_root=cache_root,
                model=model,
            )
        )
        print(
            f"[archive] {row.lat_r:.2f},{row.lon_r:.2f} {start}→{end}",
            flush=True,
        )

    all_series = pd.concat(series_list, ignore_index=True)
    lats = np.array(sorted(all_series["lat"].unique()), dtype=np.float32)
    lons = np.array(sorted(all_series["lon"].unique()), dtype=np.float32)
    times = np.array(sorted(all_series["time"].unique()), dtype="datetime64[ns]")
    lat_index = {float(v): i for i, v in enumerate(lats)}
    lon_index = {float(v): i for i, v in enumerate(lons)}
    time_index = {np.datetime64(t, "ns"): i for i, t in enumerate(times)}

    u10 = np.full((times.size, lats.size, lons.size), np.nan, dtype=np.float32)
    v10 = np.full((times.size, lats.size, lons.size), np.nan, dtype=np.float32)
    for row in all_series.itertuples(index=False):
        ti = time_index[np.datetime64(row.time, "ns")]
        i = lat_index[float(row.lat)]
        j = lon_index[float(row.lon)]
        u10[ti, i, j] = row.u10
        v10[ti, i, j] = row.v10

    return xr.Dataset(
        data_vars={
            "u10": (("time", "lat", "lon"), u10),
            "v10": (("time", "lat", "lon"), v10),
        },
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={
            "source": f"ifs_analysis_archive:{model}",
            "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "api_version": "open-meteo-archive",
            "omissions": [],
        },
    )
