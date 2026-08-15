"""Fill missing analysis-august cells from cache, then fetch only still-empty cells."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pmc.io.openmeteo import (  # noqa: E402
    DEFAULT_HOURLY,
    DiskRequestCache,
    OpenMeteoFetcher,
    _chunked,
    _grid_points,
    _filter_batches_by_indices,
    load_api_key,
    load_domain,
)

PUBLIC_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
CUSTOMER_ARCHIVE = "https://customer-archive-api.open-meteo.com/v1/archive"
WIND_PATH = ROOT / "data/wind/analysis-august.zarr"
MODEL = "ecmwf_ifs"
BATCH_SIZE = 180


def _params(batch, start: dt.date, end: dt.date, auth_mode: str) -> dict:
    return {
        "latitude": ",".join(f"{p[2]:.3f}" for p in batch),
        "longitude": ",".join(f"{p[3]:.3f}" for p in batch),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": DEFAULT_HOURLY,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "models": MODEL,
        "_auth": auth_mode,
    }


def _lookup(cache: DiskRequestCache, batch, start: dt.date, end: dt.date):
    keys = (
        (PUBLIC_ARCHIVE, "none"),
        (CUSTOMER_ARCHIVE, "require_key"),
        (CUSTOMER_ARCHIVE, "none"),
        (PUBLIC_ARCHIVE, "require_key"),
    )
    for endpoint, auth in keys:
        payload = cache.get(endpoint, _params(batch, start, end, auth))
        if payload is not None:
            return payload
    return None


def main() -> int:
    domain = load_domain(ROOT / "config/domain.yaml")
    lat_axis = domain.latitudes()
    lon_axis = domain.longitudes()
    points = _grid_points(lat_axis, lon_axis)
    full_batches = list(_chunked(points, BATCH_SIZE))

    existing = xr.open_zarr(WIND_PATH)
    times = pd.to_datetime(existing["time"].values, utc=True)
    u10 = np.array(existing["u10"].values, copy=True)
    v10 = np.array(existing["v10"].values, copy=True)
    source = str(existing.attrs.get("source", "ifs_analysis_9km"))
    existing.close()

    valid_any = np.any(np.isfinite(u10) & np.isfinite(v10), axis=0)
    sea_filter = {
        (i, j)
        for i in range(valid_any.shape[0])
        for j in range(valid_any.shape[1])
        if bool(valid_any[i, j])
    }
    filtered_batches = _filter_batches_by_indices(full_batches, sea_filter)

    cache = DiskRequestCache(ROOT / "data/cache/openmeteo")
    fetcher = OpenMeteoFetcher(api_key=load_api_key(), cache_root=ROOT / "data/cache/openmeteo")
    fetcher._hourly = DEFAULT_HOURLY
    fetcher._auth_mode = "require_key"

    cache_hits = 0
    cache_misses = 0
    for year in range(2017, 2026):
        start = dt.date(year, 8, 1)
        end = dt.date(year, 8, 31)
        year_mask = (times.year == year) & (times.month == 8)
        t_idx = np.flatnonzero(np.asarray(year_mask))
        n_time = int(t_idx.size)
        arrays = {
            "u10": np.full((n_time, lat_axis.size, lon_axis.size), np.nan, dtype=np.float32),
            "v10": np.full((n_time, lat_axis.size, lon_axis.size), np.nan, dtype=np.float32),
        }
        # Keep already-good values so we never discard downloaded data.
        arrays["u10"][:] = u10[t_idx]
        arrays["v10"][:] = v10[t_idx]

        batches = full_batches if year <= 2018 else filtered_batches
        for batch in batches:
            if not batch:
                continue
            payload = _lookup(cache, batch, start, end)
            if payload is None:
                cache_misses += 1
                continue
            cache_hits += 1
            fetcher._apply_payload_to_batch(
                payload=payload,
                batch=batch,
                arrays=arrays,
                n_time=n_time,
            )
        u10[t_idx] = arrays["u10"]
        v10[t_idx] = arrays["v10"]
        still = ~np.any(np.isfinite(arrays["u10"]) & np.isfinite(arrays["v10"]), axis=0)
        print(
            f"[fill] year={year} cache_applied remaining_empty={int(still.sum())}",
            flush=True,
        )

    print(f"[fill] cache_hits={cache_hits} cache_misses={cache_misses}", flush=True)

    # Fetch only cells that are still empty in any year, year by year.
    fetcher._auth_mode = "require_key"
    fetcher._send_api_key = True
    network_requests = 0
    for year in range(2017, 2026):
        start = dt.date(year, 8, 1)
        end = dt.date(year, 8, 31)
        year_mask = (times.year == year) & (times.month == 8)
        t_idx = np.flatnonzero(np.asarray(year_mask))
        n_time = int(t_idx.size)
        empty = [
            (i, j, float(lat_axis[i]), float(lon_axis[j]))
            for i in range(lat_axis.size)
            for j in range(lon_axis.size)
            if not np.any(np.isfinite(u10[t_idx, i, j]) & np.isfinite(v10[t_idx, i, j]))
        ]
        if not empty:
            print(f"[fill] year={year} network_points=0", flush=True)
            continue
        arrays = {"u10": u10[t_idx], "v10": v10[t_idx]}
        for batch in _chunked(empty, BATCH_SIZE):
            # Cache-first for this exact missing-point batch, then network.
            payload = _lookup(cache, batch, start, end)
            if payload is None:
                payload = fetcher._get_json(
                    endpoint=CUSTOMER_ARCHIVE,
                    params={k: v for k, v in _params(batch, start, end, "require_key").items() if k != "_auth"},
                    allow_network=True,
                )
                network_requests += 1
            fetcher._apply_payload_to_batch(
                payload=payload,
                batch=batch,
                arrays=arrays,
                n_time=n_time,
            )
        u10[t_idx] = arrays["u10"]
        v10[t_idx] = arrays["v10"]
        still = int(
            np.count_nonzero(
                ~np.any(np.isfinite(u10[t_idx]) & np.isfinite(v10[t_idx]), axis=0)
            )
        )
        print(
            f"[fill] year={year} fetched_missing={len(empty)} still_empty={still}",
            flush=True,
        )

    print(f"[fill] network_requests={network_requests}", flush=True)

    out = xr.Dataset(
        data_vars={
            "u10": (("time", "lat", "lon"), u10),
            "v10": (("time", "lat", "lon"), v10),
        },
        coords={
            "time": times.tz_convert(None).to_numpy(dtype="datetime64[ns]"),
            "lat": lat_axis.astype(np.float32),
            "lon": lon_axis.astype(np.float32),
        },
        attrs={
            "source": source,
            "fetched_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            "api_version": "open-meteo",
            "omissions": "[]",
        },
    )
    tmp = WIND_PATH.with_name("analysis-august-filled.zarr")
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp)
    time_chunk = min(720, out.sizes["time"])
    encoding = {
        "u10": {"chunks": (time_chunk, min(32, out.sizes["lat"]), min(40, out.sizes["lon"]))},
        "v10": {"chunks": (time_chunk, min(32, out.sizes["lat"]), min(40, out.sizes["lon"]))},
    }
    out.to_zarr(tmp, mode="w", encoding=encoding, consolidated=True, zarr_format=2)

    never = int(np.count_nonzero(~np.any(np.isfinite(u10) & np.isfinite(v10), axis=0)))
    print(f"[fill] wrote {tmp} never_valid_cells={never}", flush=True)

    backup = WIND_PATH.with_name("analysis-august-prefill.zarr")
    import shutil

    if backup.exists():
        shutil.rmtree(backup)
    WIND_PATH.rename(backup)
    tmp.rename(WIND_PATH)
    print(f"[fill] replaced {WIND_PATH} backup={backup}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
