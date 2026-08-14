"""Open-Meteo fetcher with disk cache and resume support.

This module intentionally focuses on Cluster D ownership (`src/pmc/io`).
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import requests
import xarray as xr
import yaml


DEFAULT_DOMAIN = {
    "lat_min": 37.5,
    "lat_max": 44.0,
    "lon_min": 6.5,
    "lon_max": 14.5,
    "resolution": 0.1,
}

DEFAULT_DOMAIN_PATH = Path("config/domain.yaml")
DEFAULT_MODELS_PATH = Path("config/models.yaml")
DEFAULT_CACHE_ROOT = Path("data/cache/openmeteo")
DEFAULT_WIND_ROOT = Path("data/wind")
DEFAULT_LOG_ROOT = Path("data/wind/logs")


@dataclass(frozen=True)
class Domain:
    """Spatial domain and grid resolution."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    resolution: float = 0.1

    def latitudes(self) -> np.ndarray:
        return _inclusive_axis(self.lat_min, self.lat_max, self.resolution)

    def longitudes(self) -> np.ndarray:
        return _inclusive_axis(self.lon_min, self.lon_max, self.resolution)


@dataclass
class FetchSummary:
    """Run summary persisted per fetch."""

    source: str
    mode: str
    model: str
    start_date: str
    end_date: str
    output_path: str
    endpoint: str
    wall_seconds: float
    days_total: int
    days_fetched: int
    days_resumed: int
    request_count: int
    cache_hits: int
    cache_misses: int
    cells_fetched: int
    cells_missing: int


@dataclass(frozen=True)
class ModeConfig:
    mode: str
    endpoint: str
    candidates: tuple[str, ...]
    source_label: str


MODE_CONFIG: dict[str, ModeConfig] = {
    "analysis": ModeConfig(
        mode="analysis",
        endpoint="https://historical-forecast-api.open-meteo.com/v1/forecast",
        candidates=(
            "ecmwf_ifs",
            "ecmwf_ifs025",
            "ecmwf_ifs04",
            "ifs",
        ),
        source_label="ifs_analysis_9km",
    ),
    "previous_runs": ModeConfig(
        mode="previous_runs",
        endpoint="https://previous-runs-api.open-meteo.com/v1/forecast",
        candidates=(
            "ecmwf_ifs",
            "ecmwf_aifs025",
            "gfs_global",
            "icon_global",
            "gem_global",
            "ukmo_global",
            "arpege_europe",
        ),
        source_label="previous_runs",
    ),
    "live": ModeConfig(
        mode="live",
        endpoint="https://api.open-meteo.com/v1/forecast",
        candidates=(
            "ecmwf_ifs",
            "ecmwf_aifs025",
            "gfs_global",
            "icon_global",
            "gem_global",
            "ukmo_global",
            "arpege_europe",
        ),
        source_label="live_forecast",
    ),
}


def load_domain(path: Path = DEFAULT_DOMAIN_PATH) -> Domain:
    """Load grid domain from config, fallback to SPEC defaults."""

    if path.exists():
        parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for key, value in DEFAULT_DOMAIN.items():
            parsed.setdefault(key, value)
        return Domain(
            lat_min=float(parsed["lat_min"]),
            lat_max=float(parsed["lat_max"]),
            lon_min=float(parsed["lon_min"]),
            lon_max=float(parsed["lon_max"]),
            resolution=float(parsed["resolution"]),
        )

    return Domain(
        lat_min=DEFAULT_DOMAIN["lat_min"],
        lat_max=DEFAULT_DOMAIN["lat_max"],
        lon_min=DEFAULT_DOMAIN["lon_min"],
        lon_max=DEFAULT_DOMAIN["lon_max"],
        resolution=DEFAULT_DOMAIN["resolution"],
    )


class DiskRequestCache:
    """Simple disk cache keyed by endpoint+params."""

    def __init__(self, root: Path = DEFAULT_CACHE_ROOT) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, endpoint: str, params: Mapping[str, Any]) -> dict[str, Any] | None:
        path = self._cache_path(endpoint, params)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, endpoint: str, params: Mapping[str, Any], payload: dict[str, Any]) -> None:
        path = self._cache_path(endpoint, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _cache_path(self, endpoint: str, params: Mapping[str, Any]) -> Path:
        canonical = json.dumps(
            {"endpoint": endpoint, "params": _canonicalize(params)},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"


class OpenMeteoFetcher:
    """Fetches wind grids and writes C1-like zarr stores."""

    def __init__(
        self,
        api_key: str | None,
        cache_root: Path = DEFAULT_CACHE_ROOT,
        output_root: Path = DEFAULT_WIND_ROOT,
        max_workers: int = 8,
        batch_size: int = 180,
        timeout_seconds: int = 60,
        retries: int = 6,
        inter_day_delay_seconds: float = 0.0,
        min_request_interval_seconds: float = 1.0,
    ) -> None:
        self.api_key = api_key
        self.cache = DiskRequestCache(cache_root)
        self.output_root = output_root
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.inter_day_delay_seconds = inter_day_delay_seconds
        self.min_request_interval_seconds = min_request_interval_seconds
        self.session = requests.Session()
        self._request_lock = threading.Lock()
        self._last_request_at = 0.0

        self.cache_hits = 0
        self.cache_misses = 0
        self.request_count = 0

    def fetch_wind(
        self,
        source: str,
        start: dt.date,
        end: dt.date,
        cfg: Domain,
        output_path: Path | None = None,
        force_model: str | None = None,
        refresh_models: bool = False,
        month_filter: set[int] | None = None,
    ) -> tuple[Path, FetchSummary]:
        """Fetch a wind store for one source and date range."""

        if source not in MODE_CONFIG:
            raise ValueError(
                f"Unknown source '{source}'. Expected one of {sorted(MODE_CONFIG)}."
            )
        if start > end:
            raise ValueError("start must be <= end")

        mode_cfg = MODE_CONFIG[source]
        mode = mode_cfg.mode
        endpoint = self._endpoint_for_mode(mode_cfg)

        discovered = self._load_cached_models(mode)
        if not discovered or refresh_models:
            discovered = self.discover_models(mode, start, end, mode_cfg, cfg)
        else:
            print(
                f"[fetch] using cached discovered models for {mode}: {discovered}",
                flush=True,
            )
        if force_model:
            if force_model not in discovered:
                raise ValueError(
                    f"Requested model '{force_model}' was not discovered for mode '{mode}'. "
                    f"Discovered models: {discovered}"
                )
            model = force_model
        else:
            model = self._select_primary_model(mode, discovered)

        target = output_path or (self.output_root / f"{source}.zarr")
        target.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_LOG_ROOT.mkdir(parents=True, exist_ok=True)

        started_at = time.time()
        summary = self._fetch_grid(
            endpoint=endpoint,
            mode_cfg=mode_cfg,
            model=model,
            start=start,
            end=end,
            cfg=cfg,
            output_path=target,
            month_filter=month_filter,
        )
        summary.wall_seconds = time.time() - started_at
        summary.output_path = str(target)

        summary_path = DEFAULT_LOG_ROOT / (
            f"{source}_{start.isoformat()}_{end.isoformat()}_{int(time.time())}.json"
        )
        summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        return target, summary

    def discover_models(
        self,
        mode: str,
        start: dt.date,
        end: dt.date,
        mode_cfg: ModeConfig,
        cfg: Domain,
    ) -> list[str]:
        """Probe candidate models and persist survivors to config/models.yaml."""

        endpoint = self._endpoint_for_mode(mode_cfg)
        sample_start = max(start, end - dt.timedelta(days=2))
        sample_end = sample_start + dt.timedelta(days=1)
        if sample_end > end:
            sample_end = end

        sample_lat = round((cfg.lat_min + cfg.lat_max) / 2.0, 3)
        sample_lon = round((cfg.lon_min + cfg.lon_max) / 2.0, 3)

        survivors: list[str] = []
        for candidate in mode_cfg.candidates:
            params = {
                "latitude": sample_lat,
                "longitude": sample_lon,
                "start_date": sample_start.isoformat(),
                "end_date": sample_end.isoformat(),
                "hourly": "wind_speed_10m,wind_direction_10m",
                "wind_speed_unit": "ms",
                "timezone": "UTC",
                "models": candidate,
            }
            try:
                payload = self._get_json(endpoint=endpoint, params=params)
            except requests.RequestException:
                continue
            except ValueError:
                continue

            if _has_hourly_values(payload):
                survivors.append(candidate)

        if not survivors:
            raise RuntimeError(
                f"Model discovery failed for mode '{mode}'. "
                "No candidate model returned hourly data."
            )

        self._write_models_yaml(mode, endpoint, survivors)
        return survivors

    def _fetch_grid(
        self,
        endpoint: str,
        mode_cfg: ModeConfig,
        model: str,
        start: dt.date,
        end: dt.date,
        cfg: Domain,
        output_path: Path,
        month_filter: set[int] | None,
    ) -> FetchSummary:
        lat = cfg.latitudes()
        lon = cfg.longitudes()
        all_points = _grid_points(lat, lon)
        point_batches = list(_chunked(all_points, self.batch_size))

        existing_days = self._existing_days(output_path)
        scheduled_days = list(_iter_days(start, end, month_filter=month_filter))
        days_total = len(scheduled_days)
        days_resumed = 0
        days_fetched = 0
        cells_fetched = 0
        cells_missing = 0

        print(
            "[fetch] "
            f"mode={mode_cfg.mode} model={model} days={days_total} "
            f"grid_points={len(all_points)} batches_per_day={len(point_batches)}"
            ,
            flush=True,
        )
        for day_start in scheduled_days:
            if day_start in existing_days:
                days_resumed += 1
                print(
                    f"[fetch] resume-skip {day_start.isoformat()} (already in store)",
                    flush=True,
                )
                continue

            day_end = day_start
            day_started_at = time.time()
            day_ds, fetched, missing = self._fetch_day_dataset(
                endpoint=endpoint,
                model=model,
                day_start=day_start,
                day_end=day_end,
                lat_axis=lat,
                lon_axis=lon,
                point_batches=point_batches,
                source_attr=self._source_attr(mode_cfg.mode, model),
            )
            self._append_day(output_path, day_ds)
            days_fetched += 1
            cells_fetched += fetched
            cells_missing += missing
            day_elapsed = time.time() - day_started_at
            total_req = max(1, self.request_count)
            cache_ratio = self.cache_hits / total_req
            print(
                "[fetch] "
                f"fetched {day_start.isoformat()} in {day_elapsed:.1f}s "
                f"fetched_cells={fetched} missing_cells={missing} "
                f"requests={self.request_count} cache_hit_ratio={cache_ratio:.3f}"
                ,
                flush=True,
            )
            if self.inter_day_delay_seconds > 0:
                time.sleep(self.inter_day_delay_seconds)

        return FetchSummary(
            source=mode_cfg.mode,
            mode=mode_cfg.mode,
            model=model,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            output_path=str(output_path),
            endpoint=endpoint,
            wall_seconds=0.0,
            days_total=days_total,
            days_fetched=days_fetched,
            days_resumed=days_resumed,
            request_count=self.request_count,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            cells_fetched=cells_fetched,
            cells_missing=cells_missing,
        )

    def _fetch_day_dataset(
        self,
        endpoint: str,
        model: str,
        day_start: dt.date,
        day_end: dt.date,
        lat_axis: np.ndarray,
        lon_axis: np.ndarray,
        point_batches: list[list[tuple[int, int, float, float]]],
        source_attr: str,
    ) -> tuple[xr.Dataset, int, int]:
        day_t0 = pd.Timestamp(day_start, tz="UTC")
        times = pd.date_range(day_t0, day_t0 + pd.Timedelta(hours=23), freq="h")
        n_time = len(times)

        u10 = np.full((n_time, lat_axis.size, lon_axis.size), np.nan, dtype=np.float32)
        v10 = np.full((n_time, lat_axis.size, lon_axis.size), np.nan, dtype=np.float32)

        futures: list[concurrent.futures.Future[list[dict[str, Any]]]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for batch in point_batches:
                futures.append(
                    pool.submit(
                        self._fetch_batch_resilient,
                        endpoint,
                        model,
                        day_start,
                        day_end,
                        batch,
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                payloads = future.result()
                for payload in payloads:
                    responses = _extract_responses(payload)
                    for response in responses:
                        lat_v = float(response.get("latitude"))
                        lon_v = float(response.get("longitude"))
                        lat_idx = int(
                            round((lat_v - float(lat_axis[0])) / float(lat_axis[1] - lat_axis[0]))
                        )
                        lon_idx = int(
                            round((lon_v - float(lon_axis[0])) / float(lon_axis[1] - lon_axis[0]))
                        )
                        if lat_idx < 0 or lat_idx >= lat_axis.size:
                            continue
                        if lon_idx < 0 or lon_idx >= lon_axis.size:
                            continue

                        hourly = response.get("hourly") or {}
                        speed = np.asarray(hourly.get("wind_speed_10m", []), dtype=float)
                        direction = np.asarray(hourly.get("wind_direction_10m", []), dtype=float)
                        if speed.size == 0 or direction.size == 0:
                            continue
                        n = min(n_time, speed.size, direction.size)
                        u10_vals, v10_vals = _wind_speed_dir_to_uv(speed[:n], direction[:n])
                        u10[:n, lat_idx, lon_idx] = u10_vals.astype(np.float32)
                        v10[:n, lat_idx, lon_idx] = v10_vals.astype(np.float32)

        finite = np.isfinite(u10) & np.isfinite(v10)
        cells_fetched = int(np.count_nonzero(finite))
        cells_total = int(np.prod(u10.shape))
        cells_missing = cells_total - cells_fetched

        ds = xr.Dataset(
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
                "source": source_attr,
                "fetched_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
                "api_version": "open-meteo",
                "omissions": "[]",
            },
        )
        return ds, cells_fetched, cells_missing

    def _fetch_batch(
        self,
        endpoint: str,
        model: str,
        day_start: dt.date,
        day_end: dt.date,
        batch: list[tuple[int, int, float, float]],
    ) -> dict[str, Any]:
        # Jitter reduces synchronized bursts against the API.
        time.sleep(random.uniform(0.0, 0.25))
        latitudes = ",".join(f"{p[2]:.3f}" for p in batch)
        longitudes = ",".join(f"{p[3]:.3f}" for p in batch)
        params = {
            "latitude": latitudes,
            "longitude": longitudes,
            "start_date": day_start.isoformat(),
            "end_date": day_end.isoformat(),
            "hourly": "wind_speed_10m,wind_direction_10m",
            "wind_speed_unit": "ms",
            "timezone": "UTC",
            "models": model,
        }
        return self._get_json(endpoint=endpoint, params=params)

    def _fetch_batch_resilient(
        self,
        endpoint: str,
        model: str,
        day_start: dt.date,
        day_end: dt.date,
        batch: list[tuple[int, int, float, float]],
    ) -> list[dict[str, Any]]:
        pending = [batch]
        payloads: list[dict[str, Any]] = []
        while pending:
            current = pending.pop()
            try:
                payloads.append(
                    self._fetch_batch(
                        endpoint=endpoint,
                        model=model,
                        day_start=day_start,
                        day_end=day_end,
                        batch=current,
                    )
                )
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status == 414 and len(current) > 1:
                    midpoint = len(current) // 2
                    print(
                        f"[fetch] split batch size={len(current)} due to 414 URI too large",
                        flush=True,
                    )
                    pending.append(current[midpoint:])
                    pending.append(current[:midpoint])
                    continue
                raise
        return payloads

    def _get_json(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        cache_key_params = dict(params)
        cache_key_params["_auth"] = "key" if self.api_key else "anon"
        cached = self.cache.get(endpoint, cache_key_params)
        if cached is not None:
            self.cache_hits += 1
            self.request_count += 1
            return cached

        request_params = dict(params)
        if self.api_key:
            request_params["apikey"] = self.api_key

        transient_statuses = {429, 500, 502, 503, 504}
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                self._wait_for_request_slot()
                response = self.session.get(
                    endpoint,
                    params=request_params,
                    timeout=self.timeout_seconds,
                )
                if response.status_code in transient_statuses:
                    raise requests.HTTPError(
                        f"Transient status {response.status_code}",
                        response=response,
                    )
                response.raise_for_status()
                payload = response.json()
                self.cache.put(endpoint, cache_key_params, payload)
                self.cache_misses += 1
                self.request_count += 1
                return payload
            except requests.HTTPError as exc:
                last_exc = exc
                status = exc.response.status_code if exc.response is not None else None
                if status not in transient_statuses:
                    raise
                delay = float(2**attempt)
                if status == 429:
                    delay = max(delay, 10.0)
                retry_after = None
                if exc.response is not None:
                    retry_after = exc.response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                if status == 429 and exc.response is not None:
                    try:
                        body = exc.response.json()
                    except ValueError:
                        body = {}
                    reason = str(body.get("reason", "")).lower()
                    if "one minute" in reason or "next minute" in reason or "minutely" in reason:
                        now_utc = dt.datetime.now(dt.timezone.utc)
                        next_minute = (
                            now_utc.replace(second=0, microsecond=0)
                            + dt.timedelta(minutes=1)
                        )
                        until_minute = (next_minute - now_utc).total_seconds() + 2.0
                        delay = max(delay, until_minute)
                    if "next hour" in reason:
                        now_utc = dt.datetime.now(dt.timezone.utc)
                        next_hour = (
                            now_utc.replace(minute=0, second=0, microsecond=0)
                            + dt.timedelta(hours=1)
                        )
                        until_hour = (next_hour - now_utc).total_seconds() + 2.0
                        delay = max(delay, until_hour)
                delay += random.uniform(0.0, 0.5)
                print(
                    "[fetch] retry "
                    f"attempt={attempt + 1}/{self.retries} "
                    f"delay={delay:.1f}s reason={type(exc).__name__} status={status}"
                    ,
                    flush=True,
                )
                time.sleep(delay)
                continue
            except (requests.ConnectionError, requests.Timeout, ValueError) as exc:
                last_exc = exc
                delay = float(2**attempt) + random.uniform(0.0, 0.5)
                print(
                    "[fetch] retry "
                    f"attempt={attempt + 1}/{self.retries} "
                    f"delay={delay:.1f}s reason={type(exc).__name__}",
                    flush=True,
                )
                time.sleep(delay)
                continue
            except requests.RequestException:
                raise

        assert last_exc is not None
        raise last_exc

    def _wait_for_request_slot(self) -> None:
        if self.min_request_interval_seconds <= 0:
            return
        with self._request_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_at
            if elapsed < self.min_request_interval_seconds:
                time.sleep(self.min_request_interval_seconds - elapsed)
            self._last_request_at = time.monotonic()

    def _existing_days(self, path: Path) -> set[dt.date]:
        if not path.exists():
            return set()
        ds = xr.open_zarr(path)
        try:
            values = pd.to_datetime(ds["time"].values, utc=True)
            return {ts.date() for ts in values}
        finally:
            ds.close()

    def _append_day(self, path: Path, day_ds: xr.Dataset) -> None:
        time_chunk = min(720, day_ds.sizes["time"])
        lat_chunk = min(32, day_ds.sizes["lat"])
        lon_chunk = min(40, day_ds.sizes["lon"])
        encoding = {
            "u10": {"chunks": (time_chunk, lat_chunk, lon_chunk)},
            "v10": {"chunks": (time_chunk, lat_chunk, lon_chunk)},
        }
        if not path.exists():
            day_ds.to_zarr(path, mode="w", encoding=encoding, consolidated=True, zarr_version=2)
            return
        day_ds.to_zarr(
            path,
            mode="a",
            append_dim="time",
            consolidated=True,
            zarr_version=2,
        )

    def _select_primary_model(self, mode: str, models: list[str]) -> str:
        if mode == "analysis":
            priority = ("ecmwf_ifs", "ecmwf_ifs025", "ecmwf_ifs04", "ifs")
            for name in priority:
                if name in models:
                    return name
        return models[0]

    def _source_attr(self, mode: str, model: str) -> str:
        if mode == "analysis":
            return "ifs_analysis_9km"
        if mode == "previous_runs":
            return f"forecast:{model}:lead1-7d"
        return f"forecast:{model}:live"

    def _endpoint_for_mode(self, mode_cfg: ModeConfig) -> str:
        if self.api_key:
            if "historical-forecast-api" in mode_cfg.endpoint:
                return mode_cfg.endpoint.replace(
                    "historical-forecast-api",
                    "customer-historical-weather-api",
                )
            if "previous-runs-api" in mode_cfg.endpoint:
                return mode_cfg.endpoint.replace(
                    "previous-runs-api",
                    "customer-previous-runs-api",
                )
            if "api.open-meteo.com" in mode_cfg.endpoint:
                return mode_cfg.endpoint.replace("api.open-meteo.com", "customer-api.open-meteo.com")
        return mode_cfg.endpoint

    def _write_models_yaml(self, mode: str, endpoint: str, models: list[str]) -> None:
        DEFAULT_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing: dict[str, Any] = {}
        if DEFAULT_MODELS_PATH.exists():
            existing = yaml.safe_load(DEFAULT_MODELS_PATH.read_text(encoding="utf-8")) or {}
        existing[mode] = {
            "endpoint": endpoint,
            "models": models,
            "updated_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        }
        DEFAULT_MODELS_PATH.write_text(
            yaml.safe_dump(existing, sort_keys=True),
            encoding="utf-8",
        )

    def _load_cached_models(self, mode: str) -> list[str]:
        if not DEFAULT_MODELS_PATH.exists():
            return []
        parsed = yaml.safe_load(DEFAULT_MODELS_PATH.read_text(encoding="utf-8")) or {}
        block = parsed.get(mode) or {}
        models = block.get("models")
        if isinstance(models, list):
            return [str(v) for v in models]
        return []


def fetch_wind(source: str, start: dt.date, end: dt.date, cfg: Domain) -> Path:
    """Contract C8 entrypoint for the IO module."""

    fetcher = OpenMeteoFetcher(api_key=_read_api_key())
    path, _summary = fetcher.fetch_wind(source=source, start=start, end=end, cfg=cfg)
    return path


def _read_api_key() -> str | None:
    from os import getenv

    return getenv("OPENMETEO_API_KEY")


def _inclusive_axis(start: float, stop: float, step: float) -> np.ndarray:
    values = np.arange(start, stop + (step * 0.5), step, dtype=np.float64)
    return np.round(values, 6)


def _grid_points(lat_axis: np.ndarray, lon_axis: np.ndarray) -> list[tuple[int, int, float, float]]:
    points: list[tuple[int, int, float, float]] = []
    for lat_idx, lat_value in enumerate(lat_axis):
        for lon_idx, lon_value in enumerate(lon_axis):
            points.append((lat_idx, lon_idx, float(lat_value), float(lon_value)))
    return points


def _chunked(values: list[Any], chunk_size: int) -> Iterable[list[Any]]:
    for i in range(0, len(values), chunk_size):
        yield values[i : i + chunk_size]


def _iter_days(start: dt.date, end: dt.date, month_filter: set[int] | None) -> Iterable[dt.date]:
    current = start
    while current <= end:
        if month_filter is None or current.month in month_filter:
            yield current
        current += dt.timedelta(days=1)


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _canonicalize(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    return value


def _extract_responses(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _has_hourly_values(payload: dict[str, Any]) -> bool:
    for response in _extract_responses(payload):
        hourly = response.get("hourly") or {}
        speed = hourly.get("wind_speed_10m")
        direction = hourly.get("wind_direction_10m")
        if isinstance(speed, list) and isinstance(direction, list) and speed and direction:
            return True
    return False


def _wind_speed_dir_to_uv(speed_ms: np.ndarray, direction_from_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(direction_from_deg)
    u10 = -speed_ms * np.sin(radians)
    v10 = -speed_ms * np.cos(radians)
    return u10, v10

