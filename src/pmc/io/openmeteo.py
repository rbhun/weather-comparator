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
DEFAULT_ENDPOINTS_PATH = Path("config/openmeteo_endpoints.yaml")
DEFAULT_CACHE_ROOT = Path("data/cache/openmeteo")
DEFAULT_WIND_ROOT = Path("data/wind")
DEFAULT_LOG_ROOT = Path("data/wind/logs")
DEFAULT_CHECKPOINT_ROOT = Path("data/wind/checkpoints")


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
    periods_total: int
    periods_fetched: int
    periods_resumed: int
    checkpoint_tasks_total: int
    checkpoint_tasks_done: int
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


@dataclass(frozen=True)
class EndpointConfig:
    endpoint: str
    auth_mode: str


MODE_CONFIG: dict[str, ModeConfig] = {
    "analysis": ModeConfig(
        mode="analysis",
        endpoint="https://archive-api.open-meteo.com/v1/archive",
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
        self._auth_mode = "none"
        self._send_api_key = False

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
        endpoint_cfg = self._endpoint_config_for_mode(mode)
        endpoint = endpoint_cfg.endpoint
        self._auth_mode = endpoint_cfg.auth_mode
        self._send_api_key = self._resolve_auth_mode(endpoint_cfg.auth_mode)
        print(
            f"[fetch] endpoint={endpoint} auth_mode={self._auth_mode}",
            flush=True,
        )

        discovered = self._load_cached_models(mode)
        if not discovered or refresh_models:
            discovered = self.discover_models(mode, start, end, mode_cfg, cfg, endpoint=endpoint)
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
        endpoint: str,
    ) -> list[str]:
        """Probe candidate models and persist survivors to config/models.yaml."""
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
            except RuntimeError:
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
        lat_axis = cfg.latitudes()
        lon_axis = cfg.longitudes()
        all_points = _grid_points(lat_axis, lon_axis)
        point_batches = list(_chunked(all_points, self.batch_size))
        periods = self._build_periods(mode_cfg.mode, start, end, month_filter)
        existing_periods = self._existing_periods(mode_cfg.mode, output_path)

        checkpoint_path = self._checkpoint_path(
            mode=mode_cfg.mode,
            model=model,
            start=start,
            end=end,
            month_filter=month_filter,
        )
        checkpoint = self._load_checkpoint(checkpoint_path)
        task_total = len(point_batches) * len(periods)
        task_done = 0
        periods_resumed = 0
        periods_fetched = 0
        cells_fetched = 0
        cells_missing = 0

        print(
            "[fetch] "
            f"mode={mode_cfg.mode} model={model} periods={len(periods)} "
            f"grid_points={len(all_points)} batches_per_period={len(point_batches)} "
            f"checkpoint={checkpoint_path}",
            flush=True,
        )

        for period_id, period_start, period_end in periods:
            if period_id in checkpoint["periods_written"] or period_id in existing_periods:
                checkpoint["periods_written"].add(period_id)
                periods_resumed += 1
                task_done += len(point_batches)
                print(f"[fetch] resume-skip period={period_id} (already written)", flush=True)
                continue

            period_started_at = time.time()
            period_ds, period_cells_fetched, period_cells_missing = self._fetch_period_dataset(
                endpoint=endpoint,
                model=model,
                period_start=period_start,
                period_end=period_end,
                lat_axis=lat_axis,
                lon_axis=lon_axis,
                point_batches=point_batches,
                source_attr=self._source_attr(mode_cfg.mode, model),
                checkpoint=checkpoint,
                checkpoint_path=checkpoint_path,
                period_id=period_id,
            )
            self._append_period(output_path, period_ds)
            checkpoint["periods_written"].add(period_id)
            self._save_checkpoint(checkpoint_path, checkpoint)

            periods_fetched += 1
            task_done += len(point_batches)
            cells_fetched += period_cells_fetched
            cells_missing += period_cells_missing
            elapsed = time.time() - period_started_at
            total_req = max(1, self.request_count)
            cache_ratio = self.cache_hits / total_req
            print(
                "[fetch] "
                f"fetched period={period_id} in {elapsed:.1f}s fetched_cells={period_cells_fetched} "
                f"missing_cells={period_cells_missing} requests={self.request_count} "
                f"cache_hit_ratio={cache_ratio:.3f}",
                flush=True,
            )

        # Ensure task_done reflects checkpoint even if we resumed from a prior run.
        task_done = len(checkpoint["completed_tasks"])
        return FetchSummary(
            source=mode_cfg.mode,
            mode=mode_cfg.mode,
            model=model,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            output_path=str(output_path),
            endpoint=endpoint,
            wall_seconds=0.0,
            periods_total=len(periods),
            periods_fetched=periods_fetched,
            periods_resumed=periods_resumed,
            checkpoint_tasks_total=task_total,
            checkpoint_tasks_done=task_done,
            request_count=self.request_count,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            cells_fetched=cells_fetched,
            cells_missing=cells_missing,
        )

    def _fetch_period_dataset(
        self,
        endpoint: str,
        model: str,
        period_start: dt.date,
        period_end: dt.date,
        lat_axis: np.ndarray,
        lon_axis: np.ndarray,
        point_batches: list[list[tuple[int, int, float, float]]],
        source_attr: str,
        checkpoint: dict[str, set[str]],
        checkpoint_path: Path,
        period_id: str,
    ) -> tuple[xr.Dataset, int, int]:
        period_t0 = pd.Timestamp(period_start, tz="UTC")
        period_t1 = pd.Timestamp(period_end, tz="UTC") + pd.Timedelta(hours=23)
        times = pd.date_range(period_t0, period_t1, freq="h")
        n_time = len(times)

        u10 = np.full((n_time, lat_axis.size, lon_axis.size), np.nan, dtype=np.float32)
        v10 = np.full((n_time, lat_axis.size, lon_axis.size), np.nan, dtype=np.float32)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_map: dict[
                concurrent.futures.Future[list[dict[str, Any]]],
                tuple[int, list[tuple[int, int, float, float]]],
            ] = {}
            for batch_index, batch in enumerate(point_batches):
                task_id = _task_id(period_id=period_id, batch_index=batch_index)
                if task_id in checkpoint["completed_tasks"]:
                    cached_payloads = self._fetch_batch_resilient(
                        endpoint=endpoint,
                        model=model,
                        period_start=period_start,
                        period_end=period_end,
                        batch=batch,
                        allow_network=False,
                    )
                    self._apply_payloads_to_arrays(
                        payloads=cached_payloads,
                        lat_axis=lat_axis,
                        lon_axis=lon_axis,
                        u10=u10,
                        v10=v10,
                        n_time=n_time,
                    )
                    continue

                future = pool.submit(
                    self._fetch_batch_resilient,
                    endpoint,
                    model,
                    period_start,
                    period_end,
                    batch,
                    True,
                )
                future_map[future] = (batch_index, batch)

            for future in concurrent.futures.as_completed(future_map):
                batch_index, _batch = future_map[future]
                payloads = future.result()
                self._apply_payloads_to_arrays(
                    payloads=payloads,
                    lat_axis=lat_axis,
                    lon_axis=lon_axis,
                    u10=u10,
                    v10=v10,
                    n_time=n_time,
                )
                task_id = _task_id(period_id=period_id, batch_index=batch_index)
                checkpoint["completed_tasks"].add(task_id)
                self._save_checkpoint(checkpoint_path, checkpoint)

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

    def _apply_payloads_to_arrays(
        self,
        payloads: list[dict[str, Any]],
        lat_axis: np.ndarray,
        lon_axis: np.ndarray,
        u10: np.ndarray,
        v10: np.ndarray,
        n_time: int,
    ) -> None:
        lat_step = float(lat_axis[1] - lat_axis[0]) if len(lat_axis) > 1 else 1.0
        lon_step = float(lon_axis[1] - lon_axis[0]) if len(lon_axis) > 1 else 1.0
        for payload in payloads:
            responses = _extract_responses(payload)
            for response in responses:
                lat_v = float(response.get("latitude"))
                lon_v = float(response.get("longitude"))
                lat_idx = int(round((lat_v - float(lat_axis[0])) / lat_step))
                lon_idx = int(round((lon_v - float(lon_axis[0])) / lon_step))
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

    def _fetch_batch(
        self,
        endpoint: str,
        model: str,
        period_start: dt.date,
        period_end: dt.date,
        batch: list[tuple[int, int, float, float]],
        allow_network: bool,
    ) -> dict[str, Any]:
        # Jitter reduces synchronized bursts against the API.
        time.sleep(random.uniform(0.0, 0.25))
        latitudes = ",".join(f"{p[2]:.3f}" for p in batch)
        longitudes = ",".join(f"{p[3]:.3f}" for p in batch)
        params = {
            "latitude": latitudes,
            "longitude": longitudes,
            "start_date": period_start.isoformat(),
            "end_date": period_end.isoformat(),
            "hourly": "wind_speed_10m,wind_direction_10m",
            "wind_speed_unit": "ms",
            "timezone": "UTC",
            "models": model,
        }
        return self._get_json(endpoint=endpoint, params=params, allow_network=allow_network)

    def _fetch_batch_resilient(
        self,
        endpoint: str,
        model: str,
        period_start: dt.date,
        period_end: dt.date,
        batch: list[tuple[int, int, float, float]],
        allow_network: bool = True,
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
                        period_start=period_start,
                        period_end=period_end,
                        batch=current,
                        allow_network=allow_network,
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

    def _get_json(self, endpoint: str, params: dict[str, Any], allow_network: bool = True) -> dict[str, Any]:
        cache_key_params = dict(params)
        cache_key_params["_auth"] = self._auth_mode
        cached = self.cache.get(endpoint, cache_key_params)
        if cached is not None:
            self.cache_hits += 1
            self.request_count += 1
            return cached
        if not allow_network:
            raise RuntimeError(
                "Checkpoint indicates task complete, but no cached payload exists for "
                f"params={params}"
            )

        request_params = dict(params)
        if self._send_api_key:
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
                    reason = _response_reason(exc.response)
                    raise RuntimeError(
                        f"Open-Meteo request failed with status {status}: {reason}"
                    ) from None
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
        if isinstance(last_exc, requests.HTTPError):
            status = last_exc.response.status_code if last_exc.response is not None else "unknown"
            reason = _response_reason(last_exc.response)
            raise RuntimeError(
                f"Open-Meteo request failed after retries with status {status}: {reason}"
            ) from None
        raise RuntimeError(
            f"Open-Meteo request failed after retries: {type(last_exc).__name__}"
        ) from None

    def _wait_for_request_slot(self) -> None:
        if self.min_request_interval_seconds <= 0:
            return
        with self._request_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_at
            if elapsed < self.min_request_interval_seconds:
                time.sleep(self.min_request_interval_seconds - elapsed)
            self._last_request_at = time.monotonic()

    def _existing_periods(self, mode: str, path: Path) -> set[str]:
        if not path.exists():
            return set()
        ds = xr.open_zarr(path)
        try:
            values = pd.to_datetime(ds["time"].values, utc=True)
            if mode == "analysis":
                return {str(int(ts.year)) for ts in values if int(ts.month) == 8}
            return {ts.date().isoformat() for ts in values}
        finally:
            ds.close()

    def _append_period(self, path: Path, period_ds: xr.Dataset) -> None:
        time_chunk = min(720, period_ds.sizes["time"])
        lat_chunk = min(32, period_ds.sizes["lat"])
        lon_chunk = min(40, period_ds.sizes["lon"])
        encoding = {
            "u10": {"chunks": (time_chunk, lat_chunk, lon_chunk)},
            "v10": {"chunks": (time_chunk, lat_chunk, lon_chunk)},
        }
        if not path.exists():
            period_ds.to_zarr(path, mode="w", encoding=encoding, consolidated=True, zarr_format=2)
            return
        period_ds.to_zarr(
            path,
            mode="a",
            append_dim="time",
            consolidated=True,
            zarr_format=2,
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

    def _endpoint_config_for_mode(self, mode: str) -> EndpointConfig:
        if not DEFAULT_ENDPOINTS_PATH.exists():
            raise RuntimeError(
                f"Endpoint config is required at {DEFAULT_ENDPOINTS_PATH}. "
                "Refusing implicit endpoint fallback."
            )
        parsed = yaml.safe_load(DEFAULT_ENDPOINTS_PATH.read_text(encoding="utf-8")) or {}
        modes_cfg = parsed.get("modes") or {}
        mode_cfg = modes_cfg.get(mode) or {}
        endpoint = str(mode_cfg.get("endpoint", "")).strip()
        auth_mode = str(mode_cfg.get("auth_mode", "")).strip()

        if not endpoint or not auth_mode:
            raise RuntimeError(
                f"Mode '{mode}' must define both endpoint and auth_mode in "
                f"{DEFAULT_ENDPOINTS_PATH}."
            )
        if auth_mode not in {"none", "require_key", "use_key_if_present"}:
            raise RuntimeError(
                f"Invalid auth_mode '{auth_mode}' for mode '{mode}' in "
                f"{DEFAULT_ENDPOINTS_PATH}"
            )
        return EndpointConfig(endpoint=endpoint, auth_mode=auth_mode)

    def _resolve_auth_mode(self, auth_mode: str) -> bool:
        if auth_mode == "none":
            return False
        if auth_mode == "use_key_if_present":
            return bool(self.api_key and self.api_key.strip())
        if auth_mode == "require_key":
            if not self.api_key or not self.api_key.strip():
                raise RuntimeError(
                    "OPENMETEO_API_KEY is required by auth_mode=require_key "
                    "for the selected endpoint."
                )
            return True
        raise RuntimeError(f"Unsupported auth_mode '{auth_mode}'")

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

    def _checkpoint_path(
        self,
        mode: str,
        model: str,
        start: dt.date,
        end: dt.date,
        month_filter: set[int] | None,
    ) -> Path:
        DEFAULT_CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
        mf = "all" if month_filter is None else "-".join(str(m) for m in sorted(month_filter))
        filename = f"{mode}_{model}_{start.isoformat()}_{end.isoformat()}_{mf}.json"
        return DEFAULT_CHECKPOINT_ROOT / filename

    def _load_checkpoint(self, path: Path) -> dict[str, set[str]]:
        if not path.exists():
            return {"completed_tasks": set(), "periods_written": set()}
        parsed = json.loads(path.read_text(encoding="utf-8"))
        return {
            "completed_tasks": set(str(v) for v in parsed.get("completed_tasks", [])),
            "periods_written": set(str(v) for v in parsed.get("periods_written", [])),
        }

    def _save_checkpoint(self, path: Path, checkpoint: dict[str, set[str]]) -> None:
        serialized = {
            "completed_tasks": sorted(checkpoint["completed_tasks"]),
            "periods_written": sorted(checkpoint["periods_written"]),
            "updated_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        }
        path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    def _build_periods(
        self,
        mode: str,
        start: dt.date,
        end: dt.date,
        month_filter: set[int] | None,
    ) -> list[tuple[str, dt.date, dt.date]]:
        if mode == "analysis":
            year_periods = _build_year_periods(start=start, end=end, month_filter=month_filter)
            return [(str(year), period_start, period_end) for year, period_start, period_end in year_periods]
        return [(day.isoformat(), day, day) for day in _iter_days(start, end, month_filter)]


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


def _build_year_periods(
    start: dt.date,
    end: dt.date,
    month_filter: set[int] | None,
) -> list[tuple[int, dt.date, dt.date]]:
    periods: list[tuple[int, dt.date, dt.date]] = []
    years = range(start.year, end.year + 1)
    for year in years:
        august_start = dt.date(year, 8, 1)
        august_end = dt.date(year, 8, 31)
        if month_filter is not None and 8 not in month_filter:
            continue
        if august_end < start or august_start > end:
            continue
        period_start = max(start, august_start)
        period_end = min(end, august_end)
        periods.append((year, period_start, period_end))
    return periods


def _task_id(period_id: str, batch_index: int) -> str:
    return f"{period_id}:{batch_index}"


def _response_reason(response: requests.Response | None) -> str:
    if response is None:
        return "no response body"
    try:
        payload = response.json()
    except ValueError:
        return "unparseable response body"
    reason = payload.get("reason")
    if reason is None:
        return "no reason provided"
    return str(reason)


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

