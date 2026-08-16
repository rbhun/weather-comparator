"""Fetch / open C9 Sentinel-1 L3 scalar wind stores."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml

from contracts.schemas import validate_sar_store

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURE = REPO_ROOT / "contracts" / "fixtures" / "sar_scenes_small.zarr"


def load_sar_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (REPO_ROOT / "config" / "sar.yaml")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid SAR config at {cfg_path}")
    return raw


def open_sar_store(path: Path) -> xr.Dataset:
    """Open a C10 SAR store from zarr or netCDF and validate."""
    path = Path(path)
    if path.suffix == ".nc":
        ds = xr.open_dataset(path)
    else:
        ds = xr.open_zarr(path, consolidated=True)
    validate_sar_store(ds)
    return ds


def _cache_key(params: dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def fetch_sar_scenes(
    *,
    start: date,
    end: date,
    cfg: dict[str, Any] | None = None,
    use_fixture: bool = False,
    output_dir: Path | None = None,
) -> Path:
    """Fetch (or resume) Sentinel-1 L3 daily wind into a C9 zarr store.

    Without Copernicus Marine credentials, or when ``use_fixture=True``, returns
    the committed synthetic fixture path. Real downloads require the optional
    ``copernicusmarine`` package and account env vars.
    """
    config = cfg or load_sar_config()
    if use_fixture:
        if not DEFAULT_FIXTURE.exists():
            raise FileNotFoundError(f"Missing SAR fixture: {DEFAULT_FIXTURE}")
        return DEFAULT_FIXTURE

    out_root = Path(output_dir or (REPO_ROOT / config.get("cache_dir", "data/sar")))
    out_root.mkdir(parents=True, exist_ok=True)
    params = {
        "product_id": config.get("product_id"),
        "start": str(start),
        "end": str(end),
        "years": config.get("years"),
        "months": config.get("months"),
    }
    key = _cache_key(params)
    target = out_root / f"s1_l3_{key}.zarr"
    meta_path = out_root / f"s1_l3_{key}.meta.json"

    if target.exists():
        LOGGER.info("SAR cache hit: %s", target)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            LOGGER.info(
                "SAR cache meta: scenes=%s cache_hits=%s",
                meta.get("n_scenes"),
                meta.get("cache_hits", 1),
            )
        return target

    try:
        import copernicusmarine  # type: ignore
    except ImportError:
        LOGGER.warning(
            "copernicusmarine not installed; falling back to fixture SAR store."
        )
        return DEFAULT_FIXTURE

    # Real fetch path — subset to the union of study + control corridors.
    sard = config["sardinia_corridor"]
    ctrl = config["control_corridor"]
    lat_min = min(float(sard["lat_min"]), float(ctrl["lat_min"])) - 0.2
    lat_max = max(float(sard["lat_max"]), float(ctrl["lat_max"])) + 0.2
    lon_min = min(float(sard["lon_min"]), float(ctrl["lon_min"])) - 0.2
    lon_max = max(float(sard["lon_max"]), float(ctrl["lon_max"])) + 0.2

    tmp_nc = out_root / f"s1_l3_{key}_raw.nc"
    try:
        copernicusmarine.subset(
            dataset_id=str(config["product_id"]),
            variables=["wind_speed", "wind_speed_flags", "incidence_angle"],
            minimum_longitude=lon_min,
            maximum_longitude=lon_max,
            minimum_latitude=lat_min,
            maximum_latitude=lat_max,
            start_datetime=f"{start.isoformat()}T00:00:00",
            end_datetime=f"{end.isoformat()}T23:59:59",
            output_filename=str(tmp_nc.name),
            output_directory=str(out_root),
            force_download=True,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Copernicus Marine subset failed (%s); using fixture.", exc)
        return DEFAULT_FIXTURE

    raw_path = out_root / tmp_nc.name
    if not raw_path.exists():
        # client may write without forcing our name
        candidates = sorted(out_root.glob("*.nc"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            LOGGER.error("No NetCDF written by copernicusmarine; using fixture.")
            return DEFAULT_FIXTURE
        raw_path = candidates[-1]

    ds = _normalise_cmems_to_c9(raw_path, product_id=str(config["product_id"]))
    validate_sar_store(ds)
    ds.to_zarr(target, mode="w", consolidated=True)
    meta_path.write_text(
        json.dumps(
            {
                "params": params,
                "n_scenes": int(ds.sizes["scene"]),
                "fetched_utc": ds.attrs.get("fetched_utc"),
                "cache_hits": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "SAR fetch complete: scenes=%d path=%s",
        int(ds.sizes["scene"]),
        target,
    )
    return target


def _normalise_cmems_to_c9(path: Path, *, product_id: str) -> xr.Dataset:
    """Map heterogeneous CMEMS variable names onto the C9 schema."""
    raw = xr.open_dataset(path)
    # Time axis → scene
    if "time" not in raw.coords and "time" not in raw.dims:
        raise ValueError("CMEMS SAR product missing time coordinate")
    speed_name = next(
        (n for n in ("wind_speed", "wind_speed_ms", "wspd", "speed") if n in raw),
        None,
    )
    if speed_name is None:
        raise ValueError(f"No wind speed variable in {path}; vars={list(raw.data_vars)}")
    speed = raw[speed_name]
    # Ensure (time, lat, lon)
    rename = {}
    for cand, canon in (("latitude", "lat"), ("longitude", "lon")):
        if cand in speed.dims or cand in speed.coords:
            rename[cand] = canon
    if rename:
        speed = speed.rename(rename)
        raw = raw.rename(rename)

    times = speed["time"].values
    n_scene = int(times.shape[0])
    lat = speed["lat"].values.astype("float32")
    lon = speed["lon"].values.astype("float32")
    wind = np.asarray(speed.values, dtype="float32")

    inc = np.full_like(wind, np.nan, dtype="float32")
    for cand in ("incidence_angle", "incidence_deg", "inc"):
        if cand in raw:
            inc = np.asarray(raw[cand].values, dtype="float32")
            break

    quality = np.zeros(wind.shape, dtype="int8")
    for cand in ("wind_speed_flags", "quality_flag", "flags"):
        if cand in raw:
            quality = np.asarray(raw[cand].values, dtype="int8")
            break

    # Quality: treat non-finite speed as bad
    quality = np.where(np.isfinite(wind), quality, np.int8(1))

    ds = xr.Dataset(
        data_vars={
            "wind_speed_ms": (("scene", "lat", "lon"), wind),
            "incidence_deg": (("scene", "lat", "lon"), inc),
            "quality_flag": (("scene", "lat", "lon"), quality),
        },
        coords={
            "scene": np.arange(n_scene, dtype="int32"),
            "time": ("scene", times.astype("datetime64[ns]")),
            "lat": lat.astype("float32"),
            "lon": lon.astype("float32"),
        },
        attrs={
            "source": "sentinel1_l3_cmems",
            "product_id": product_id,
            "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "direction_policy": "speed_only_no_direction",
            "units_note": "wind_speed_ms is SI; analysis layer converts to knots once",
        },
    )
    return ds
