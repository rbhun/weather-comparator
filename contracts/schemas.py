"""Frozen schema contracts and shared math helpers for PMC-2026."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

MS_TO_KT = 1.9438445
KT_TO_MS = 1.0 / MS_TO_KT
EARTH_RADIUS_M = 6_371_000.0
NM_PER_M = 0.000539956803
EARTH_RADIUS_NM = EARTH_RADIUS_M * NM_PER_M

C4_COLUMNS: dict[str, str] = {
    "start_time": "datetime64[ns]",
    "route_id": "str",
    "elapsed_hours": "float32",
    "distance_nm": "float32",
    "mean_tws_kt": "float32",
    "hours_below_5kt": "float32",
    "hours_upwind": "float32",
    "land_fill_fraction": "float32",
    "stalled": "bool",
    "max_stall_hours": "float32",
}

C5_COLUMNS: dict[str, str] = {
    **C4_COLUMNS,
    "track": "object",
    "n_manoeuvres": "int32",
    "gate_time": "datetime64[ns]",
    "corridor": "str",
}


def uv_to_tws_twd(u10_ms: Any, v10_ms: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert eastward/northward wind components to speed and direction-from."""
    u_arr = np.asarray(u10_ms, dtype=float)
    v_arr = np.asarray(v10_ms, dtype=float)
    tws_kt = np.hypot(u_arr, v_arr) * MS_TO_KT
    twd_deg = (np.degrees(np.arctan2(-u_arr, -v_arr)) + 360.0) % 360.0
    return tws_kt, twd_deg


def tws_twd_to_uv(tws_kt: Any, twd_deg: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert speed and meteorological direction-from to u/v components."""
    tws_ms = np.asarray(tws_kt, dtype=float) * KT_TO_MS
    twd_rad = np.radians(np.asarray(twd_deg, dtype=float))
    u10_ms = -tws_ms * np.sin(twd_rad)
    v10_ms = -tws_ms * np.cos(twd_rad)
    return u10_ms, v10_ms


def angular_difference(a_deg: Any, b_deg: Any) -> np.ndarray:
    """Return signed smallest angle a-b in [-180, 180)."""
    a_arr = np.asarray(a_deg, dtype=float)
    b_arr = np.asarray(b_deg, dtype=float)
    return ((a_arr - b_arr + 180.0) % 360.0) - 180.0


def haversine_nm(lat0: Any, lon0: Any, lat1: Any, lon1: Any) -> np.ndarray:
    """Great-circle distance in nautical miles."""
    lat0_rad = np.radians(np.asarray(lat0, dtype=float))
    lon0_rad = np.radians(np.asarray(lon0, dtype=float))
    lat1_rad = np.radians(np.asarray(lat1, dtype=float))
    lon1_rad = np.radians(np.asarray(lon1, dtype=float))
    dlat = lat1_rad - lat0_rad
    dlon = lon1_rad - lon0_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0_rad) * np.cos(lat1_rad) * np.sin(
        dlon / 2.0
    ) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return c * EARTH_RADIUS_NM


def initial_bearing_deg(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    """Initial great-circle bearing in degrees true."""
    lat0_rad, lat1_rad = np.radians([lat0, lat1])
    lon0_rad, lon1_rad = np.radians([lon0, lon1])
    dlon = lon1_rad - lon0_rad
    y = np.sin(dlon) * np.cos(lat1_rad)
    x = np.cos(lat0_rad) * np.sin(lat1_rad) - np.sin(lat0_rad) * np.cos(
        lat1_rad
    ) * np.cos(dlon)
    return float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)


def advance_position(
    lat: float, lon: float, bearing_deg: float, distance_nm: float
) -> tuple[float, float]:
    """Advance a point along a great-circle arc by distance_nm."""
    if distance_nm == 0:
        return lat, lon
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    brg_rad = np.radians(bearing_deg)
    d_rad = distance_nm / EARTH_RADIUS_NM
    lat2 = np.arcsin(
        np.sin(lat_rad) * np.cos(d_rad)
        + np.cos(lat_rad) * np.sin(d_rad) * np.cos(brg_rad)
    )
    lon2 = lon_rad + np.arctan2(
        np.sin(brg_rad) * np.sin(d_rad) * np.cos(lat_rad),
        np.cos(d_rad) - np.sin(lat_rad) * np.sin(lat2),
    )
    lon2 = (lon2 + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.degrees(lat2)), float(np.degrees(lon2))


@dataclass(frozen=True)
class Domain:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    resolution: float
    fixture_resolution: float = 0.25
    fixture_days: int = 30


@dataclass(frozen=True)
class Course:
    start: tuple[float, float]
    gate: tuple[float, float]
    gate_tolerance_nm: float
    finish: tuple[float, float]
    start_time_utc: datetime
    climatology_months: tuple[int, ...]
    climatology_years: tuple[int, ...]
    display_timezone: str


@dataclass(frozen=True)
class Route:
    id: str
    label: str
    description: str
    legs: tuple[tuple[float, float], ...]
    tags: tuple[str, ...] = ()

    def assert_passes_gate(
        self, gate_lat: float, gate_lon: float, tolerance_nm: float = 2.0
    ) -> None:
        """Assert route passes within tolerance of the mandatory gate."""
        for lat, lon in self.legs:
            if haversine_nm(lat, lon, gate_lat, gate_lon) <= tolerance_nm:
                return

        min_dist = float("inf")
        for (lat0, lon0), (lat1, lon1) in zip(self.legs[:-1], self.legs[1:]):
            frac = np.linspace(0.0, 1.0, 200)
            seg_lat = lat0 + (lat1 - lat0) * frac
            seg_lon = lon0 + (lon1 - lon0) * frac
            d = haversine_nm(seg_lat, seg_lon, gate_lat, gate_lon)
            min_dist = min(min_dist, float(np.nanmin(d)))
        if min_dist > tolerance_nm:
            raise ValueError(
                f"Route '{self.id}' misses gate by {min_dist:.2f} nm "
                f"(tolerance {tolerance_nm:.2f} nm)."
            )


@dataclass(frozen=True)
class FollowResult:
    start_time: datetime
    route_id: str
    elapsed_hours: float
    distance_nm: float
    mean_tws_kt: float
    hours_below_5kt: float
    hours_upwind: float
    stalled: bool
    max_stall_hours: float
    land_fill_fraction: float

    def as_row(self) -> dict[str, Any]:
        ts = pd.Timestamp(self.start_time)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return {
            "start_time": np.datetime64(ts.to_datetime64()),
            "route_id": self.route_id,
            "elapsed_hours": np.float32(self.elapsed_hours),
            "distance_nm": np.float32(self.distance_nm),
            "mean_tws_kt": np.float32(self.mean_tws_kt),
            "hours_below_5kt": np.float32(self.hours_below_5kt),
            "hours_upwind": np.float32(self.hours_upwind),
            "land_fill_fraction": np.float32(self.land_fill_fraction),
            "stalled": bool(self.stalled),
            "max_stall_hours": np.float32(self.max_stall_hours),
        }


@dataclass(frozen=True)
class RouteResult(FollowResult):
    track: tuple[tuple[float, float, str], ...]
    n_manoeuvres: int
    gate_time: datetime
    corridor: str


@dataclass(frozen=True)
class Polar:
    tws_kt: np.ndarray
    twa_deg: np.ndarray
    bsp_kt: np.ndarray
    name: str
    source_file: str
    _vmg_cache: dict[tuple[float, bool], tuple[float, float]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.tws_kt.ndim != 1 or self.twa_deg.ndim != 1:
            raise ValueError("tws_kt and twa_deg must be 1-D arrays.")
        if self.bsp_kt.shape != (self.twa_deg.size, self.tws_kt.size):
            raise ValueError("bsp_kt shape must be (n_twa, n_tws).")
        if not np.all(np.diff(self.tws_kt) > 0):
            raise ValueError("tws_kt must be strictly ascending.")
        if not np.all(np.diff(self.twa_deg) > 0):
            raise ValueError("twa_deg must be strictly ascending.")

    def speed(self, twa_deg: Any, tws_kt: Any) -> np.ndarray:
        """Vectorised bilinear interpolation with TWS clamping."""
        twa = np.abs(np.asarray(twa_deg, dtype=float))
        tws = np.asarray(tws_kt, dtype=float)
        twa = np.clip(twa, self.twa_deg[0], self.twa_deg[-1])
        tws = np.clip(tws, self.tws_kt[0], self.tws_kt[-1])

        i = np.searchsorted(self.twa_deg, twa, side="right") - 1
        j = np.searchsorted(self.tws_kt, tws, side="right") - 1
        i = np.clip(i, 0, self.twa_deg.size - 2)
        j = np.clip(j, 0, self.tws_kt.size - 2)

        twa0 = self.twa_deg[i]
        twa1 = self.twa_deg[i + 1]
        tws0 = self.tws_kt[j]
        tws1 = self.tws_kt[j + 1]

        dtwa = np.where(twa1 == twa0, 1.0, twa1 - twa0)
        dtws = np.where(tws1 == tws0, 1.0, tws1 - tws0)
        wa = (twa - twa0) / dtwa
        wb = (tws - tws0) / dtws

        b00 = self.bsp_kt[i, j]
        b10 = self.bsp_kt[i + 1, j]
        b01 = self.bsp_kt[i, j + 1]
        b11 = self.bsp_kt[i + 1, j + 1]

        bsp = (
            (1.0 - wa) * (1.0 - wb) * b00
            + wa * (1.0 - wb) * b10
            + (1.0 - wa) * wb * b01
            + wa * wb * b11
        )
        return np.where(bsp <= 0.0, 0.0, bsp)

    def vmg_optimum(self, tws_kt: float, upwind: bool) -> tuple[float, float]:
        """Return (best_twa_deg, bsp_kt) that maximises VMG."""
        tws = float(np.clip(tws_kt, self.tws_kt[0], self.tws_kt[-1]))
        key = (round(tws, 4), upwind)
        cached = self._vmg_cache.get(key)
        if cached is not None:
            return cached

        bsp = self.speed(self.twa_deg, tws)
        rad = np.radians(self.twa_deg)
        if upwind:
            mask = self.twa_deg <= 90.0
            score = np.where(mask, bsp * np.cos(rad), -np.inf)
        else:
            mask = self.twa_deg >= 90.0
            score = np.where(mask, -bsp * np.cos(rad), -np.inf)

        idx = int(np.nanargmax(score))
        result = (float(self.twa_deg[idx]), float(bsp[idx]))
        self._vmg_cache[key] = result
        return result

    def validate(self) -> dict[str, Any]:
        """Validate basic polar quality checks and report diagnostics."""
        report: dict[str, Any] = {
            "name": self.name,
            "monotonic_tws": bool(np.all(np.diff(self.tws_kt) > 0)),
            "monotonic_twa": bool(np.all(np.diff(self.twa_deg) > 0)),
            "interpolated_tws_range": (
                float(self.tws_kt[0]),
                float(self.tws_kt[-1]),
            ),
            "implausible_cells": [],
        }
        for col_idx, tws in enumerate(self.tws_kt):
            if tws > 10.0:
                bsp_col = self.bsp_kt[:, col_idx]
                bad = np.where(bsp_col > 1.5 * tws)[0]
                for row_idx in bad:
                    report["implausible_cells"].append(
                        {
                            "twa_deg": float(self.twa_deg[row_idx]),
                            "tws_kt": float(tws),
                            "bsp_kt": float(bsp_col[row_idx]),
                        }
                    )
        if float(self.tws_kt.min()) < 8.0:
            print("WARNING: POLAR TWS < 8 kt IS NOT VALIDATED; TREAT OUTPUT AS RELATIVE.")
        return report


def directional_constancy(u10_ms: Any, v10_ms: Any) -> np.ndarray:
    """Return directional constancy = |vector-mean| / mean speed."""
    u_arr = np.asarray(u10_ms, dtype=float)
    v_arr = np.asarray(v10_ms, dtype=float)
    valid = np.isfinite(u_arr) & np.isfinite(v_arr)
    count = valid.sum(axis=0)
    safe_count = np.where(count == 0, 1, count)
    mean_u = np.sum(np.where(valid, u_arr, 0.0), axis=0) / safe_count
    mean_v = np.sum(np.where(valid, v_arr, 0.0), axis=0) / safe_count
    mean_speed = np.sum(np.where(valid, np.hypot(u_arr, v_arr), 0.0), axis=0) / safe_count
    mean_u = np.where(count == 0, np.nan, mean_u)
    mean_v = np.where(count == 0, np.nan, mean_v)
    mean_speed = np.where(count == 0, np.nan, mean_speed)
    with np.errstate(invalid="ignore", divide="ignore"):
        constancy = np.hypot(mean_u, mean_v) / mean_speed
    return np.where(np.isfinite(constancy), np.clip(constancy, 0.0, 1.0), np.nan)


def validate_wind_store(wind: xr.Dataset) -> None:
    """Validate C1 wind-store shape and conventions."""
    for dim in ("time", "lat", "lon"):
        if dim not in wind.coords:
            raise ValueError(f"Missing coordinate: {dim}")
    for var in ("u10", "v10"):
        if var not in wind.data_vars:
            raise ValueError(f"Missing variable: {var}")
        if wind[var].dims != ("time", "lat", "lon"):
            raise ValueError(f"{var} dims must be (time, lat, lon)")
    time_vals = wind["time"].values.astype("datetime64[ns]")
    if time_vals.size < 2:
        raise ValueError("time axis must contain at least two entries.")
    if np.any(np.diff(time_vals) <= np.timedelta64(0, "ns")):
        raise ValueError("time axis must be strictly ascending with no duplicates.")
    lat_vals = wind["lat"].values.astype(float)
    lon_vals = wind["lon"].values.astype(float)
    if np.any(np.diff(lat_vals) <= 0.0):
        raise ValueError("lat axis must be strictly ascending.")
    if np.any(np.diff(lon_vals) <= 0.0):
        raise ValueError("lon axis must be strictly ascending.")

    u_nan = np.isnan(wind["u10"].values)
    v_nan = np.isnan(wind["v10"].values)
    if not np.array_equal(u_nan, v_nan):
        raise ValueError("u10/v10 NaN masks must be identical.")
    if not np.any(u_nan):
        raise ValueError("Expected some NaN cells representing land.")

    for attr in ("source", "fetched_utc", "api_version", "omissions"):
        if attr not in wind.attrs:
            raise ValueError(f"Missing dataset attr: {attr}")


def validate_climatology(climo: xr.Dataset) -> None:
    """Validate C6 climatology structure and low-sample masking rule."""
    required_dims = {"hour", "lat", "lon"}
    if not required_dims.issubset(set(climo.dims)):
        raise ValueError(f"Climatology dims must include {required_dims}")
    required_vars = {
        "mean_tws_kt",
        "vector_mean_u",
        "vector_mean_v",
        "p_below_5kt",
        "p_below_8kt",
        "p_above_20kt",
        "directional_const",
        "n_samples",
    }
    if not required_vars.issubset(set(climo.data_vars)):
        raise ValueError("Climatology dataset is missing required variables.")
    low_samples = climo["n_samples"] < 200
    for var in required_vars - {"n_samples"}:
        invalid = np.logical_and(low_samples.values, ~np.isnan(climo[var].values))
        if np.any(invalid):
            raise ValueError(
                f"Variable '{var}' has unmasked values where n_samples < 200."
            )


def validate_dashboard_payload(payload: dict[str, Any]) -> None:
    """Validate C7 dashboard payload shape."""
    top = {"meta", "climatology", "routes", "head_to_head", "skill"}
    if not top.issubset(payload.keys()):
        raise ValueError("Dashboard payload missing top-level keys.")

    meta = payload["meta"]
    for key in ("generated_utc", "display_timezone", "course", "warnings"):
        if key not in meta:
            raise ValueError(f"Dashboard meta missing '{key}'.")
    if not isinstance(meta["warnings"], list):
        raise ValueError("meta.warnings must be a list.")

    climo = payload["climatology"]
    if "grid" not in climo or "by_hour" not in climo:
        raise ValueError("climatology must include grid and by_hour.")
    if not isinstance(climo["by_hour"], list) or len(climo["by_hour"]) == 0:
        raise ValueError("climatology.by_hour must be a non-empty list.")

    if not isinstance(payload["routes"], list):
        raise ValueError("routes must be a list.")
    if not isinstance(payload["head_to_head"], list):
        raise ValueError("head_to_head must be a list.")
    if not isinstance(payload["skill"], list):
        raise ValueError("skill must be a list.")


def load_domain(path: Path) -> Domain:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Domain(
        lat_min=float(raw["lat"]["min"]),
        lat_max=float(raw["lat"]["max"]),
        lon_min=float(raw["lon"]["min"]),
        lon_max=float(raw["lon"]["max"]),
        resolution=float(raw["resolution"]),
        fixture_resolution=float(raw.get("fixture_resolution", 0.25)),
        fixture_days=int(raw.get("fixture_days", 30)),
    )


def load_course(path: Path) -> Course:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    start_time = pd.Timestamp(raw["start_time_utc"], tz="UTC").to_pydatetime()
    years = tuple(int(y) for y in raw["climatology"]["years"])
    months = tuple(int(m) for m in raw["climatology"]["months"])
    return Course(
        start=(float(raw["start"]["lat"]), float(raw["start"]["lon"])),
        gate=(float(raw["gate"]["lat"]), float(raw["gate"]["lon"])),
        gate_tolerance_nm=float(raw["gate"]["tolerance_nm"]),
        finish=(float(raw["finish"]["lat"]), float(raw["finish"]["lon"])),
        start_time_utc=start_time,
        climatology_months=months,
        climatology_years=years,
        display_timezone=str(raw["display_timezone"]),
    )


def load_routes(path: Path) -> list[Route]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    routes: list[Route] = []
    # Import lazily to avoid import-time cycles between contracts and pmc modules.
    from pmc.geo import crosses_land

    for item in raw:
        legs = tuple((float(lat), float(lon)) for lat, lon in item["legs"])
        for (lat0, lon0), (lat1, lon1) in zip(legs[:-1], legs[1:]):
            if crosses_land(lat0, lon0, lat1, lon1, safety_buffer_nm=0.5):
                raise ValueError(
                    f"Route '{item['id']}' has land crossing segment "
                    f"({lat0:.4f},{lon0:.4f}) -> ({lat1:.4f},{lon1:.4f})"
                )
        routes.append(
            Route(
                id=str(item["id"]),
                label=str(item["label"]),
                description=str(item["description"]),
                legs=legs,
                tags=tuple(str(x) for x in item.get("tags", [])),
            )
        )
    return routes

