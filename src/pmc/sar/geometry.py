"""Coast-distance banding for the SAR lee-shadow study."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

from contracts.schemas import EARTH_RADIUS_NM, MS_TO_KT

# Re-use the race coastline; imported lazily to keep unit tests light when mocked.
NM_PER_KM = 1.0 / 1.852


def km_to_nm(km: float) -> float:
    return float(km) * NM_PER_KM


def ms_to_kt(speed_ms: Any) -> np.ndarray:
    return np.asarray(speed_ms, dtype=float) * MS_TO_KT


def _land_polygons():
    from pmc.geo import _ensure_land_index

    polygons, _, index = _ensure_land_index()
    return polygons, index


def distance_to_coast_nm(lat: Any, lon: Any) -> np.ndarray:
    """Great-circle-ish distance to nearest land polygon edge, in nautical miles.

    Uses an equirectangular approximation local to each point (adequate at ~1 km
    SAR resolution over a few tens of nm). Land points return 0.
    """
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    lat_b, lon_b = np.broadcast_arrays(lat_arr, lon_arr)
    out = np.full(lat_b.shape, np.nan, dtype=float)
    polygons, index = _land_polygons()
    flat_lat = lat_b.ravel()
    flat_lon = lon_b.ravel()
    flat_out = out.ravel()
    for i, (la, lo) in enumerate(zip(flat_lat, flat_lon)):
        if not np.isfinite(la) or not np.isfinite(lo):
            continue
        pt = Point(float(lo), float(la))
        nearest_idx = index.nearest(pt)
        if isinstance(nearest_idx, (list, tuple, np.ndarray)):
            nearest_idx = int(np.asarray(nearest_idx).ravel()[0])
        else:
            nearest_idx = int(nearest_idx)
        nearest_geom = polygons[nearest_idx]
        if nearest_geom.contains(pt) or nearest_geom.covers(pt):
            flat_out[i] = 0.0
            continue
        _np1, np2 = nearest_points(pt, nearest_geom)
        dlat = (np2.y - pt.y) * (math.pi / 180.0) * EARTH_RADIUS_NM
        mean_lat = 0.5 * (pt.y + np2.y)
        dlon = (
            (np2.x - pt.x)
            * (math.pi / 180.0)
            * math.cos(math.radians(mean_lat))
            * EARTH_RADIUS_NM
        )
        flat_out[i] = float(math.hypot(dlat, dlon))
    return out


def distance_to_virtual_meridian_nm(lat: Any, lon: Any, coast_lon: float) -> np.ndarray:
    """Distance in nm from points to a N–S virtual coast at ``coast_lon``."""
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    lat_b, lon_b = np.broadcast_arrays(lat_arr, lon_arr)
    # East of the virtual coast is "offshore" (positive distance).
    dlon_deg = lon_b - float(coast_lon)
    dist = np.abs(dlon_deg) * 60.0 * np.cos(np.radians(lat_b))
    # Signed: points west of the line get negative (treated as landward / invalid).
    return np.where(dlon_deg >= 0.0, dist, -dist)


def in_bbox(
    lat: Any,
    lon: Any,
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> np.ndarray:
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    return (
        (lat_arr >= lat_min)
        & (lat_arr <= lat_max)
        & (lon_arr >= lon_min)
        & (lon_arr <= lon_max)
    )


def assign_band(
    distance_nm: Any,
    bands_nm: list[tuple[float, float]],
) -> np.ndarray:
    """Return band index 0..n-1, or -1 if outside all bands / invalid."""
    dist = np.asarray(distance_nm, dtype=float)
    out = np.full(dist.shape, -1, dtype=np.int16)
    for idx, (lo, hi) in enumerate(bands_nm):
        mask = np.isfinite(dist) & (dist >= lo) & (dist < hi)
        out = np.where(mask, idx, out)
    return out


def band_label(lo: float, hi: float) -> str:
    return f"{lo:g}–{hi:g} nm"
