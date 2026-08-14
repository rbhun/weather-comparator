"""A2 — Geometry module for land masking and route intersection checks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.prepared import prep
from shapely.strtree import STRtree

from contracts.schemas import EARTH_RADIUS_NM, haversine_nm, initial_bearing_deg

DEFAULT_SAFETY_BUFFER_NM = 0.5
_LAND_POLYGONS: list[Polygon] | None = None
_LAND_PREPARED = None
_LAND_INDEX: STRtree | None = None
_LAND_SOURCE: str | None = None


def _fallback_polygons() -> list[Polygon]:
    """
    Simplified race-area coastline polygons used when external datasets are absent.

    These are hand-curated extracts for Sicily, Sardinia/Corsica, and nearby
    islands sufficient for route-crossing checks in the race corridor.
    """
    sardinia = Polygon(
        [
            (8.16, 38.86),
            (8.30, 39.18),
            (8.54, 39.62),
            (8.75, 40.03),
            (8.86, 40.43),
            (9.01, 40.79),
            (9.26, 41.02),
            (9.58, 41.09),
            (9.84, 41.01),
            (9.83, 40.84),
            (9.66, 40.66),
            (9.51, 40.45),
            (9.36, 40.19),
            (9.43, 39.88),
            (9.62, 39.50),
            (9.55, 39.13),
            (9.19, 38.93),
            (8.75, 38.83),
            (8.36, 38.79),
        ]
    )
    corsica = Polygon(
        [
            (8.54, 41.35),
            (8.76, 41.53),
            (8.97, 41.78),
            (9.20, 42.05),
            (9.45, 42.34),
            (9.54, 42.57),
            (9.45, 42.84),
            (9.23, 43.06),
            (9.04, 43.22),
            (8.75, 43.28),
            (8.46, 43.23),
            (8.22, 43.11),
            (8.03, 42.89),
            (8.08, 42.61),
            (8.20, 42.35),
            (8.33, 42.06),
            (8.39, 41.80),
            (8.42, 41.56),
        ]
    )
    maddalena = Polygon(
        [
            (9.18, 41.13),
            (9.31, 41.17),
            (9.43, 41.23),
            (9.44, 41.30),
            (9.31, 41.31),
            (9.20, 41.24),
        ]
    )
    sicily_northwest = Polygon(
        [
            (12.32, 37.95),
            (12.70, 37.90),
            (13.10, 38.00),
            (13.37, 38.16),
            (13.95, 38.45),
            (14.50, 38.40),
            (14.50, 37.55),
            (13.60, 37.50),
            (12.55, 37.50),
        ]
    )
    return [sardinia, corsica, maddalena, sicily_northwest]


def _candidate_geojson_paths() -> list[Path]:
    return [
        Path("/usr/share/gshhg/gshhs_f.geojson"),
        Path("/usr/share/gshhg/gshhs_h.geojson"),
        Path("/usr/share/gshhg/gshhs_l.geojson"),
        Path.home() / ".cache" / "pmc" / "coastline_10m.geojson",
    ]


def _ensure_cached_natural_earth() -> None:
    cache_path = Path.home() / ".cache" / "pmc" / "coastline_10m.geojson"
    if cache_path.exists():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
        "master/geojson/ne_10m_land.geojson"
    )
    try:
        with urlopen(url, timeout=20) as response:
            payload = response.read()
        cache_path.write_bytes(payload)
    except (URLError, TimeoutError, OSError):
        # Offline mode or download failure: fallback polygons are used.
        return


def _load_external_polygons() -> tuple[list[Polygon], str | None]:
    domain_bbox = Polygon([(6.0, 37.0), (15.0, 37.0), (15.0, 44.5), (6.0, 44.5)])
    _ensure_cached_natural_earth()
    polygons: list[Polygon] = []
    for candidate in _candidate_geojson_paths():
        if not candidate.exists():
            continue
        raw = json.loads(candidate.read_text(encoding="utf-8"))
        features = raw.get("features", [])
        for feat in features:
            geom = shape(feat.get("geometry"))
            if geom.geom_type == "Polygon":
                if geom.intersects(domain_bbox):
                    polygons.append(geom)
            elif geom.geom_type == "MultiPolygon":
                polygons.extend(
                    [poly for poly in geom.geoms if isinstance(poly, Polygon) and poly.intersects(domain_bbox)]
                )
        if polygons:
            return polygons, str(candidate)
    return [], None


def _ensure_land_index() -> tuple[list[Polygon], list, STRtree]:
    global _LAND_POLYGONS, _LAND_PREPARED, _LAND_INDEX, _LAND_SOURCE
    if _LAND_POLYGONS is None or _LAND_PREPARED is None or _LAND_INDEX is None:
        polygons, source = _load_external_polygons()
        if not polygons:
            polygons = _fallback_polygons()
            source = "fallback-polygons"
        _LAND_POLYGONS = polygons
        _LAND_SOURCE = source
        _LAND_PREPARED = [prep(poly) for poly in polygons]
        _LAND_INDEX = STRtree(polygons)
    return _LAND_POLYGONS, _LAND_PREPARED, _LAND_INDEX


def coastline_info() -> dict[str, object]:
    """Return runtime coastline source metadata."""
    polygons, _, _ = _ensure_land_index()
    return {"source": _LAND_SOURCE, "polygon_count": len(polygons)}


def _buffer_nm_to_degrees(safety_buffer_nm: float, ref_lat_deg: float) -> float:
    lat_deg = safety_buffer_nm / 60.0
    cos_lat = max(0.1, abs(math.cos(math.radians(ref_lat_deg))))
    lon_deg = safety_buffer_nm / (60.0 * cos_lat)
    return max(lat_deg, lon_deg)


def _point_on_land(lat: float, lon: float) -> bool:
    polygons, prepared, index = _ensure_land_index()
    point = Point(float(lon), float(lat))
    candidate_idx = index.query(point, predicate="intersects")
    for idx in candidate_idx:
        if prepared[int(idx)].intersects(point):
            return True
    return False


def is_sea(lat, lon) -> np.ndarray:
    """Return True where point(s) are sea using polygon coastline checks."""
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    lat_b, lon_b = np.broadcast_arrays(lat_arr, lon_arr)
    sea = np.ones(lat_b.shape, dtype=bool)
    it = np.nditer(
        [lat_b, lon_b, sea],
        flags=["multi_index", "refs_ok"],
        op_flags=[["readonly"], ["readonly"], ["readwrite"]],
    )
    for lat_v, lon_v, out in it:
        out[...] = not _point_on_land(float(lat_v), float(lon_v))
    return sea


def crosses_land(
    lat0, lon0, lat1, lon1, *, safety_buffer_nm: float = DEFAULT_SAFETY_BUFFER_NM
) -> bool:
    """Return whether segment (lat0,lon0)->(lat1,lon1) intersects buffered land."""
    _, prepared, index = _ensure_land_index()
    line = LineString([(float(lon0), float(lat0)), (float(lon1), float(lat1))])
    if safety_buffer_nm > 0:
        ref_lat = 0.5 * (float(lat0) + float(lat1))
        line = line.buffer(_buffer_nm_to_degrees(safety_buffer_nm, ref_lat), cap_style=2)
    candidate_idx = index.query(line, predicate="intersects")
    for idx in candidate_idx:
        if prepared[int(idx)].intersects(line):
            return True
    return False


def rhumb_distance_nm(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    """Rhumb-line distance in nautical miles."""
    lat0_r = math.radians(lat0)
    lat1_r = math.radians(lat1)
    dlat = lat1_r - lat0_r
    dlon = math.radians(lon1 - lon0)
    if dlon > math.pi:
        dlon -= 2 * math.pi
    elif dlon < -math.pi:
        dlon += 2 * math.pi
    dpsi = math.log(
        math.tan(math.pi / 4 + lat1_r / 2) / math.tan(math.pi / 4 + lat0_r / 2)
    )
    q = dlat / dpsi if abs(dpsi) > 1e-12 else math.cos(lat0_r)
    dist_rad = math.hypot(dlat, q * dlon)
    return dist_rad * EARTH_RADIUS_NM


def rhumb_bearing_deg(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    """Rhumb-line bearing in degrees true."""
    lat0_r = math.radians(lat0)
    lat1_r = math.radians(lat1)
    dlon = math.radians(lon1 - lon0)
    if dlon > math.pi:
        dlon -= 2 * math.pi
    elif dlon < -math.pi:
        dlon += 2 * math.pi
    dpsi = math.log(
        math.tan(math.pi / 4 + lat1_r / 2) / math.tan(math.pi / 4 + lat0_r / 2)
    )
    bearing = math.degrees(math.atan2(dlon, dpsi))
    return (bearing + 360.0) % 360.0


def cross_track_distance_nm(
    lat: float,
    lon: float,
    leg_start: tuple[float, float],
    leg_end: tuple[float, float],
) -> float:
    """Cross-track distance from point to great-circle leg in nautical miles."""
    d13 = float(haversine_nm(leg_start[0], leg_start[1], lat, lon)) / EARTH_RADIUS_NM
    theta13 = math.radians(initial_bearing_deg(leg_start[0], leg_start[1], lat, lon))
    theta12 = math.radians(
        initial_bearing_deg(leg_start[0], leg_start[1], leg_end[0], leg_end[1])
    )
    dxt = math.asin(math.sin(d13) * math.sin(theta13 - theta12))
    return dxt * EARTH_RADIUS_NM
