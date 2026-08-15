"""A2 — Geometry module for land masking and route intersection checks."""

from __future__ import annotations

import json
import logging
import math
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box, mapping, shape
from shapely.prepared import prep
from shapely.strtree import STRtree

from contracts.schemas import EARTH_RADIUS_NM, haversine_nm, initial_bearing_deg

LOGGER = logging.getLogger(__name__)

DEFAULT_SAFETY_BUFFER_NM = 0.5
MIN_DOMAIN_POLYGON_COUNT = 200
DOMAIN_BBOX = box(6.0, 37.0, 15.0, 44.5)

REPO_ROOT = Path(__file__).resolve().parents[3]
OSM_CLIPPED_REPO_PATH = REPO_ROOT / "data" / "coastline" / "land-polygons-complete-4326-clipped.geojson"

CACHE_ROOT = Path.home() / ".cache" / "pmc"
CACHE_COASTLINE = CACHE_ROOT / "coastline"
CACHE_RAW = CACHE_ROOT / "raw"

GSHHG_URL = "https://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip"
GSHHG_RAW_ZIP = CACHE_RAW / "gshhg-shp-2.3.7.zip"
GSHHG_CLIPPED_CACHE = CACHE_COASTLINE / "gshhg-f-l1-clipped.geojson"
NATURAL_EARTH_10M_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_10m_land.geojson"
)
NATURAL_EARTH_RAW = CACHE_COASTLINE / "ne_10m_land.geojson"
NATURAL_EARTH_CLIPPED = CACHE_COASTLINE / "ne_10m_land_clipped.geojson"

_LAND_POLYGONS: list[Polygon] | None = None
_LAND_PREPARED = None
_LAND_INDEX: STRtree | None = None
_LAND_SOURCE: str | None = None
_LAND_SCALE: str | None = None


def _download_once(url: str, target: Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return True
    try:
        with urlopen(url, timeout=90) as response:
            payload = response.read()
        target.write_bytes(payload)
        return True
    except (URLError, TimeoutError, OSError):
        return False


def _clip_to_domain(geom) -> list[Polygon]:
    if geom.is_empty:
        return []
    clipped = geom.intersection(DOMAIN_BBOX)
    if clipped.is_empty:
        return []
    if clipped.geom_type == "Polygon":
        return [clipped]
    if clipped.geom_type == "MultiPolygon":
        return [poly for poly in clipped.geoms if not poly.is_empty]
    if hasattr(clipped, "geoms"):
        return [g for g in clipped.geoms if g.geom_type == "Polygon" and not g.is_empty]
    return []


def _load_geojson_polygons(path: Path) -> list[Polygon]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    features = raw.get("features", [])
    polygons: list[Polygon] = []
    for feat in features:
        geom_data = feat.get("geometry")
        if not geom_data:
            continue
        geom = shape(geom_data)
        if not geom.is_valid:
            geom = geom.buffer(0)
        polygons.extend(_clip_to_domain(geom))
    return polygons


def _write_geojson(path: Path, polygons: list[Polygon], name: str) -> None:
    features = [
        {"type": "Feature", "properties": {}, "geometry": mapping(poly)} for poly in polygons
    ]
    fc = {"type": "FeatureCollection", "name": name, "features": features}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fc, separators=(",", ":"), ensure_ascii=True), encoding="utf-8")


def _extract_gshhg_f_l1(shp_root: Path) -> Path | None:
    for candidate in shp_root.rglob("GSHHS_f_L1.shp"):
        return candidate
    return None


def _build_gshhg_clipped_cache() -> bool:
    if GSHHG_CLIPPED_CACHE.exists():
        return True
    if not _download_once(GSHHG_URL, GSHHG_RAW_ZIP):
        return False

    try:
        import shapefile  # pyshp
    except ImportError:
        return False

    extract_dir = CACHE_RAW / "gshhg-shp-2.3.7"
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(GSHHG_RAW_ZIP) as zf:
            zf.extractall(extract_dir)

    shp_path = _extract_gshhg_f_l1(extract_dir)
    if shp_path is None:
        return False

    polygons: list[Polygon] = []
    reader = shapefile.Reader(str(shp_path))
    for shp in reader.shapes():
        xmin, ymin, xmax, ymax = shp.bbox
        if xmax < 6.0 or xmin > 15.0 or ymax < 37.0 or ymin > 44.5:
            continue
        geom = shape(shp.__geo_interface__)
        if not geom.is_valid:
            geom = geom.buffer(0)
        polygons.extend(_clip_to_domain(geom))
    if not polygons:
        return False
    _write_geojson(GSHHG_CLIPPED_CACHE, polygons, "gshhg-f-l1-clipped")
    return True


def _build_natural_earth_10m_clipped_cache() -> bool:
    if NATURAL_EARTH_CLIPPED.exists():
        return True
    if not NATURAL_EARTH_RAW.exists():
        if not _download_once(NATURAL_EARTH_10M_URL, NATURAL_EARTH_RAW):
            return False
    polygons = _load_geojson_polygons(NATURAL_EARTH_RAW)
    if not polygons:
        return False
    _write_geojson(NATURAL_EARTH_CLIPPED, polygons, "natural-earth-10m-clipped")
    return True


def _load_land_polygons() -> tuple[list[Polygon], str, str]:
    if OSM_CLIPPED_REPO_PATH.exists():
        polygons = _load_geojson_polygons(OSM_CLIPPED_REPO_PATH)
        if polygons:
            return polygons, str(OSM_CLIPPED_REPO_PATH), "OSM land-polygons-complete-4326 (ODbL)"

    if _build_gshhg_clipped_cache():
        polygons = _load_geojson_polygons(GSHHG_CLIPPED_CACHE)
        if polygons:
            return polygons, str(GSHHG_CLIPPED_CACHE), "GSHHG 2.3.7 full (f), level 1"

    if _build_natural_earth_10m_clipped_cache():
        polygons = _load_geojson_polygons(NATURAL_EARTH_CLIPPED)
        if polygons:
            return polygons, str(NATURAL_EARTH_CLIPPED), "Natural Earth 10m physical land"

    raise RuntimeError("No coastline dataset available (OSM/GSHHG/NE-10m).")


def _ensure_land_index() -> tuple[list[Polygon], list, STRtree]:
    global _LAND_POLYGONS, _LAND_PREPARED, _LAND_INDEX, _LAND_SOURCE, _LAND_SCALE
    if _LAND_POLYGONS is None or _LAND_PREPARED is None or _LAND_INDEX is None:
        polygons, source, scale = _load_land_polygons()
        polygon_count = len(polygons)
        if polygon_count <= MIN_DOMAIN_POLYGON_COUNT:
            raise AssertionError(
                "Loaded coastline is too coarse for race geometry: "
                f"polygon_count={polygon_count} (must exceed {MIN_DOMAIN_POLYGON_COUNT})."
            )
        _LAND_POLYGONS = polygons
        _LAND_SOURCE = source
        _LAND_SCALE = scale
        _LAND_PREPARED = [prep(poly) for poly in polygons]
        _LAND_INDEX = STRtree(polygons)
        LOGGER.warning(
            "Loaded coastline source=%s scale=%s polygon_count=%d",
            _LAND_SOURCE,
            _LAND_SCALE,
            polygon_count,
        )
    return _LAND_POLYGONS, _LAND_PREPARED, _LAND_INDEX


def coastline_info() -> dict[str, object]:
    """Return runtime coastline source metadata."""
    polygons, _, _ = _ensure_land_index()
    return {
        "source": _LAND_SOURCE,
        "scale": _LAND_SCALE,
        "polygon_count": len(polygons),
    }


def polygon_count_in_bbox(
    *, lon_min: float, lon_max: float, lat_min: float, lat_max: float
) -> int:
    """Count coastline polygons intersecting the given bounding box."""
    polygons, prepared, index = _ensure_land_index()
    query = box(lon_min, lat_min, lon_max, lat_max)
    idxs = index.query(query, predicate="intersects")
    total = 0
    for idx in idxs:
        if prepared[int(idx)].intersects(query):
            total += 1
    return total


def _buffer_nm_to_degrees(safety_buffer_nm: float, ref_lat_deg: float) -> float:
    lat_deg = safety_buffer_nm / 60.0
    cos_lat = max(0.1, abs(math.cos(math.radians(ref_lat_deg))))
    lon_deg = safety_buffer_nm / (60.0 * cos_lat)
    return max(lat_deg, lon_deg)


def _point_on_land(lat: float, lon: float) -> bool:
    _, prepared, index = _ensure_land_index()
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


_DISTANCE_GRID_LAT: np.ndarray | None = None
_DISTANCE_GRID_LON: np.ndarray | None = None
_DISTANCE_GRID: np.ndarray | None = None
_DISTANCE_GRID_RES = 0.1
_DISTANCE_GRID_CACHE = REPO_ROOT / "data" / "cache" / "distance_to_land_0p1.npz"


def distance_to_land_nm(lat, lon) -> np.ndarray:
    """Great-circle distance from point(s) to the nearest land polygon edge.

    Uses a cached domain raster (0.1°) with vectorised bilinear lookup.
    """
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    lat_b, lon_b = np.broadcast_arrays(lat_arr, lon_arr)
    grid_lat, grid_lon, grid_dist = _ensure_distance_grid()
    return _sample_distance_grid_vec(grid_lat, grid_lon, grid_dist, lat_b, lon_b)


def _ensure_distance_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global _DISTANCE_GRID_LAT, _DISTANCE_GRID_LON, _DISTANCE_GRID
    if _DISTANCE_GRID is not None:
        assert _DISTANCE_GRID_LAT is not None and _DISTANCE_GRID_LON is not None
        return _DISTANCE_GRID_LAT, _DISTANCE_GRID_LON, _DISTANCE_GRID

    if _DISTANCE_GRID_CACHE.exists():
        payload = np.load(_DISTANCE_GRID_CACHE)
        _DISTANCE_GRID_LAT = payload["lat"]
        _DISTANCE_GRID_LON = payload["lon"]
        _DISTANCE_GRID = payload["dist"]
        return _DISTANCE_GRID_LAT, _DISTANCE_GRID_LON, _DISTANCE_GRID

    from shapely.ops import nearest_points

    polygons, _, index = _ensure_land_index()
    lats = np.round(np.arange(37.0, 44.5 + 1e-9, _DISTANCE_GRID_RES), 4)
    lons = np.round(np.arange(6.0, 15.0 + 1e-9, _DISTANCE_GRID_RES), 4)
    dist = np.empty((lats.size, lons.size), dtype=np.float32)
    LOGGER.warning(
        "Building distance-to-land raster %dx%d at %.2f deg (one-time)",
        lats.size,
        lons.size,
        _DISTANCE_GRID_RES,
    )
    for i, lat_v in enumerate(lats):
        for j, lon_v in enumerate(lons):
            point = Point(float(lon_v), float(lat_v))
            nearest = index.nearest(point)
            idx = int(nearest[0]) if isinstance(nearest, (list, tuple, np.ndarray)) else int(nearest)
            poly = polygons[idx]
            if poly.contains(point) or poly.touches(point):
                dist[i, j] = 0.0
                continue
            _, land_pt = nearest_points(point, poly)
            dist[i, j] = float(haversine_nm(lat_v, lon_v, land_pt.y, land_pt.x))
    _DISTANCE_GRID_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_DISTANCE_GRID_CACHE, lat=lats, lon=lons, dist=dist)
    _DISTANCE_GRID_LAT = lats
    _DISTANCE_GRID_LON = lons
    _DISTANCE_GRID = dist
    return lats, lons, dist


def _sample_distance_grid_vec(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    grid_dist: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    lat_c = np.clip(lat, grid_lat[0], grid_lat[-1])
    lon_c = np.clip(lon, grid_lon[0], grid_lon[-1])
    i1 = np.searchsorted(grid_lat, lat_c, side="right")
    j1 = np.searchsorted(grid_lon, lon_c, side="right")
    i0 = np.clip(i1 - 1, 0, grid_lat.size - 1)
    j0 = np.clip(j1 - 1, 0, grid_lon.size - 1)
    i1 = np.clip(np.maximum(i1, i0 + 1), 0, grid_lat.size - 1)
    j1 = np.clip(np.maximum(j1, j0 + 1), 0, grid_lon.size - 1)
    lat0 = grid_lat[i0]
    lat1 = grid_lat[i1]
    lon0 = grid_lon[j0]
    lon1 = grid_lon[j1]
    with np.errstate(divide="ignore", invalid="ignore"):
        wa = np.where(lat1 == lat0, 0.0, (lat_c - lat0) / (lat1 - lat0))
        wb = np.where(lon1 == lon0, 0.0, (lon_c - lon0) / (lon1 - lon0))
    return (
        (1 - wa) * (1 - wb) * grid_dist[i0, j0]
        + wa * (1 - wb) * grid_dist[i1, j0]
        + (1 - wa) * wb * grid_dist[i0, j1]
        + wa * wb * grid_dist[i1, j1]
    )
