#!/usr/bin/env python3
"""Embed z7 OSM tiles and a clipped coastline so the dashboard works on file://."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import requests

from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent
TILES = ROOT / "tiles"
BBOX = box(6.5, 37.5, 14.5, 44.0)
NE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_10m_land.geojson"
)
NE_CACHE = Path.home() / ".cache" / "pmc" / "coastline" / "ne_10m_land.geojson"


def embed_osm_z7() -> dict:
    zoom = 7
    x_min, x_max, y_min, y_max = 66, 69, 46, 49
    embedded = []
    total = 0
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            path = TILES / "osm" / str(zoom) / str(x) / f"{y}.png"
            if not path.exists() or path.stat().st_size < 200:
                continue
            raw = path.read_bytes()
            total += len(raw)
            embedded.append(
                {
                    "z": zoom,
                    "x": x,
                    "y": y,
                    "source_data_uri": "data:image/png;base64," + base64.b64encode(raw).decode("ascii"),
                }
            )
    manifest = {
        "name": "OpenStreetMap base map (CARTO Voyager)",
        "attribution": "OpenStreetMap contributors, ODbL. Tiles by CARTO.",
        "warning": "Base map uses OpenStreetMap data via CARTO. Not for navigation.",
        "bbox": {"lon_min": 6.5, "lon_max": 14.5, "lat_min": 37.5, "lat_max": 44.0},
        "included_zooms": [7],
        "zoom_ranges": {
            "7": {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "count": len(embedded),
            }
        },
        "total_bytes": total,
        "tile_count": len(embedded),
        "tile_url_template": "tiles/osm/{z}/{x}/{y}.png",
        "embedded_tiles": embedded,
    }
    (TILES / "osm_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    slim = {k: v for k, v in manifest.items() if k != "embedded_tiles"}
    slim["embedded_tiles"] = embedded
    (TILES / "osm_manifest.js").write_text(
        "window.OPEN_STREET_MAP_MANIFEST = "
        + json.dumps(slim, separators=(",", ":"), ensure_ascii=True)
        + ";\n",
        encoding="utf-8",
    )
    return manifest


def _rings(geom) -> list[list[list[float]]]:
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [list(geom.exterior.coords)]
    rings: list[list[list[float]]] = []
    if hasattr(geom, "geoms"):
        for part in geom.geoms:
            rings.extend(_rings(part))
    return rings


def write_coastline() -> int:
    NE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not NE_CACHE.exists():
        resp = requests.get(NE_URL, timeout=90)
        resp.raise_for_status()
        NE_CACHE.write_bytes(resp.content)
    raw = json.loads(NE_CACHE.read_text(encoding="utf-8"))
    pieces = []
    for feat in raw.get("features", []):
        geom = shape(feat.get("geometry"))
        if geom.is_empty:
            continue
        clipped = geom.intersection(BBOX)
        if not clipped.is_empty:
            pieces.append(clipped)
    if not pieces:
        raise RuntimeError("No land polygons in domain")
    merged = unary_union(pieces).simplify(0.02, preserve_topology=True)
    rings = []
    for ring in _rings(merged):
        if len(ring) < 4:
            continue
        rings.append([[round(x, 4), round(y, 4)] for x, y in ring])
    payload = {
        "attribution": "Natural Earth 10m land, public domain",
        "rings": rings,
    }
    (TILES / "land.js").write_text(
        "window.LAND_RINGS = " + json.dumps(payload, separators=(",", ":"), ensure_ascii=True) + ";\n",
        encoding="utf-8",
    )
    return len(rings)


def main() -> int:
    manifest = embed_osm_z7()
    n_rings = write_coastline()
    print(
        f"Embedded {manifest['tile_count']} z7 OSM tiles "
        f"({manifest['total_bytes'] / 1024:.0f} KB), {n_rings} coastline rings"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
