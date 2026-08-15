#!/usr/bin/env python3
"""Download and vendor OSM + OpenSeaMap tiles for offline dashboard use."""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

LON_MIN = 6.5
LON_MAX = 14.5
LAT_MIN = 37.5
LAT_MAX = 44.0
ZOOMS = (7, 8, 9)
MAX_TOTAL_BYTES = 30 * 1024 * 1024
USER_AGENT = "pmc-weather-comparator/1.0 (offline race dashboard tile cache)"

LAYERS = {
    "osm": {
        "name": "OpenStreetMap base map (CARTO Voyager)",
        "attribution": "OpenStreetMap contributors, ODbL. Tiles by CARTO.",
        "url": "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "warning": "Base map uses OpenStreetMap data via CARTO. Not for navigation.",
    },
    "openseamap": {
        "name": "OpenSeaMap seamark overlay",
        "attribution": "OpenStreetMap contributors, ODbL",
        "url": "https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png",
        "warning": "Chart layer is OpenSeaMap, crowd-sourced, NOT for navigation.",
    },
}


@dataclass(frozen=True)
class Tile:
    z: int
    x: int
    y: int


def lon_to_tile_x(lon_deg: float, z: int) -> int:
    return int((lon_deg + 180.0) / 360.0 * (2**z))


def lat_to_tile_y(lat_deg: float, z: int) -> int:
    lat_rad = math.radians(lat_deg)
    n = 2**z
    return int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)


def tiles_for_bbox(z: int) -> Iterable[Tile]:
    x_min = lon_to_tile_x(LON_MIN, z)
    x_max = lon_to_tile_x(LON_MAX, z)
    y_min = lat_to_tile_y(LAT_MAX, z)  # north
    y_max = lat_to_tile_y(LAT_MIN, z)  # south
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            yield Tile(z=z, x=x, y=y)


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def remove_zoom(root: Path, z: int) -> None:
    zoom_dir = root / str(z)
    if zoom_dir.exists():
        shutil.rmtree(zoom_dir)


def download_layer(session: requests.Session, layer_id: str, root: Path) -> list[dict]:
    url_template = LAYERS[layer_id]["url"]
    downloaded: list[dict] = []
    for z in ZOOMS:
        for tile in tiles_for_bbox(z):
            out_path = root / str(tile.z) / str(tile.x) / f"{tile.y}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists() or out_path.stat().st_size < 200:
                url = url_template.format(z=tile.z, x=tile.x, y=tile.y)
                for attempt in range(7):
                    resp = session.get(url, timeout=20)
                    if resp.status_code == 200 and resp.content and len(resp.content) > 200:
                        out_path.write_bytes(resp.content)
                        break
                    if attempt == 6:
                        print(
                            f"Skip tile {layer_id} {tile.z}/{tile.x}/{tile.y} "
                            f"(status {resp.status_code}, {len(resp.content)} bytes)"
                        )
                    else:
                        time.sleep(min(8.0, 0.6 * (2**attempt)))
                time.sleep(0.08)
            if out_path.exists():
                downloaded.append(
                    {
                        "z": tile.z,
                        "x": tile.x,
                        "y": tile.y,
                        "path": f"tiles/{layer_id}/{tile.z}/{tile.x}/{tile.y}.png",
                    }
                )
    return downloaded


def write_manifest(layer_id: str, root: Path, downloaded: list[dict]) -> dict:
    existing = []
    for row in downloaded:
        p = root / str(row["z"]) / str(row["x"]) / f"{row['y']}.png"
        if p.exists() and p.stat().st_size > 200:
            existing.append(row)

    included_zooms = sorted({int(row["z"]) for row in existing})
    by_zoom: dict[int, dict[str, int]] = {}
    for z in included_zooms:
        zrows = [r for r in existing if int(r["z"]) == z]
        xs = [int(r["x"]) for r in zrows]
        ys = [int(r["y"]) for r in zrows]
        by_zoom[z] = {
            "x_min": min(xs),
            "x_max": max(xs),
            "y_min": min(ys),
            "y_max": max(ys),
            "count": len(zrows),
        }

    spec = LAYERS[layer_id]
    manifest = {
        "name": spec["name"],
        "attribution": spec["attribution"],
        "warning": spec["warning"],
        "bbox": {"lon_min": LON_MIN, "lon_max": LON_MAX, "lat_min": LAT_MIN, "lat_max": LAT_MAX},
        "included_zooms": included_zooms,
        "zoom_ranges": {str(z): by_zoom[z] for z in included_zooms},
        "total_bytes": directory_size(root),
        "tile_count": len(existing),
        "tile_url_template": f"tiles/{layer_id}/{{z}}/{{x}}/{{y}}.png",
    }
    tiles_parent = root.parent
    (tiles_parent / f"{layer_id}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    js_name = "OPEN_STREET_MAP_MANIFEST" if layer_id == "osm" else "OPEN_SEA_MAP_MANIFEST"
    (tiles_parent / f"{layer_id}_manifest.js").write_text(
        f"window.{js_name} = {json.dumps(manifest, separators=(',', ':'), ensure_ascii=True)};\n",
        encoding="utf-8",
    )
    return manifest


def enforce_budget(root: Path) -> None:
    while directory_size(root) > MAX_TOTAL_BYTES:
        existing_zooms = sorted(
            int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()
        )
        if not existing_zooms:
            break
        drop_zoom = max(existing_zooms)
        remove_zoom(root, drop_zoom)
        print(f"Dropped zoom {drop_zoom} in {root.name} to stay under 30 MB.")


def main() -> int:
    tiles_root = Path(__file__).resolve().parent / "tiles"
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "image/png"})

    for layer_id in ("osm", "openseamap"):
        layer_root = tiles_root / layer_id
        layer_root.mkdir(parents=True, exist_ok=True)
        downloaded = download_layer(session, layer_id, layer_root)
        enforce_budget(layer_root)
        manifest = write_manifest(layer_id, layer_root, downloaded)
        print(
            f"{layer_id} ready: {manifest['tile_count']} tiles, "
            f"{manifest['total_bytes'] / (1024 * 1024):.2f} MB, "
            f"zooms={manifest['included_zooms']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
