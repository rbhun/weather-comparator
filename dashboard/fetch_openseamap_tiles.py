#!/usr/bin/env python3
"""Download and vendor OpenSeaMap tiles for offline dashboard use."""

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
BASE_URL = "https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png"


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


def tile_lon(x: int, z: int) -> float:
    return x / (2**z) * 360.0 - 180.0


def tile_lat(y: int, z: int) -> float:
    n = math.pi - 2.0 * math.pi * y / (2**z)
    return math.degrees(math.atan(math.sinh(n)))


def tiles_for_bbox(z: int) -> Iterable[Tile]:
    x_min = lon_to_tile_x(LON_MIN, z)
    x_max = lon_to_tile_x(LON_MAX, z)
    y_min = lat_to_tile_y(LAT_MAX, z)  # north
    y_max = lat_to_tile_y(LAT_MIN, z)  # south
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            yield Tile(z=z, x=x, y=y)


def download_tiles(root: Path) -> list[dict]:
    session = requests.Session()
    downloaded: list[dict] = []
    for z in ZOOMS:
        for tile in tiles_for_bbox(z):
            out_path = root / str(tile.z) / str(tile.x) / f"{tile.y}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists():
                url = BASE_URL.format(z=tile.z, x=tile.x, y=tile.y)
                for attempt in range(5):
                    resp = session.get(url, timeout=20)
                    if resp.status_code == 200 and resp.content:
                        out_path.write_bytes(resp.content)
                        break
                    if attempt == 4:
                        raise RuntimeError(
                            f"Failed tile {tile.z}/{tile.x}/{tile.y}, status {resp.status_code}"
                        )
                    time.sleep(0.5 * (2**attempt))
                # Be polite with the tile server.
                time.sleep(0.03)
            downloaded.append(
                {
                    "z": tile.z,
                    "x": tile.x,
                    "y": tile.y,
                    "path": f"tiles/openseamap/{tile.z}/{tile.x}/{tile.y}.png",
                }
            )
    return downloaded


def directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def remove_zoom(root: Path, z: int) -> None:
    zoom_dir = root / str(z)
    if zoom_dir.exists():
        shutil.rmtree(zoom_dir)


def write_manifest(root: Path, downloaded: list[dict]) -> dict:
    existing = []
    for row in downloaded:
        p = root / str(row["z"]) / str(row["x"]) / f"{row['y']}.png"
        if p.exists():
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

    total_bytes = directory_size(root)
    manifest = {
        "name": "OpenSeaMap seamark chart layer",
        "attribution": "OpenStreetMap contributors, ODbL",
        "warning": "Chart layer is OpenSeaMap, crowd-sourced, NOT for navigation.",
        "bbox": {"lon_min": LON_MIN, "lon_max": LON_MAX, "lat_min": LAT_MIN, "lat_max": LAT_MAX},
        "included_zooms": included_zooms,
        "zoom_ranges": by_zoom,
        "total_bytes": total_bytes,
        "tile_count": len(existing),
        "tile_url_template": "tiles/openseamap/{z}/{x}/{y}.png",
    }
    (root.parent / "openseamap_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_js = (
        "window.OPEN_SEA_MAP_MANIFEST = "
        + json.dumps(manifest, separators=(",", ":"), ensure_ascii=True)
        + ";\n"
    )
    (root.parent / "openseamap_manifest.js").write_text(manifest_js, encoding="utf-8")
    return manifest


def main() -> int:
    tiles_root = Path(__file__).resolve().parent / "tiles" / "openseamap"
    tiles_root.mkdir(parents=True, exist_ok=True)
    downloaded = download_tiles(tiles_root)

    while True:
        current_bytes = directory_size(tiles_root)
        if current_bytes <= MAX_TOTAL_BYTES:
            break
        existing_zooms = sorted(
            int(p.name) for p in tiles_root.iterdir() if p.is_dir() and p.name.isdigit()
        )
        if not existing_zooms:
            break
        drop_zoom = max(existing_zooms)
        remove_zoom(tiles_root, drop_zoom)
        print(f"Dropped zoom {drop_zoom} to stay under 30 MB.")

    manifest = write_manifest(tiles_root, downloaded)
    print(
        "OpenSeaMap tiles ready: "
        f"{manifest['tile_count']} tiles, "
        f"{manifest['total_bytes'] / (1024 * 1024):.2f} MB, "
        f"zooms={manifest['included_zooms']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
