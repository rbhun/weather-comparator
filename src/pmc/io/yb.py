"""YB Tracking client for Palermo–Montecarlo historical tracks."""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

KML_NS = {
    "kml": "http://www.opengis.net/kml/2.2",
    "gx": "http://www.google.com/kml/ext/2.2",
}
DEFAULT_YEARS = (2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025)
DEFAULT_CACHE = Path("data/cache/yb")
SKIP_NAME_RE = re.compile(
    r"^(committee|race\s*comit+e+e|controstarter|mark\s*\d+|riserva(\s+\d+)?|markers?|null)$",
    re.I,
)
ABSOLUTE_CLASS_NAMES = {"line honours", "all boats"}
NON_CLASS_SECTIONS = {"committee", "markers", "race markers", "null"}
USER_AGENT = (
    "Mozilla/5.0 (compatible; pmc-yb/0.1; +https://github.com/) "
    "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
)


def slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "", value.lower())
    return text or "class"


def norm_name(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).upper()


def is_skip_name(name: str) -> bool:
    return bool(SKIP_NAME_RE.match((name or "").strip()))


def is_absolute_class(name: str) -> bool:
    return name.strip().lower() in ABSOLUTE_CLASS_NAMES


@dataclass
class ClassResult:
    class_id: str
    class_name: str
    rank: int | None
    finished: bool
    status: str
    tcf: float | None
    finish_utc: str | None
    dtf_nm: float | None


@dataclass
class Boat:
    year: int
    name: str
    status: str
    finished: bool
    absolute_rank: int | None
    elapsed_s: int | None
    elapsed_label: str | None
    start_utc: str | None
    finish_utc: str | None
    classes: list[ClassResult] = field(default_factory=list)
    lon: list[float] = field(default_factory=list)
    lat: list[float] = field(default_factory=list)


def _http_get(url: str, dest: Path | None = None, timeout: int = 120) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Referer": "https://yb.tl/"})
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if dest is not None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            return data
        except (HTTPError, URLError, TimeoutError) as exc:
            last_exc = exc
            time.sleep(1.5 * (2**attempt))
    raise RuntimeError(f"Failed to fetch {url}: {last_exc}") from last_exc


def fetch_year(year: int, cache_root: Path = DEFAULT_CACHE, refresh: bool = False) -> Path:
    """Download leaderboard CSV and KML tracks for one edition."""

    slug = f"pm{year}"
    folder = cache_root / slug
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "leaderboard.csv"
    kml_path = folder / "tracks.kml"
    links_path = folder / "links.html"
    if refresh or not links_path.exists() or links_path.stat().st_size < 200:
        _http_get(f"https://yb.tl/Links/{slug}", links_path)
        time.sleep(0.4)
    if refresh or not csv_path.exists() or csv_path.stat().st_size < 200:
        _http_get(f"https://yb.tl/l/{slug}", csv_path)
        time.sleep(0.4)
    if refresh or not kml_path.exists() or kml_path.stat().st_size < 50_000:
        _http_get(f"https://yb.tl/{slug}.kml", kml_path)
        time.sleep(0.6)
    return folder


def parse_leaderboard_csv(text: str, year: int) -> dict[str, Boat]:
    """Parse the all-classes YB CSV into boats keyed by normalised name."""

    boats: dict[str, Boat] = {}
    section = None
    reader = csv.reader(io.StringIO(text))
    for raw in reader:
        if not raw:
            continue
        first = (raw[0] or "").strip()
        if not first or first.startswith("THESE RESULTS") or first.startswith("Rank"):
            continue
        if len(raw) == 1 or (len(raw) >= 2 and not (raw[1] or "").strip() and first):
            section = first
            continue
        if first.lower() == "null" or (section or "").lower() in NON_CLASS_SECTIONS:
            continue
        if section is None:
            continue
        name = (raw[1] if len(raw) > 1 else "").strip()
        if not name or is_skip_name(name):
            continue
        status = "RETIRED" if first.upper() in {"RTD", "DNF", "DNS", "DSQ"} else "FINISHED"
        rank = None
        if first.isdigit():
            rank = int(first)
        finished = status == "FINISHED" and rank is not None
        tcf = _maybe_float(raw[2] if len(raw) > 2 else None)
        finish_utc = _parse_yb_time(raw[3] if len(raw) > 3 else "")
        dtf = _maybe_float(raw[9] if len(raw) > 9 else None)
        if finished and dtf is not None and dtf > 1.0:
            finished = False
            status = "RACING"
        key = norm_name(name)
        boat = boats.get(key)
        if boat is None:
            boat = Boat(
                year=year,
                name=name,
                status=status,
                finished=finished,
                absolute_rank=rank if is_absolute_class(section) else None,
                elapsed_s=None,
                elapsed_label=None,
                start_utc=None,
                finish_utc=finish_utc if finished else None,
            )
            boats[key] = boat
        else:
            if is_absolute_class(section) and boat.absolute_rank is None:
                boat.absolute_rank = rank
            if finished:
                boat.finished = True
                boat.status = "FINISHED"
                if finish_utc:
                    boat.finish_utc = finish_utc
            elif boat.status == "FINISHED":
                pass
            else:
                boat.status = status
        if (section or "").lower() in NON_CLASS_SECTIONS:
            continue
        class_name = "Line Honours" if section.lower() == "all boats" else section
        class_id = slugify(class_name)
        if any(item.class_id == class_id for item in boat.classes):
            continue
        boat.classes.append(
            ClassResult(
                class_id=class_id,
                class_name=class_name,
                rank=rank,
                finished=finished,
                status=status,
                tcf=tcf,
                finish_utc=finish_utc,
                dtf_nm=dtf,
            )
        )
    return boats


def parse_kml_tracks(kml_bytes: bytes) -> dict[str, tuple[list[str], list[float], list[float]]]:
    """Return name -> (iso times, lon, lat) from a YB gx:Track KML."""

    root = ET.fromstring(kml_bytes)
    tracks: dict[str, tuple[list[str], list[float], list[float]]] = {}
    for placemark in root.findall(".//kml:Placemark", KML_NS):
        name_el = placemark.find("kml:name", KML_NS)
        name = (name_el.text or "").strip() if name_el is not None else ""
        if not name or name.lower() == "tracks" or is_skip_name(name):
            continue
        track = placemark.find("gx:Track", KML_NS)
        if track is None:
            continue
        times = [el.text.strip() for el in track.findall("kml:when", KML_NS) if el.text]
        lons: list[float] = []
        lats: list[float] = []
        for coord in track.findall("gx:coord", KML_NS):
            parts = (coord.text or "").replace(",", " ").split()
            if len(parts) < 2:
                continue
            lons.append(float(parts[0]))
            lats.append(float(parts[1]))
        if len(lons) < 2:
            continue
        if times and len(times) != len(lons):
            n = min(len(times), len(lons))
            times, lons, lats = times[:n], lons[:n], lats[:n]
        tracks[norm_name(name)] = (times, lons, lats)
    return tracks


def downsample_track(
    lons: list[float],
    lats: list[float],
    times: list[str] | None = None,
    max_points: int = 140,
) -> tuple[list[float], list[float], list[str] | None]:
    n = len(lons)
    if n <= max_points:
        out_lon = [round(v, 4) for v in lons]
        out_lat = [round(v, 4) for v in lats]
        return out_lon, out_lat, times
    step = max(1, math.ceil((n - 1) / (max_points - 1)))
    idxs = list(range(0, n - 1, step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    out_lon = [round(lons[i], 4) for i in idxs]
    out_lat = [round(lats[i], 4) for i in idxs]
    out_times = [times[i] for i in idxs] if times and len(times) == n else None
    return out_lon, out_lat, out_times


def _elapsed_label(seconds: int) -> str:
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h {minutes:02d}m"
    return f"{hours}h {minutes:02d}m"


def attach_tracks(boats: dict[str, Boat], tracks: dict[str, tuple[list[str], list[float], list[float]]]) -> None:
    for key, boat in boats.items():
        payload = tracks.get(key)
        if payload is None:
            # tolerate minor punctuation differences
            compact = re.sub(r"[^A-Z0-9]", "", key)
            for tkey, value in tracks.items():
                if re.sub(r"[^A-Z0-9]", "", tkey) == compact:
                    payload = value
                    break
        if payload is None:
            continue
        times, lons, lats = payload
        lons, lats, times = downsample_track(lons, lats, times)
        boat.lon = lons
        boat.lat = lats
        if times:
            boat.start_utc = times[0].replace("Z", "") + "Z" if not times[0].endswith("Z") else times[0]
            if boat.finished:
                boat.finish_utc = boat.finish_utc or (
                    times[-1].replace("Z", "") + "Z" if not times[-1].endswith("Z") else times[-1]
                )
            if boat.start_utc and boat.finish_utc:
                start = _parse_iso(boat.start_utc)
                finish = _parse_iso(boat.finish_utc)
                if start and finish and finish > start:
                    boat.elapsed_s = int((finish - start).total_seconds())
                    boat.elapsed_label = _elapsed_label(boat.elapsed_s)


def load_edition(year: int, cache_root: Path = DEFAULT_CACHE) -> list[Boat]:
    folder = cache_root / f"pm{year}"
    csv_text = (folder / "leaderboard.csv").read_text(encoding="utf-8", errors="replace")
    boats = parse_leaderboard_csv(csv_text, year)
    kml_path = folder / "tracks.kml"
    if kml_path.exists():
        attach_tracks(boats, parse_kml_tracks(kml_path.read_bytes()))
    return [boat for boat in boats.values() if boat.lon]


def build_overlay(years: tuple[int, ...] = DEFAULT_YEARS, cache_root: Path = DEFAULT_CACHE) -> dict[str, Any]:
    editions: list[dict[str, Any]] = []
    for year in years:
        boats = load_edition(year, cache_root)
        class_names: dict[str, str] = {}
        for boat in boats:
            for item in boat.classes:
                class_names[item.class_id] = item.class_name
        starts = [b.start_utc for b in boats if b.start_utc]
        finishes = [b.finish_utc for b in boats if b.finished and b.finish_utc]
        editions.append(
            {
                "year": year,
                "slug": f"pm{year}",
                "title": f"Palermo–Montecarlo {year}",
                "source": f"https://yb.tl/pm{year}",
                "start_utc": min(starts) if starts else None,
                "end_utc": max(finishes) if finishes else None,
                "classes": [
                    {"id": cid, "name": class_names[cid], "absolute": is_absolute_class(class_names[cid])}
                    for cid in sorted(class_names, key=lambda k: (not is_absolute_class(class_names[k]), class_names[k]))
                ],
                "boats": [_boat_to_json(boat) for boat in sorted(boats, key=_boat_sort)],
            }
        )
    return {
        "source": "YB Tracking",
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "filter_notes": {
            "absolute": "Line honours / elapsed time, uncorrected.",
            "per_class": "YB class rank as published (handicap classes are corrected).",
        },
        "editions": editions,
    }


def write_overlay(overlay: dict[str, Any], json_path: Path, js_path: Path) -> None:
    text = json.dumps(overlay, ensure_ascii=False, separators=(",", ":"))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(text + "\n", encoding="utf-8")
    js_path.write_text("window.YB_RESULTS = " + text + ";\n", encoding="utf-8")


def _boat_sort(boat: Boat) -> tuple[int, int, str]:
    rank = boat.absolute_rank if boat.absolute_rank is not None else 999
    return (0 if boat.finished else 1, rank, boat.name)


def _boat_to_json(boat: Boat) -> dict[str, Any]:
    return {
        "id": f"{boat.year}-{slugify(boat.name)}",
        "name": boat.name,
        "status": boat.status,
        "finished": boat.finished,
        "absolute_rank": boat.absolute_rank,
        "elapsed_s": boat.elapsed_s,
        "elapsed_label": boat.elapsed_label,
        "start_utc": boat.start_utc,
        "finish_utc": boat.finish_utc,
        "classes": [
            {
                "id": item.class_id,
                "name": item.class_name,
                "rank": item.rank,
                "finished": item.finished,
            }
            for item in boat.classes
        ],
        "lon": boat.lon,
        "lat": boat.lat,
    }


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_yb_time(value: str) -> str | None:
    text = (value or "").strip()
    if not text:
        return None
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            parsed = dt.datetime.strptime(text, fmt).replace(tzinfo=dt.timezone.utc)
            return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
    return None


def _parse_iso(value: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
