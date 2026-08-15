#!/usr/bin/env python3
"""Same-day resolution comparison: AROME HD vs IFS 9 km analysis.

August 2023–2025 only. Not a climatology — paired difference on identical days.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MS_TO_KT = 1.9438445
NM_PER_DEG_LAT = 60.0
CACHE_DIR = Path("data/cache/resolution_cmp")
OUT_JSON = Path("data/resolution/arome_vs_ifs_aug2023_2025.json")
OUT_MD = Path("docs/arome-vs-ifs-resolution.md")

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
HIST_FC = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Cross-shore cuts aligned to user latitudes / project bearings.
TRANSECTS = (
    {
        "id": "sardinia_e",
        "label": "Sardinia East 40.5N",
        "coast_lat": 40.5,
        "coast_lon": 9.72,  # east-coast shoreline approx
        "bearing_deg": 90.0,  # offshore = east
    },
    {
        "id": "liguria",
        "label": "Ligurian 43.4N cut (coast ~43.70N)",
        "coast_lat": 43.70,  # Riviera shoreline (Nice–Monaco band)
        "coast_lon": 7.80,
        "bearing_deg": 180.0,  # offshore = south (through 43.4N)
    },
)

# Distances (nm) from shore along offshore bearing.
DISTANCES_NM = (0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0)
YEARS = (2023, 2024, 2025)
LOCAL_HOURS_FOCUS = (14, 15, 16, 17, 18)  # Europe/Rome CEST in August = UTC+2
UTC_OFFSET_AUG = 2  # CEST


@dataclass(frozen=True)
class Point:
    transect_id: str
    distance_nm: float
    lat: float
    lon: float


def destination(lat: float, lon: float, bearing_deg: float, distance_nm: float) -> tuple[float, float]:
    """Great-circle destination; Earth radius in nm."""
    r = 3440.065
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    d = distance_nm / r
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(brng))
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), (math.degrees(lon2) + 540.0) % 360.0 - 180.0


def build_points() -> list[Point]:
    points: list[Point] = []
    for t in TRANSECTS:
        for d in DISTANCES_NM:
            lat, lon = destination(t["coast_lat"], t["coast_lon"], t["bearing_deg"], d)
            points.append(Point(t["id"], d, round(lat, 5), round(lon, 5)))
    return points


def _cache_path(tag: str, params: dict) -> Path:
    key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    return CACHE_DIR / f"{tag}_{key}.json"


def fetch_json(url: str, params: dict, tag: str, retries: int = 6) -> list[dict] | dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(tag, {"url": url, **params})
    if path.exists():
        return json.loads(path.read_text())

    full = url + "?" + urllib.parse.urlencode(params)
    delay = 1.0
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(full, timeout=120) as resp:
                raw = resp.read().decode()
            data = json.loads(raw)
            path.write_text(json.dumps(data))
            return data
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:200]
            last_err = RuntimeError(f"HTTP {e.code}: {body}")
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise last_err from e
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError(f"fetch failed {tag}: {last_err}")


def fetch_month(
    model: str,
    endpoint: str,
    year: int,
    points: list[Point],
    batch_size: int = 8,
) -> dict[tuple[str, float], dict[str, np.ndarray]]:
    """Return {(transect_id, dist): {time, speed_kt, direction_deg}} for one August."""
    start = f"{year}-08-01"
    end = f"{year}-08-31"
    out: dict[tuple[str, float], dict[str, np.ndarray]] = {}

    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        params = {
            "latitude": ",".join(str(p.lat) for p in batch),
            "longitude": ",".join(str(p.lon) for p in batch),
            "hourly": "wind_speed_10m,wind_direction_10m",
            "models": model,
            "start_date": start,
            "end_date": end,
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        }
        tag = f"{model}_{year}_{i}"
        data = fetch_json(endpoint, params, tag)
        rows = data if isinstance(data, list) else [data]
        if len(rows) != len(batch):
            raise RuntimeError(f"expected {len(batch)} locations, got {len(rows)} for {tag}")
        for p, row in zip(batch, rows):
            hourly = row.get("hourly") or {}
            times = np.array(hourly.get("time") or [])
            spd = np.array(hourly.get("wind_speed_10m") or [], dtype=float)
            direc = np.array(hourly.get("wind_direction_10m") or [], dtype=float)
            if spd.size == 0:
                continue
            # Open-Meteo may return nulls as None → nan
            spd = spd.astype(float)
            direc = direc.astype(float)
            out[(p.transect_id, p.distance_nm)] = {
                "time": times,
                "speed_kt": spd * MS_TO_KT,
                "direction_deg": direc,
                "lat": p.lat,
                "lon": p.lon,
            }
        time.sleep(0.25)
    return out


def local_hour_mask(times: np.ndarray, local_hours: tuple[int, ...]) -> np.ndarray:
    # times are ISO UTC like 2024-08-01T14:00
    hours_utc = np.array([int(t[11:13]) for t in times], dtype=int)
    local = (hours_utc + UTC_OFFSET_AUG) % 24
    return np.isin(local, list(local_hours))


def circular_mean_deg(deg: np.ndarray) -> float:
    rad = np.deg2rad(deg)
    s = np.nanmean(np.sin(rad))
    c = np.nanmean(np.cos(rad))
    if not np.isfinite(s) or not np.isfinite(c):
        return float("nan")
    return float(np.rad2deg(np.arctan2(s, c)) % 360.0)


def angular_diff_deg(a: float, b: float) -> float:
    """Signed smallest difference a - b in (-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def paired_stats(
    ifs: dict[str, np.ndarray],
    arome: dict[str, np.ndarray],
    local_hours: tuple[int, ...] | None = None,
) -> dict:
    t_ifs = ifs["time"]
    t_aro = arome["time"]
    # align on common timestamps
    common = sorted(set(t_ifs.tolist()) & set(t_aro.tolist()))
    if not common:
        return {"n": 0}
    idx_i = {t: i for i, t in enumerate(t_ifs)}
    idx_a = {t: i for i, t in enumerate(t_aro)}
    ii = np.array([idx_i[t] for t in common])
    aa = np.array([idx_a[t] for t in common])
    times = np.array(common)
    spd_i = ifs["speed_kt"][ii]
    spd_a = arome["speed_kt"][aa]
    dir_i = ifs["direction_deg"][ii]
    dir_a = arome["direction_deg"][aa]

    if local_hours is not None:
        m = local_hour_mask(times, local_hours)
        times, spd_i, spd_a, dir_i, dir_a = times[m], spd_i[m], spd_a[m], dir_i[m], dir_a[m]

    valid = np.isfinite(spd_i) & np.isfinite(spd_a) & np.isfinite(dir_i) & np.isfinite(dir_a)
    n = int(valid.sum())
    if n == 0:
        return {"n": 0}
    spd_i, spd_a = spd_i[valid], spd_a[valid]
    dir_i, dir_a = dir_i[valid], dir_a[valid]
    delta = spd_a - spd_i
    dir_delta = np.array([angular_diff_deg(float(a), float(b)) for a, b in zip(dir_a, dir_i)])
    return {
        "n": n,
        "mean_ifs_kt": float(np.mean(spd_i)),
        "mean_arome_kt": float(np.mean(spd_a)),
        "mean_delta_kt": float(np.mean(delta)),  # AROME - IFS
        "median_delta_kt": float(np.median(delta)),
        "p10_delta_kt": float(np.percentile(delta, 10)),
        "p90_delta_kt": float(np.percentile(delta, 90)),
        "mean_dir_ifs": circular_mean_deg(dir_i),
        "mean_dir_arome": circular_mean_deg(dir_a),
        "mean_dir_delta_deg": float(np.mean(dir_delta)),  # AROME - IFS signed
        "mean_abs_dir_delta_deg": float(np.mean(np.abs(dir_delta))),
    }


def by_local_hour(
    ifs: dict[str, np.ndarray],
    arome: dict[str, np.ndarray],
) -> dict[int, dict]:
    out = {}
    for h in range(24):
        out[h] = paired_stats(ifs, arome, local_hours=(h,))
    return out


def main() -> None:
    points = build_points()
    print("Points:")
    for p in points:
        print(f"  {p.transect_id:12s} {p.distance_nm:5.1f} nm  {p.lat:.4f},{p.lon:.4f}")

    # Accumulate per (transect, dist) lists of monthly arrays, then concat
    series: dict[tuple[str, float], dict[str, list]] = {
        (p.transect_id, p.distance_nm): {"ifs": [], "arome": []} for p in points
    }

    for year in YEARS:
        print(f"\n=== Fetching August {year} ===")
        ifs_month = fetch_month("ecmwf_ifs", ARCHIVE, year, points)
        arome_month = fetch_month("meteofrance_arome_france_hd", HIST_FC, year, points)
        for key in series:
            if key not in ifs_month or key not in arome_month:
                print(f"  MISSING {key}")
                continue
            series[key]["ifs"].append(ifs_month[key])
            series[key]["arome"].append(arome_month[key])
            ni = int(np.isfinite(ifs_month[key]["speed_kt"]).sum())
            na = int(np.isfinite(arome_month[key]["speed_kt"]).sum())
            print(f"  {key}: IFS {ni}/744  AROME {na}/744")

    def concat_model(chunks: list[dict]) -> dict[str, np.ndarray]:
        return {
            "time": np.concatenate([c["time"] for c in chunks]),
            "speed_kt": np.concatenate([c["speed_kt"] for c in chunks]),
            "direction_deg": np.concatenate([c["direction_deg"] for c in chunks]),
            "lat": chunks[0]["lat"],
            "lon": chunks[0]["lon"],
        }

    results: dict = {
        "meta": {
            "years": list(YEARS),
            "models": {"ifs": "ecmwf_ifs (archive)", "arome": "meteofrance_arome_france_hd (historical-forecast)"},
            "local_tz": "Europe/Rome CEST UTC+2 in August",
            "delta_definition": "AROME_HD - IFS_9km (kt or deg)",
            "transects": TRANSECTS,
            "distances_nm": list(DISTANCES_NM),
        },
        "transects": {},
    }

    for t in TRANSECTS:
        tid = t["id"]
        t_out: dict = {"points": {}, "focus_14_18_local_within_10nm": {}, "hourly_by_distance": {}}
        inshore_deltas = []
        inshore_dir = []
        offshore_ref = None  # 30 nm for bend comparison

        for d in DISTANCES_NM:
            key = (tid, d)
            chunks_i = series[key]["ifs"]
            chunks_a = series[key]["arome"]
            if not chunks_i or not chunks_a:
                continue
            ifs = concat_model(chunks_i)
            arome = concat_model(chunks_a)
            all_day = paired_stats(ifs, arome, local_hours=None)
            focus = paired_stats(ifs, arome, local_hours=LOCAL_HOURS_FOCUS)
            hourly = by_local_hour(ifs, arome)
            t_out["points"][str(d)] = {
                "lat": ifs["lat"],
                "lon": ifs["lon"],
                "all_hours": all_day,
                "local_14_18": focus,
                "by_local_hour": {str(h): hourly[h] for h in range(24)},
            }
            if d <= 10.0 and focus.get("n", 0) > 0:
                inshore_deltas.append(focus["mean_delta_kt"])
                inshore_dir.append(focus)
            if d == 30.0:
                offshore_ref = focus

        # Aggregate within 10 nm, 14–18 local: pool all inshore points equally by re-fetching pooled n
        # Report mean of per-distance means (distance-weighted equal bins 0..10)
        inshore_ds = [d for d in DISTANCES_NM if d <= 10.0]
        focus_rows = [t_out["points"][str(d)]["local_14_18"] for d in inshore_ds if str(d) in t_out["points"]]
        focus_rows = [r for r in focus_rows if r.get("n", 0) > 0]
        if focus_rows:
            # sample-weighted mean delta
            n_tot = sum(r["n"] for r in focus_rows)
            mean_delta = sum(r["mean_delta_kt"] * r["n"] for r in focus_rows) / n_tot
            mean_ifs = sum(r["mean_ifs_kt"] * r["n"] for r in focus_rows) / n_tot
            mean_aro = sum(r["mean_arome_kt"] * r["n"] for r in focus_rows) / n_tot
            # direction: average of circular means is approximate; use first-moment of unit vectors
            def unit(deg: float) -> complex:
                return complex(math.cos(math.radians(deg)), math.sin(math.radians(deg)))

            u_i = sum(unit(r["mean_dir_ifs"]) * r["n"] for r in focus_rows)
            u_a = sum(unit(r["mean_dir_arome"]) * r["n"] for r in focus_rows)
            dir_i = math.degrees(math.atan2(u_i.imag, u_i.real)) % 360.0
            dir_a = math.degrees(math.atan2(u_a.imag, u_a.real)) % 360.0
            t_out["focus_14_18_local_within_10nm"] = {
                "n_hours_summed_across_points": n_tot,
                "mean_ifs_kt": round(mean_ifs, 2),
                "mean_arome_kt": round(mean_aro, 2),
                "mean_delta_kt_arome_minus_ifs": round(mean_delta, 2),
                "mean_dir_ifs_deg": round(dir_i, 1),
                "mean_dir_arome_deg": round(dir_a, 1),
                "mean_dir_delta_deg_arome_minus_ifs": round(angular_diff_deg(dir_a, dir_i), 1),
                "per_distance_nm": {
                    str(d): {
                        "delta_kt": t_out["points"][str(d)]["local_14_18"].get("mean_delta_kt"),
                        "ifs_kt": t_out["points"][str(d)]["local_14_18"].get("mean_ifs_kt"),
                        "arome_kt": t_out["points"][str(d)]["local_14_18"].get("mean_arome_kt"),
                        "dir_ifs": t_out["points"][str(d)]["local_14_18"].get("mean_dir_ifs"),
                        "dir_arome": t_out["points"][str(d)]["local_14_18"].get("mean_dir_arome"),
                        "dir_delta": t_out["points"][str(d)]["local_14_18"].get("mean_dir_delta_deg"),
                        "n": t_out["points"][str(d)]["local_14_18"].get("n"),
                    }
                    for d in inshore_ds
                    if str(d) in t_out["points"]
                },
            }
            if offshore_ref and offshore_ref.get("n", 0) > 0:
                # Coastal bend = (inshore dir - offshore dir) difference between models
                # Positive means AROME turns more than IFS from the 30 nm reference.
                inshore_dir_delta_vs_off_ifs = angular_diff_deg(dir_i, offshore_ref["mean_dir_ifs"])
                inshore_dir_delta_vs_off_aro = angular_diff_deg(dir_a, offshore_ref["mean_dir_arome"])
                t_out["focus_14_18_local_within_10nm"]["coastal_bend"] = {
                    "ref_distance_nm": 30.0,
                    "ifs_inshore_minus_30nm_deg": round(inshore_dir_delta_vs_off_ifs, 1),
                    "arome_inshore_minus_30nm_deg": round(inshore_dir_delta_vs_off_aro, 1),
                    "extra_bend_arome_minus_ifs_deg": round(
                        angular_diff_deg(inshore_dir_delta_vs_off_aro, inshore_dir_delta_vs_off_ifs), 1
                    ),
                    "dir_at_30nm_ifs": round(offshore_ref["mean_dir_ifs"], 1),
                    "dir_at_30nm_arome": round(offshore_ref["mean_dir_arome"], 1),
                }

        # Compact transect table: mean TWS by distance × local hour (14–18 detail + full)
        table = []
        for d in DISTANCES_NM:
            if str(d) not in t_out["points"]:
                continue
            row = {"distance_nm": d}
            for h in range(24):
                cell = t_out["points"][str(d)]["by_local_hour"][str(h)]
                if cell.get("n", 0) == 0:
                    continue
                row[f"h{h:02d}_ifs"] = round(cell["mean_ifs_kt"], 2)
                row[f"h{h:02d}_arome"] = round(cell["mean_arome_kt"], 2)
                row[f"h{h:02d}_delta"] = round(cell["mean_delta_kt"], 2)
            table.append(row)
        t_out["hourly_by_distance"] = table
        results["transects"][tid] = t_out

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    # Strip bulky by_local_hour from points for the saved JSON? Keep — useful. Round floats.
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {OUT_JSON}")

    write_md(results)
    print(f"Wrote {OUT_MD}")


def write_md(results: dict) -> None:
    lines: list[str] = []
    lines.append("# AROME HD vs IFS 9 km — resolution comparison")
    lines.append("")
    lines.append("August **2023–2025**, same days. Δ = AROME HD − IFS analysis (kt / deg).")
    lines.append("Local = Europe/Rome CEST (UTC+2). Not a climatology.")
    lines.append("")
    lines.append("## Headline (14–18 local, ≤10 nm offshore)")
    lines.append("")
    lines.append("| Transect | IFS kt | AROME kt | Δ kt | IFS dir° | AROME dir° | Δ dir° | Extra coastal bend° |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for tid, t in results["transects"].items():
        f = t.get("focus_14_18_local_within_10nm") or {}
        bend = (f.get("coastal_bend") or {}).get("extra_bend_arome_minus_ifs_deg", "")
        lines.append(
            f"| {tid} | {f.get('mean_ifs_kt')} | {f.get('mean_arome_kt')} | "
            f"**{f.get('mean_delta_kt_arome_minus_ifs')}** | {f.get('mean_dir_ifs_deg')} | "
            f"{f.get('mean_dir_arome_deg')} | {f.get('mean_dir_delta_deg_arome_minus_ifs')} | {bend} |"
        )
    lines.append("")

    for tid, t in results["transects"].items():
        f = t.get("focus_14_18_local_within_10nm") or {}
        lines.append(f"## {tid}")
        lines.append("")
        lines.append("### Δ kt by distance (14–18 local)")
        lines.append("")
        lines.append("| nm | IFS | AROME | Δ | n |")
        lines.append("|---:|---:|---:|---:|---:|")
        for d, row in (f.get("per_distance_nm") or {}).items():
            lines.append(
                f"| {d} | {row.get('ifs_kt') and round(row['ifs_kt'], 2)} | "
                f"{row.get('arome_kt') and round(row['arome_kt'], 2)} | "
                f"{row.get('delta_kt') and round(row['delta_kt'], 2)} | {row.get('n')} |"
            )
        lines.append("")
        bend = f.get("coastal_bend") or {}
        if bend:
            lines.append("### Direction bend (14–18 local)")
            lines.append("")
            lines.append(f"- IFS inshore−30 nm: **{bend.get('ifs_inshore_minus_30nm_deg')}°**")
            lines.append(f"- AROME inshore−30 nm: **{bend.get('arome_inshore_minus_30nm_deg')}°**")
            lines.append(f"- Extra bend (AROME−IFS): **{bend.get('extra_bend_arome_minus_ifs_deg')}°**")
            lines.append(f"- Dir at 30 nm: IFS {bend.get('dir_at_30nm_ifs')}° / AROME {bend.get('dir_at_30nm_arome')}°")
            lines.append("")

        lines.append("### Mean Δ kt by local hour × distance")
        lines.append("")
        # header for focus hours + a few others
        hours = list(range(24))
        header = "| nm | " + " | ".join(f"{h:02d}" for h in hours) + " |"
        sep = "|---:|" + "|".join(["---:"] * 24) + "|"
        lines.append(header)
        lines.append(sep)
        for row in t.get("hourly_by_distance") or []:
            cells = []
            for h in hours:
                v = row.get(f"h{h:02d}_delta")
                cells.append("" if v is None else f"{v:.1f}")
            lines.append(f"| {row['distance_nm']} | " + " | ".join(cells) + " |")
        lines.append("")

        lines.append("### Mean AROME kt by local hour × distance")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for row in t.get("hourly_by_distance") or []:
            cells = []
            for h in hours:
                v = row.get(f"h{h:02d}_arome")
                cells.append("" if v is None else f"{v:.1f}")
            lines.append(f"| {row['distance_nm']} | " + " | ".join(cells) + " |")
        lines.append("")

        lines.append("### Mean IFS kt by local hour × distance")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for row in t.get("hourly_by_distance") or []:
            cells = []
            for h in hours:
                v = row.get(f"h{h:02d}_ifs")
                cells.append("" if v is None else f"{v:.1f}")
            lines.append(f"| {row['distance_nm']} | " + " | ".join(cells) + " |")
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
