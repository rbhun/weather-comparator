#!/usr/bin/env python3
"""AROME HD-only Sardinia East cross-shore transect (Aug 2023–2025).

Reuses resolution_cmp cache from compare_arome_ifs_resolution.py.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from compare_arome_ifs_resolution import (
    DISTANCES_NM,
    HIST_FC,
    MS_TO_KT,
    UTC_OFFSET_AUG,
    YEARS,
    Point,
    build_points,
    circular_mean_deg,
    fetch_month,
)

OUT_JSON = Path("data/resolution/arome_sardinia_transect.json")
OUT_MD = Path("docs/arome-sardinia-transect.md")


def main() -> None:
    points = [p for p in build_points() if p.transect_id == "sardinia_e"]
    assert len(points) == len(DISTANCES_NM)

    # year -> dist -> arrays
    by_year: dict[int, dict[float, dict]] = {}
    for year in YEARS:
        month = fetch_month("meteofrance_arome_france_hd", HIST_FC, year, points)
        by_year[year] = {}
        for p in points:
            key = (p.transect_id, p.distance_nm)
            by_year[year][p.distance_nm] = month[key]

    def local_hours(times: np.ndarray) -> np.ndarray:
        return (np.array([int(t[11:13]) for t in times]) + UTC_OFFSET_AUG) % 24

    # Aggregate: all years + per year
    def stats_for(chunks: list[dict]) -> dict:
        """Return mean_tws / mean_dir / n by local hour for concatenated chunks."""
        times = np.concatenate([c["time"] for c in chunks])
        spd = np.concatenate([c["speed_kt"] for c in chunks])
        direc = np.concatenate([c["direction_deg"] for c in chunks])
        lh = local_hours(times)
        out = {}
        for h in range(24):
            m = (lh == h) & np.isfinite(spd) & np.isfinite(direc)
            n = int(m.sum())
            if n == 0:
                out[h] = {"n": 0, "mean_tws_kt": None, "mean_dir_deg": None}
            else:
                out[h] = {
                    "n": n,
                    "mean_tws_kt": round(float(np.mean(spd[m])), 2),
                    "mean_dir_deg": round(circular_mean_deg(direc[m]), 1),
                }
        return out

    result = {
        "meta": {
            "model": "meteofrance_arome_france_hd",
            "transect": "sardinia_e",
            "coast": {"lat": 40.5, "lon": 9.72, "bearing_deg": 90.0},
            "years": list(YEARS),
            "local_tz": "Europe/Rome CEST UTC+2",
            "distances_nm": list(DISTANCES_NM),
        },
        "all_years": {},
        "by_year": {},
    }

    for d in DISTANCES_NM:
        chunks = [by_year[y][d] for y in YEARS]
        # convert speed to kt already in cache structure - check units
        # fetch_month stores speed_kt already
        result["all_years"][str(d)] = {
            "lat": chunks[0]["lat"],
            "lon": chunks[0]["lon"],
            "by_local_hour": {str(h): v for h, v in stats_for(chunks).items()},
        }

    for y in YEARS:
        result["by_year"][str(y)] = {}
        for d in DISTANCES_NM:
            chunk = by_year[y][d]
            result["by_year"][str(y)][str(d)] = {
                "by_local_hour": {str(h): v for h, v in stats_for([chunk]).items()},
            }

    # Focus: afternoon inshore weakening consistency
    focus_hours = (14, 15, 16, 17, 18)
    consistency = {}
    for y in YEARS:
        row = {}
        for d in DISTANCES_NM:
            cells = [
                result["by_year"][str(y)][str(d)]["by_local_hour"][str(h)]
                for h in focus_hours
            ]
            ns = [c["n"] for c in cells]
            tws = [c["mean_tws_kt"] for c in cells if c["mean_tws_kt"] is not None]
            dirs = [c["mean_dir_deg"] for c in cells if c["mean_dir_deg"] is not None]
            n = sum(ns)
            # sample-weighted mean across afternoon hours
            tws_w = sum(
                result["by_year"][str(y)][str(d)]["by_local_hour"][str(h)]["mean_tws_kt"]
                * result["by_year"][str(y)][str(d)]["by_local_hour"][str(h)]["n"]
                for h in focus_hours
            ) / n
            row[str(d)] = {
                "n": n,
                "mean_tws_kt_14_18": round(tws_w, 2),
            }
        # deltas vs 10 nm and vs 30 nm
        t10 = row["10.0"]["mean_tws_kt_14_18"]
        consistency[str(y)] = {
            "tws_by_nm": {k: v["mean_tws_kt_14_18"] for k, v in row.items()},
            "n_by_nm": {k: v["n"] for k, v in row.items()},
            "delta_0_minus_10": round(row["0.0"]["mean_tws_kt_14_18"] - t10, 2),
            "delta_2_5_minus_10": round(row["2.5"]["mean_tws_kt_14_18"] - t10, 2),
            "delta_5_minus_10": round(row["5.0"]["mean_tws_kt_14_18"] - t10, 2),
            "delta_0_minus_7_5": round(
                row["0.0"]["mean_tws_kt_14_18"] - row["7.5"]["mean_tws_kt_14_18"], 2
            ),
        }
    result["afternoon_14_18_per_year"] = consistency

    # all-years afternoon
    all_row = {}
    for d in DISTANCES_NM:
        cells = [
            result["all_years"][str(d)]["by_local_hour"][str(h)] for h in focus_hours
        ]
        n = sum(c["n"] for c in cells)
        tws_w = sum(c["mean_tws_kt"] * c["n"] for c in cells) / n
        all_row[str(d)] = round(tws_w, 2)
    result["afternoon_14_18_all_years"] = all_row

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))
    write_md(result)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


def write_md(r: dict) -> None:
    lines: list[str] = []
    lines.append("# AROME HD — Sardinia East transect (Aug 2023–2025)\n")
    lines.append("Model: `meteofrance_arome_france_hd`. Coast 40.5N 9.72E, offshore 090°.")
    lines.append("Local = CEST (UTC+2). Not a climatology backbone.\n")

    lines.append("## Afternoon 14–18 local — mean TWS (kt) by distance, per year\n")
    lines.append("| year | 0 | 2.5 | 5 | 7.5 | 10 | 15 | 20 | 30 | 0−10 | 5−10 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for y in YEARS:
        tws = r["afternoon_14_18_per_year"][str(y)]["tws_by_nm"]
        d0 = tws["0.0"] - tws["10.0"]
        d5 = tws["5.0"] - tws["10.0"]
        lines.append(
            f"| {y} | {tws['0.0']:.2f} | {tws['2.5']:.2f} | {tws['5.0']:.2f} | "
            f"{tws['7.5']:.2f} | {tws['10.0']:.2f} | {tws['15.0']:.2f} | "
            f"{tws['20.0']:.2f} | {tws['30.0']:.2f} | {d0:+.2f} | {d5:+.2f} |"
        )
    tws = r["afternoon_14_18_all_years"]
    d0 = tws["0.0"] - tws["10.0"]
    d5 = tws["5.0"] - tws["10.0"]
    lines.append(
        f"| all | {tws['0.0']:.2f} | {tws['2.5']:.2f} | {tws['5.0']:.2f} | "
        f"{tws['7.5']:.2f} | {tws['10.0']:.2f} | {tws['15.0']:.2f} | "
        f"{tws['20.0']:.2f} | {tws['30.0']:.2f} | {d0:+.2f} | {d5:+.2f} |"
    )
    lines.append("")
    lines.append("n per year per distance for 14–18 local (5 hours × 31 days = 155):\n")
    lines.append("| year | n@0 | n@5 | n@10 |")
    lines.append("|---:|---:|---:|---:|")
    for y in YEARS:
        nn = r["afternoon_14_18_per_year"][str(y)]["n_by_nm"]
        lines.append(f"| {y} | {nn['0.0']} | {nn['5.0']} | {nn['10.0']} |")
    lines.append("")

    # Full all-years TWS table
    lines.append("## All years — mean TWS (kt) by local hour × distance\n")
    hours = list(range(24))
    lines.append("| nm | " + " | ".join(f"{h:02d}" for h in hours) + " |")
    lines.append("|---:|" + "|".join(["---:"] * 24) + "|")
    for d in DISTANCES_NM:
        cells = []
        for h in hours:
            v = r["all_years"][str(d)]["by_local_hour"][str(h)]["mean_tws_kt"]
            cells.append("" if v is None else f"{v:.1f}")
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## All years — n by local hour × distance\n")
    lines.append("| nm | " + " | ".join(f"{h:02d}" for h in hours) + " |")
    lines.append("|---:|" + "|".join(["---:"] * 24) + "|")
    for d in DISTANCES_NM:
        cells = [
            str(r["all_years"][str(d)]["by_local_hour"][str(h)]["n"]) for h in hours
        ]
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## All years — mean direction (°) by local hour × distance\n")
    lines.append("| nm | " + " | ".join(f"{h:02d}" for h in hours) + " |")
    lines.append("|---:|" + "|".join(["---:"] * 24) + "|")
    for d in DISTANCES_NM:
        cells = []
        for h in hours:
            v = r["all_years"][str(d)]["by_local_hour"][str(h)]["mean_dir_deg"]
            cells.append("" if v is None else f"{v:.0f}")
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    lines.append("")

    # Per-year afternoon hour detail for inshore
    lines.append("## Per-year afternoon hours — TWS at 0 / 5 / 10 nm\n")
    lines.append("| year | hour | 0 nm | 5 nm | 10 nm | 0−10 | 5−10 | n |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for y in YEARS:
        for h in focus_hours:
            a = r["by_year"][str(y)]["0.0"]["by_local_hour"][str(h)]
            b = r["by_year"][str(y)]["5.0"]["by_local_hour"][str(h)]
            c = r["by_year"][str(y)]["10.0"]["by_local_hour"][str(h)]
            lines.append(
                f"| {y} | {h} | {a['mean_tws_kt']:.2f} | {b['mean_tws_kt']:.2f} | "
                f"{c['mean_tws_kt']:.2f} | {a['mean_tws_kt']-c['mean_tws_kt']:+.2f} | "
                f"{b['mean_tws_kt']-c['mean_tws_kt']:+.2f} | {a['n']} |"
            )
    lines.append("")

    lines.append("## Per-year afternoon hours — direction at 0 / 5 / 10 nm\n")
    lines.append("| year | hour | 0 nm | 5 nm | 10 nm | n |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for y in YEARS:
        for h in focus_hours:
            a = r["by_year"][str(y)]["0.0"]["by_local_hour"][str(h)]
            b = r["by_year"][str(y)]["5.0"]["by_local_hour"][str(h)]
            c = r["by_year"][str(y)]["10.0"]["by_local_hour"][str(h)]
            lines.append(
                f"| {y} | {h} | {a['mean_dir_deg']:.0f} | {b['mean_dir_deg']:.0f} | "
                f"{c['mean_dir_deg']:.0f} | {a['n']} |"
            )
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")


# fix focus_hours reference in write_md
focus_hours = (14, 15, 16, 17, 18)

if __name__ == "__main__":
    main()
