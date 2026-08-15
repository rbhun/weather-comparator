#!/usr/bin/env python3
"""Sardinia-east cuts: LOA 45-60 raw, by year, leader-lag confound check."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pmc.io.yb import DEFAULT_CACHE, load_edition_full_tracks  # noqa: E402
from pmc.stats.boat_loa import attach_loa as resolve_boat_loa  # noqa: E402
from pmc.stats.boat_loa import in_loa_range  # noqa: E402
from pmc.stats.yb_wind_check import assign_region, assign_tod_band  # noqa: E402

FINISH = (43.73, 7.42)
OFFSHORE_ORDER = ["0-5nm", "5-10nm", "10-20nm", "20-40nm", "40nm+"]
EARTH_NM = 3440.065


def haversine_nm(lat1, lon1, lat2, lon2) -> float:
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return EARTH_NM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_samples(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    if "tod_band" not in df.columns:
        df["tod_band"] = df["hour_local"].map(assign_tod_band)
    if "region" not in df.columns:
        df["region"] = [
            assign_region(float(a), float(b))
            for a, b in zip(df["lat"].to_numpy(), df["lon"].to_numpy())
        ]
    return df


def attach_loa_column(df: pd.DataFrame, cache_root: Path) -> pd.DataFrame:
    boats = []
    for year in sorted({int(y) for y in df["year"].unique()}):
        boats.extend(load_edition_full_tracks(year, cache_root=cache_root))
    loa_map = resolve_boat_loa(boats)
    out = df.copy()
    keys = list(zip(out["year"].astype(int), out["boat"].astype(str)))
    out["loa_ft"] = [loa_map.get(k, {}).get("loa_ft") for k in keys]
    return out


def band_table(df: pd.DataFrame, residual_col: str = "residual_kt") -> pd.DataFrame:
    rows = []
    for band in OFFSHORE_ORDER:
        sub = df[df["offshore_bin"] == band]
        if sub.empty:
            continue
        rows.append(
            {
                "offshore_bin": band,
                "n": int(len(sub)),
                "residual_median_kt": round(float(sub[residual_col].median()), 2),
                "residual_p10_kt": round(float(sub[residual_col].quantile(0.1)), 2),
                "residual_p90_kt": round(float(sub[residual_col].quantile(0.9)), 2),
                "sog_median_kt": round(float(sub["sog_kt"].median()), 2),
                "predicted_median_kt": round(float(sub["predicted_sog_kt"].median()), 2),
                "analysis_tws_median_kt": round(float(sub["analysis_tws_kt"].median()), 2),
            }
        )
    return pd.DataFrame(rows)


def mark_behind_leaders(df: pd.DataFrame, top_n: int = 3, window_min: float = 30.0) -> pd.DataFrame:
    """Flag samples behind concurrent race leaders by distance-to-finish.

    For each (year, time_utc rounded to interval), rank boats by DTF ascending.
    Leaders = ranks 1..top_n. Sample is behind if rank > top_n.
    """
    out = df.copy()
    out["dtf_nm"] = [
        haversine_nm(float(a), float(b), FINISH[0], FINISH[1])
        for a, b in zip(out["lat"], out["lon"])
    ]
    out["time_utc"] = pd.to_datetime(out["time_utc"], utc=True)
    # bucket to 20 min to match sampling
    out["t_bucket"] = out["time_utc"].dt.floor(f"{int(window_min)}min")
    behind = []
    rank_list = []
    n_fleet = []
    for (year, tbucket), group in out.groupby(["year", "t_bucket"], sort=False):
        # one row per boat (latest in bucket)
        g = group.sort_values("time_utc").groupby("boat", as_index=False).tail(1)
        g = g.sort_values("dtf_nm")
        ranks = {b: i + 1 for i, b in enumerate(g["boat"].tolist())}
        fleet_n = len(ranks)
        for boat in group["boat"]:
            r = ranks.get(boat)
            rank_list.append(r)
            n_fleet.append(fleet_n)
            behind.append(bool(r is not None and r > top_n))
    # groupby iteration order may not match out index — redo with map
    key_rank = {}
    key_n = {}
    for (year, tbucket), group in out.groupby(["year", "t_bucket"], sort=False):
        g = group.sort_values("time_utc").groupby("boat", as_index=False).tail(1)
        g = g.sort_values("dtf_nm")
        ranks = {b: i + 1 for i, b in enumerate(g["boat"].tolist())}
        for b, r in ranks.items():
            key_rank[(int(year), tbucket, str(b))] = r
            key_n[(int(year), tbucket, str(b))] = len(ranks)

    out["fleet_rank_dtf"] = [
        key_rank.get((int(y), t, str(b)))
        for y, t, b in zip(out["year"], out["t_bucket"], out["boat"])
    ]
    out["fleet_n"] = [
        key_n.get((int(y), t, str(b)))
        for y, t, b in zip(out["year"], out["t_bucket"], out["boat"])
    ]
    out["behind_leaders"] = out["fleet_rank_dtf"].map(
        lambda r: bool(r is not None and r > top_n)
    )
    return out


def main() -> None:
    samples_path = Path("data/yb_wind_check/samples.parquet")
    if not samples_path.exists():
        samples_path = Path("data/yb_wind_check/samples.csv")
    if not samples_path.exists():
        raise SystemExit(f"missing samples under data/yb_wind_check/")

    cache_root = Path(DEFAULT_CACHE)
    df = load_samples(samples_path)
    df = attach_loa_column(df, cache_root)
    df["loa_45_60"] = df["loa_ft"].map(lambda v: in_loa_range(v, 45.0, 60.0) if pd.notna(v) else False)

    sard = df[df["region"] == "sardinia_east"].copy()
    loa = sard[sard["loa_45_60"]].copy()

    report: dict = {"meta": {"n_sardinia_east": int(len(sard)), "n_loa_45_60": int(len(loa))}}

    # 1. LOA raw afternoon / night
    for tod in ("afternoon", "night"):
        sub = loa[loa["tod_band"] == tod]
        report[f"loa45_60_{tod}"] = band_table(sub).to_dict(orient="records")

    # 2. By year — all boats raw + LOA raw, afternoon
    by_year = []
    for year in sorted(sard["year"].unique()):
        for tod in ("afternoon", "night"):
            for label, frame in (
                ("all_raw", sard),
                ("loa45_60_raw", loa),
            ):
                sub = frame[(frame["year"] == year) & (frame["tod_band"] == tod)]
                tab = band_table(sub)
                for _, row in tab.iterrows():
                    by_year.append(
                        {
                            "year": int(year),
                            "tod_band": tod,
                            "subset": label,
                            **row.to_dict(),
                        }
                    )
    report["by_year"] = by_year

    # 3. Behind leaders fraction for 0-5 nm
    sard = mark_behind_leaders(sard, top_n=3)
    band05 = sard[sard["offshore_bin"] == "0-5nm"]
    n05 = len(band05)
    n_behind = int(band05["behind_leaders"].sum())
    report["behind_leaders_0_5nm"] = {
        "definition": (
            "Concurrent DTF rank vs finish (43.73N 7.42E); "
            "leaders = ranks 1–3 in same year & 20-min time bucket; "
            "behind = rank > 3"
        ),
        "n_0_5nm": n05,
        "n_behind_leaders": n_behind,
        "fraction_behind": round(n_behind / n05, 3) if n05 else None,
        "by_tod": {},
        "by_year": {},
    }
    for tod in ("afternoon", "night"):
        sub = band05[band05["tod_band"] == tod]
        nb = int(sub["behind_leaders"].sum())
        report["behind_leaders_0_5nm"]["by_tod"][tod] = {
            "n": int(len(sub)),
            "n_behind": nb,
            "fraction_behind": round(nb / len(sub), 3) if len(sub) else None,
            "residual_median_behind": round(float(sub.loc[sub["behind_leaders"], "residual_kt"].median()), 2)
            if nb
            else None,
            "residual_median_leaders": round(
                float(sub.loc[~sub["behind_leaders"], "residual_kt"].median()), 2
            )
            if len(sub) > nb
            else None,
        }
    for year in sorted(band05["year"].unique()):
        sub = band05[band05["year"] == year]
        nb = int(sub["behind_leaders"].sum())
        report["behind_leaders_0_5nm"]["by_year"][str(int(year))] = {
            "n": int(len(sub)),
            "n_behind": nb,
            "fraction_behind": round(nb / len(sub), 3) if len(sub) else None,
        }

    # Also top-25% leaders definition
    sard2 = sard.copy()
    sard2["behind_top25"] = [
        bool(r is not None and n is not None and r > max(1, math.ceil(0.25 * n)))
        for r, n in zip(sard2["fleet_rank_dtf"], sard2["fleet_n"])
    ]
    b05 = sard2[sard2["offshore_bin"] == "0-5nm"]
    nb = int(b05["behind_top25"].sum())
    report["behind_top25_0_5nm"] = {
        "definition": "behind = DTF rank worse than top 25% of fleet in same 20-min bucket",
        "n_0_5nm": int(len(b05)),
        "n_behind": nb,
        "fraction_behind": round(nb / len(b05), 3) if len(b05) else None,
    }

    out = Path("docs/yb_wind_check/sardinia_east_cuts.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
