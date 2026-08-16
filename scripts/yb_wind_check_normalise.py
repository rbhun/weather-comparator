#!/usr/bin/env python3
"""Re-bin YB vs analysis residuals: per-boat norm, 45–60 ft, by region.

Reads an existing annotated samples table (CSV/parquet from
``yb_analysis_wind_check.py``), applies per-boat median normalisation,
optional LOA filter, region tags, and writes numbers-only markdown + CSVs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pmc.io.yb import DEFAULT_CACHE, DEFAULT_YEARS, load_edition_full_tracks  # noqa: E402
from pmc.stats.boat_loa import attach_loa, in_loa_range  # noqa: E402
from pmc.stats.yb_wind_check import (  # noqa: E402
    assign_region,
    assign_tod_band,
    normalise_residuals_per_boat,
    summarise_residuals,
)

OFFSHORE_ORDER = ["0-5nm", "5-10nm", "10-20nm", "20-40nm", "40nm+"]
REGIONS = ["sardinia_east", "tyrrhenian", "ligurian"]


def _load_samples(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _ensure_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "tod_band" not in out.columns:
        out["tod_band"] = out["hour_local"].map(assign_tod_band)
    if "region" not in out.columns:
        out["region"] = [
            assign_region(float(a), float(b))
            for a, b in zip(out["lat"].to_numpy(), out["lon"].to_numpy())
        ]
    if "afternoon_local" not in out.columns:
        out["afternoon_local"] = out["tod_band"] == "afternoon"
    return out


def _attach_loa_column(df: pd.DataFrame, cache_root: Path) -> pd.DataFrame:
    boats = []
    years = sorted({int(y) for y in df["year"].unique()})
    for year in years:
        boats.extend(load_edition_full_tracks(year, cache_root=cache_root))
    loa_map = attach_loa(boats)
    out = df.copy()
    keys = list(zip(out["year"].astype(int), out["boat"].astype(str)))
    out["loa_ft"] = [loa_map.get(k, {}).get("loa_ft") for k in keys]
    out["loa_source"] = [loa_map.get(k, {}).get("loa_source", "unknown") for k in keys]
    return out


def _fmt_table(frame: pd.DataFrame, cols: list[str] | None = None) -> str:
    if frame.empty:
        return "(empty)\n"
    use = frame if cols is None else frame.loc[:, [c for c in cols if c in frame.columns]]
    return use.to_markdown(index=False) + "\n"


def _band_order_key(label: str) -> int:
    try:
        return OFFSHORE_ORDER.index(str(label))
    except ValueError:
        return 99


def _ordered_offshore(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "offshore_bin" not in frame.columns:
        return frame
    out = frame.copy()
    out["_ord"] = out["offshore_bin"].map(_band_order_key)
    return out.sort_values(["_ord"]).drop(columns="_ord").reset_index(drop=True)


def _numbers_report(
    *,
    raw_aft: pd.DataFrame,
    norm_aft: pd.DataFrame,
    loa_aft: pd.DataFrame,
    norm_region_tod: pd.DataFrame,
    loa_region_tod: pd.DataFrame,
    meta: dict,
) -> str:
    lines: list[str] = []
    lines.append("# YB vs analysis — normalised & regional")
    lines.append("")
    lines.append("## Meta")
    lines.append("")
    for k in (
        "generated_utc",
        "n_samples",
        "n_boats",
        "n_boats_loa_45_60",
        "n_samples_loa_45_60",
        "loa_resolved_boats",
        "polar",
    ):
        if k in meta:
            lines.append(f"- {k}: {meta[k]}")
    lines.append("")

    lines.append("## 1. Per-boat normalised — afternoon offshore bands (all regions)")
    lines.append("")
    lines.append("residual = residual_kt − median_residual(boat, edition)")
    lines.append("")
    lines.append(_fmt_table(_ordered_offshore(norm_aft)))

    lines.append("## 2. Unnormalised 45–60 ft LOA — afternoon offshore bands")
    lines.append("")
    lines.append(_fmt_table(_ordered_offshore(loa_aft)))

    lines.append("## 3. Band ordering check (afternoon, median residual)")
    lines.append("")
    def _order(frame: pd.DataFrame, top_n: int | None = None) -> list[str]:
        if frame.empty:
            return []
        ranked = frame.sort_values("residual_median_kt", ascending=False)
        labels = [str(x) for x in ranked["offshore_bin"].tolist()]
        return labels if top_n is None else labels[:top_n]

    lines.append(f"- normalised_all: {' > '.join(_order(norm_aft))}")
    lines.append(f"- loa_45_60_raw: {' > '.join(_order(loa_aft))}")
    lines.append(f"- raw_all: {' > '.join(_order(raw_aft))}")
    lines.append(
        f"- top3_normalised: {' > '.join(_order(norm_aft, 3))}"
    )
    lines.append(f"- top3_loa_45_60: {' > '.join(_order(loa_aft, 3))}")
    lines.append(
        f"- top3_same: {_order(norm_aft, 3) == _order(loa_aft, 3)}"
    )
    lines.append(
        f"- full_order_same: {_order(norm_aft) == _order(loa_aft)}"
    )
    lines.append("")

    lines.append("## 4. Per-boat normalised — by region × tod × offshore")
    lines.append("")
    for region in REGIONS:
        lines.append(f"### {region}")
        lines.append("")
        for tod in ("afternoon", "night"):
            sub = norm_region_tod[
                (norm_region_tod["region"] == region) & (norm_region_tod["tod_band"] == tod)
            ]
            lines.append(f"#### {tod}")
            lines.append("")
            lines.append(_fmt_table(_ordered_offshore(sub.drop(columns=["region", "tod_band"], errors="ignore"))))

    lines.append("## 5. 45–60 ft LOA unnormalised — by region × tod × offshore")
    lines.append("")
    for region in REGIONS:
        lines.append(f"### {region}")
        lines.append("")
        for tod in ("afternoon", "night"):
            sub = loa_region_tod[
                (loa_region_tod["region"] == region) & (loa_region_tod["tod_band"] == tod)
            ]
            lines.append(f"#### {tod}")
            lines.append("")
            lines.append(_fmt_table(_ordered_offshore(sub.drop(columns=["region", "tod_band"], errors="ignore"))))

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="data/yb_wind_check/samples.csv")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE))
    parser.add_argument("--out-dir", default="data/yb_wind_check")
    parser.add_argument("--docs", default="docs/yb-analysis-wind-check.md")
    parser.add_argument("--loa-min", type=float, default=45.0)
    parser.add_argument("--loa-max", type=float, default=60.0)
    args = parser.parse_args()

    samples = _ensure_derived(_load_samples(Path(args.samples)))
    samples = _attach_loa_column(samples, Path(args.cache_root))
    samples = normalise_residuals_per_boat(samples)

    loa_mask = samples["loa_ft"].map(
        lambda v: in_loa_range(None if pd.isna(v) else float(v), args.loa_min, args.loa_max)
    )
    loa_samples = samples.loc[loa_mask].copy()

    raw_sum = summarise_residuals(samples, residual_col="residual_kt")
    norm_sum = summarise_residuals(samples, residual_col="residual_norm_kt")
    loa_sum = summarise_residuals(loa_samples, residual_col="residual_kt")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = ROOT / "docs" / "yb_wind_check"
    docs_dir.mkdir(parents=True, exist_ok=True)

    for prefix, summary in (
        ("raw", raw_sum),
        ("norm", norm_sum),
        ("loa45_60", loa_sum),
    ):
        for name, frame in summary.items():
            frame.to_csv(out_dir / f"{prefix}_{name}.csv", index=False)
            frame.to_csv(docs_dir / f"{prefix}_{name}.csv", index=False)

    n_boats = int(samples.groupby(["year", "boat"]).ngroups)
    n_boats_loa = int(loa_samples.groupby(["year", "boat"]).ngroups)
    resolved = int(
        samples.drop_duplicates(["year", "boat"])
        .loc[lambda d: d["loa_source"] != "unknown"]
        .shape[0]
    )
    meta = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_samples": int(len(samples)),
        "n_boats": n_boats,
        "n_boats_loa_45_60": n_boats_loa,
        "n_samples_loa_45_60": int(len(loa_samples)),
        "loa_resolved_boats": resolved,
        "loa_range_ft": [args.loa_min, args.loa_max],
        "polar": str(samples["polar_name"].iloc[0]) if len(samples) else None,
        "years": sorted({int(y) for y in samples["year"].unique()}),
    }
    (out_dir / "meta_normalised.json").write_text(json.dumps(meta, indent=2) + "\n")
    (docs_dir / "meta_normalised.json").write_text(json.dumps(meta, indent=2) + "\n")

    report = _numbers_report(
        raw_aft=raw_sum["by_offshore_afternoon"],
        norm_aft=norm_sum["by_offshore_afternoon"],
        loa_aft=loa_sum["by_offshore_afternoon"],
        norm_region_tod=norm_sum["by_region_tod_offshore"],
        loa_region_tod=loa_sum["by_region_tod_offshore"],
        meta=meta,
    )
    Path(args.docs).write_text(report, encoding="utf-8")
    (out_dir / "report_normalised.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
