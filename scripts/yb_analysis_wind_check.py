#!/usr/bin/env python3
"""YB historical tracks vs IFS analysis — independent coastal-wind check.

For each past Palermo–Montecarlo edition with YB tracks:
  1. Sample boat positions at ~20 min (15–30 min) intervals; compute SOG/COG.
  2. Sample analysis u/v at each (time, lat, lon); polar-predict speed on COG.
  3. Residual = SOG − predicted. Bin by offshore distance, local hour, TWS.

Fetch YB + archive analysis for real editions::

    python3 scripts/yb_analysis_wind_check.py --fetch-yb --fetch-analysis \\
        --polar contracts/fixtures/polar_52ft.pol --out-dir data/yb_wind_check

Or point at an existing C1 zarr::

    python3 scripts/yb_analysis_wind_check.py --fetch-yb \\
        --wind data/wind/ifs_analysis_9km.zarr --polar config/polar/boat.pol
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pmc.io.yb import DEFAULT_CACHE, DEFAULT_YEARS, fetch_year  # noqa: E402
from pmc.polar import load_polar  # noqa: E402
from pmc.stats.archive_wind import (  # noqa: E402
    DEFAULT_CACHE as ARCHIVE_CACHE,
    build_wind_dataset_for_samples,
)
from pmc.stats.yb_wind_check import (  # noqa: E402
    WindCheckConfig,
    annotate_samples_with_wind,
    collect_motion_samples,
    format_summary_markdown,
    load_years_full_tracks,
    summarise_residuals,
)


def _open_wind(path: Path) -> xr.Dataset:
    ds = xr.open_zarr(path, consolidated=False)
    needed = {"time", "lat", "lon"}
    if not needed.issubset(set(ds.dims)):
        raise SystemExit(f"Wind store {path} missing dims {sorted(needed - set(ds.dims))}")
    for var in ("u10", "v10"):
        if var not in ds:
            raise SystemExit(f"Wind store {path} missing {var}")
    return ds


def _resolve_polar(path: Path | None) -> Path:
    candidates = []
    if path is not None:
        candidates.append(path)
    candidates.extend(
        [
            ROOT / "config" / "polar" / "boat.pol",
            ROOT / "contracts" / "fixtures" / "polar_52ft.pol",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit("No polar file found (boat.pol or fixtures/polar_52ft.pol)")


def _write_tables(summaries: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in summaries.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        default=",".join(str(y) for y in DEFAULT_YEARS),
        help="Comma-separated edition years",
    )
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE))
    parser.add_argument("--fetch-yb", action="store_true", help="Download YB KML/CSV if missing")
    parser.add_argument(
        "--fetch-analysis",
        action="store_true",
        help="Fetch IFS analysis at sample cells via Open-Meteo archive (cached)",
    )
    parser.add_argument(
        "--wind",
        default=None,
        help="Path to C1 analysis zarr (optional if --fetch-analysis)",
    )
    parser.add_argument("--polar", default=None, help="Polar .pol path (default boat.pol then fixture)")
    parser.add_argument("--interval-min", type=int, default=20)
    parser.add_argument("--timezone", default="Europe/Rome")
    parser.add_argument("--out-dir", default="data/yb_wind_check")
    parser.add_argument("--archive-cache", default=str(ARCHIVE_CACHE))
    parser.add_argument("--model", default="ecmwf_ifs")
    parser.add_argument(
        "--max-boats-per-year",
        type=int,
        default=0,
        help="Optional cap for smoke runs (0 = all boats)",
    )
    args = parser.parse_args()

    if not args.wind and not args.fetch_analysis:
        raise SystemExit("Provide --wind PATH and/or --fetch-analysis")

    years = tuple(int(tok.strip()) for tok in args.years.split(",") if tok.strip())
    cache_root = Path(args.cache_root)
    if args.fetch_yb:
        for year in years:
            print(f"[yb-wind-check] fetch pm{year}", flush=True)
            fetch_year(year, cache_root=cache_root, refresh=False)

    polar_path = _resolve_polar(Path(args.polar) if args.polar else None)
    polar = load_polar(polar_path)
    print(f"[yb-wind-check] polar={polar.name} ({polar_path})", flush=True)

    boats = load_years_full_tracks(years, cache_root=cache_root)
    if args.max_boats_per_year > 0:
        capped = []
        for year in years:
            year_boats = [b for b in boats if b.year == year]
            capped.extend(year_boats[: args.max_boats_per_year])
        boats = capped
    timed = sum(1 for b in boats if b.times)
    print(f"[yb-wind-check] boats={len(boats)} with_times={timed}", flush=True)
    if timed == 0:
        raise SystemExit("No timed tracks — run with --fetch-yb against a writable cache")

    cfg = WindCheckConfig(interval_min=args.interval_min, display_timezone=args.timezone)
    motion = collect_motion_samples(boats, cfg=cfg)
    print(f"[yb-wind-check] motion_samples={len(motion)}", flush=True)
    if motion.empty:
        raise SystemExit("No motion samples in the 15–30 min window")

    if args.fetch_analysis:
        print("[yb-wind-check] fetching archive analysis for sample cells…", flush=True)
        wind = build_wind_dataset_for_samples(
            motion,
            cache_root=Path(args.archive_cache),
            model=args.model,
        )
        wind_label = f"archive:{args.model}"
        zarr_out = Path(args.out_dir) / "analysis_samples.zarr"
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        if zarr_out.exists():
            import shutil

            shutil.rmtree(zarr_out)
        wind.to_zarr(zarr_out, mode="w")
        print(f"[yb-wind-check] wrote {zarr_out}", flush=True)
    else:
        wind = _open_wind(Path(args.wind))
        wind_label = str(args.wind)
        print(
            f"[yb-wind-check] wind={args.wind} times={wind.sizes.get('time')} "
            f"source={wind.attrs.get('source', '?')}",
            flush=True,
        )

    samples = annotate_samples_with_wind(motion, wind, polar, cfg=cfg)
    summaries = summarise_residuals(samples)
    valid_n = int(samples["residual_kt"].notna().sum()) if not samples.empty else 0
    print(f"[yb-wind-check] samples={len(samples)} with_residual={valid_n}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        samples.to_parquet(out_dir / "samples.parquet", index=False)
    except Exception:
        samples.to_csv(out_dir / "samples.csv", index=False)
    _write_tables(summaries, out_dir)

    meta = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "years": list(years),
        "n_boats": len(boats),
        "n_samples": len(samples),
        "n_residuals": valid_n,
        "polar": polar.name,
        "polar_path": str(polar_path),
        "polar_is_validated": polar_path.name == "boat.pol",
        "wind": wind_label,
        "wind_source": wind.attrs.get("source", "unknown"),
        "interval_min": args.interval_min,
        "display_timezone": args.timezone,
        "warnings": [],
    }
    if not meta["polar_is_validated"]:
        meta["warnings"].append(
            "Using fabricated/generic polar — residuals inherit polar bias; "
            "do not treat magnitudes as calibrated without boat.pol."
        )

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    report = format_summary_markdown(summaries, meta=meta)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    docs_path = ROOT / "docs" / "yb-analysis-wind-check.md"
    docs_path.write_text(report, encoding="utf-8")
    print(f"[yb-wind-check] wrote {out_dir} and {docs_path}", flush=True)

    coast = summaries["by_coast_afternoon"]
    if not coast.empty:
        print(coast.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
