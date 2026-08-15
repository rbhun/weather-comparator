#!/usr/bin/env python3
"""Download YB Palermo–Montecarlo tracks and write the dashboard overlay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pmc.io.yb import DEFAULT_CACHE, DEFAULT_YEARS, build_overlay, fetch_year, write_overlay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--years",
        default=",".join(str(y) for y in DEFAULT_YEARS),
        help="Comma-separated years",
    )
    parser.add_argument("--json-out", default="dashboard/yb_results.json")
    parser.add_argument("--js-out", default="dashboard/yb_results.js")
    args = parser.parse_args()

    years = tuple(int(tok.strip()) for tok in args.years.split(",") if tok.strip())
    cache_root = Path(args.cache_root)
    for year in years:
        print(f"[yb] fetch pm{year}", flush=True)
        fetch_year(year, cache_root=cache_root, refresh=args.refresh)
    overlay = build_overlay(years, cache_root=cache_root)
    write_overlay(overlay, Path(args.json_out), Path(args.js_out))
    boats = sum(len(ed["boats"]) for ed in overlay["editions"])
    tracked = sum(1 for ed in overlay["editions"] for b in ed["boats"] if b["lon"])
    print(
        f"[yb] wrote {args.json_out} and {args.js_out}: "
        f"{len(overlay['editions'])} editions, {tracked}/{boats} boats with tracks",
        flush=True,
    )
    for ed in overlay["editions"]:
        classes = ", ".join(c["name"] for c in ed["classes"])
        print(
            f"  {ed['year']}: {len(ed['boats'])} boats, "
            f"start={ed['start_utc']} classes=[{classes}]",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
