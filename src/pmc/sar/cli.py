"""CLI for the Sentinel-1 SAR lee-shadow falsification test."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pmc.sar.analyse import analyse_shadow_test  # noqa: E402
from pmc.sar.fetch import fetch_sar_scenes, load_sar_config, open_sar_store  # noqa: E402


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sentinel-1 SAR paired within-scene test of the Sardinian east-coast "
            "wind-shadow hypothesis (speed only; no direction)."
        )
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch or resume C9 SAR scenes (cached).")
    p_fetch.add_argument("--start", type=_parse_date, default=date(2018, 8, 1))
    p_fetch.add_argument("--end", type=_parse_date, default=date(2025, 8, 31))
    p_fetch.add_argument(
        "--fixture",
        action="store_true",
        help="Force the committed synthetic fixture (no network).",
    )
    p_fetch.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "sar.yaml",
    )

    p_run = sub.add_parser("analyse", help="Run the paired lee-shadow falsification test.")
    p_run.add_argument(
        "--sar",
        type=Path,
        default=None,
        help="Path to C9 zarr/NetCDF. Default: fetch --fixture.",
    )
    p_run.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "sar.yaml",
    )
    p_run.add_argument(
        "--arome",
        type=Path,
        default=None,
        help="Optional AROME speed field NetCDF/zarr with u10/v10 or wind_speed_kt.",
    )
    p_run.add_argument(
        "--era5",
        type=Path,
        default=None,
        help="Optional ERA5 speed field NetCDF/zarr with u10/v10 or wind_speed_kt.",
    )
    p_run.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "sar" / "shadow_test.json",
    )
    p_run.add_argument(
        "--fixture",
        action="store_true",
        help="Use the committed synthetic SAR fixture.",
    )

    args = parser.parse_args(argv)
    cfg = load_sar_config(args.config)

    if args.cmd == "fetch":
        path = fetch_sar_scenes(
            start=args.start,
            end=args.end,
            cfg=cfg,
            use_fixture=bool(args.fixture),
        )
        print(f"SAR store: {path}")
        return 0

    if args.cmd == "analyse":
        if args.sar is not None:
            sar_path = args.sar
        else:
            sar_path = fetch_sar_scenes(
                start=date(2018, 8, 1),
                end=date(2025, 8, 31),
                cfg=cfg,
                use_fixture=True if args.fixture or args.sar is None else False,
            )
        sar = open_sar_store(sar_path)
        arome = _optional_speed_kt(args.arome) if args.arome else None
        era5 = _optional_speed_kt(args.era5) if args.era5 else None
        result = analyse_shadow_test(
            sar, cfg, arome_speed_kt=arome, era5_speed_kt=era5
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        print(f"Wrote {args.output}")
        print(
            f"verdict={result['verdict']} n={result['n_scenes_retained']} "
            f"pipeline_valid={result['pipeline_valid']}"
        )
        return 0

    parser.error(f"Unknown command {args.cmd}")
    return 2


def _optional_speed_kt(path: Path) -> xr.DataArray:
    path = Path(path)
    ds = xr.open_zarr(path, consolidated=True) if path.suffix != ".nc" else xr.open_dataset(path)
    if "wind_speed_kt" in ds:
        return ds["wind_speed_kt"]
    if "u10" in ds and "v10" in ds:
        from contracts.schemas import MS_TO_KT

        return np_hypot_kt(ds["u10"], ds["v10"], MS_TO_KT)
    raise ValueError(f"No wind_speed_kt or u10/v10 in {path}")


def np_hypot_kt(u, v, ms_to_kt: float) -> xr.DataArray:
    import numpy as np

    return xr.DataArray(
        np.hypot(u.values, v.values) * ms_to_kt,
        coords=u.coords,
        dims=u.dims,
        name="wind_speed_kt",
    )


if __name__ == "__main__":
    raise SystemExit(main())
