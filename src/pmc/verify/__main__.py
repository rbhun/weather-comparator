"""CLI for live observation verification."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pmc.verify.ingest import ingest_observations
from pmc.verify.pipeline import build_current_weather_payload, run_verification_pass
from pmc.verify.store import VerifyStore, load_verify_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Current-weather live verification against scatterometer / METAR / Sentinel-1."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Parse an observation file (auto-detect format).")
    p_ingest.add_argument("path", type=Path)
    p_ingest.add_argument("--dry-run", action="store_true")

    p_pass = sub.add_parser("pass", help="Run a verification pass from obs + forecast tables.")
    p_pass.add_argument("--pass-id", required=True)
    p_pass.add_argument("--obs", type=Path, nargs="*", default=[])
    p_pass.add_argument("--forecasts", type=Path, help="Parquet/CSV of model forecasts")
    p_pass.add_argument("--config", type=Path, default=ROOT / "config/verify.yaml")
    p_pass.add_argument("--store", type=Path, default=ROOT / "data/verify")

    p_emit = sub.add_parser("emit-payload", help="Build current_weather JSON section from store.")
    p_emit.add_argument("--store", type=Path, default=ROOT / "data/verify")
    p_emit.add_argument("--config", type=Path, default=ROOT / "config/verify.yaml")
    p_emit.add_argument("--stations", type=Path, default=ROOT / "config/stations.yaml")
    p_emit.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        report = ingest_observations(args.path, dry_run=args.dry_run)
        print(
            json.dumps(
                {
                    "path": report.path,
                    "detected_format": report.detected_format,
                    "obs_class": report.obs_class,
                    "instrument": report.instrument,
                    "n_cells": report.n_cells,
                    "n_rejected_qc": report.n_rejected_qc,
                    "dry_run": report.dry_run,
                    "source_file_hash": report.source_file_hash,
                    "messages": report.messages,
                },
                indent=2,
            )
        )
        return 0 if report.n_cells or report.messages else 1

    if args.cmd == "pass":
        cfg = load_verify_config(args.config, store_dir=args.store)
        forecasts = None
        if args.forecasts:
            if args.forecasts.suffix.lower() == ".parquet":
                forecasts = pd.read_parquet(args.forecasts)
            else:
                forecasts = pd.read_csv(args.forecasts)
                for col in ("run_init", "valid_time"):
                    if col in forecasts.columns:
                        forecasts[col] = pd.to_datetime(forecasts[col], utc=True).dt.tz_localize(
                            None
                        )
        summary = run_verification_pass(
            args.pass_id,
            cfg,
            observation_files=list(args.obs),
            forecasts=forecasts,
        )
        print(
            json.dumps(
                {
                    "pass_id": summary.pass_id,
                    "scatterometer_new_rows": summary.scatterometer_new_rows,
                    "land_station_new_rows": summary.land_station_new_rows,
                    "sentinel_status": summary.sentinel_status,
                    "bucket_counts": summary.bucket_counts,
                    "noop_classes": summary.noop_classes,
                    "messages": summary.messages,
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "emit-payload":
        cfg = load_verify_config(args.config, store_dir=args.store)
        payload = build_current_weather_payload(
            args.store,
            cfg=cfg,
            stations_yaml=args.stations,
            generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
