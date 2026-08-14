"""CLI for the Cluster D Open-Meteo fetcher."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

from .openmeteo import OpenMeteoFetcher, load_domain


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch Open-Meteo wind grids into zarr stores.")
    parser.add_argument("--source", required=True, choices=["analysis", "previous_runs", "live"])
    parser.add_argument("--start", required=True, type=parse_date, help="Start date (YYYY-MM-DD, UTC)")
    parser.add_argument("--end", required=True, type=parse_date, help="End date (YYYY-MM-DD, UTC)")
    parser.add_argument("--domain-config", default="config/domain.yaml", help="Domain config path")
    parser.add_argument("--output-path", default=None, help="Override output .zarr path")
    parser.add_argument("--force-model", default=None, help="Force one discovered model id")
    parser.add_argument(
        "--refresh-models",
        action="store_true",
        help="Ignore config/models.yaml cache and probe model IDs again",
    )
    parser.add_argument("--batch-size", type=int, default=180, help="Coordinates per API request")
    parser.add_argument("--max-workers", type=int, default=8, help="Concurrent request workers (max 8)")
    parser.add_argument("--cache-root", default="data/cache/openmeteo", help="Request cache directory")
    parser.add_argument("--wind-root", default="data/wind", help="Wind output directory root")
    parser.add_argument("--timeout-seconds", type=int, default=60, help="Per-request timeout")
    parser.add_argument("--retries", type=int, default=6, help="Retry attempts for transient failures")
    parser.add_argument(
        "--inter-day-delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay between processed days (useful for testing resume/rate pacing).",
    )
    parser.add_argument(
        "--min-request-interval-seconds",
        type=float,
        default=1.0,
        help="Minimum spacing between uncached HTTP requests.",
    )
    parser.add_argument(
        "--month-filter",
        default=None,
        help="Comma-separated month numbers to include (e.g. 8 or 6,7,8).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.max_workers > 8:
        parser.error("--max-workers cannot exceed 8")

    domain = load_domain(Path(args.domain_config))
    month_filter = None
    if args.month_filter:
        month_filter = {
            int(token.strip())
            for token in args.month_filter.split(",")
            if token.strip()
        }
        if any(month < 1 or month > 12 for month in month_filter):
            parser.error("--month-filter only accepts month numbers 1-12")

    fetcher = OpenMeteoFetcher(
        api_key=os.getenv("OPENMETEO_API_KEY"),
        cache_root=Path(args.cache_root),
        output_root=Path(args.wind_root),
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        inter_day_delay_seconds=args.inter_day_delay_seconds,
        min_request_interval_seconds=args.min_request_interval_seconds,
    )
    output_path = Path(args.output_path) if args.output_path else None
    path, summary = fetcher.fetch_wind(
        source=args.source,
        start=args.start,
        end=args.end,
        cfg=domain,
        output_path=output_path,
        force_model=args.force_model,
        refresh_models=args.refresh_models,
        month_filter=month_filter,
    )
    print(path)
    print(json.dumps(summary.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

