"""Build production dashboard payload from real fetched data."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from contracts.schemas import load_course, load_routes, validate_dashboard_payload  # noqa: E402
from pmc import follow as follow_mod  # noqa: E402
from pmc import polar as polar_mod  # noqa: E402
from pmc import report as report_mod  # noqa: E402
from pmc import stats as stats_mod  # noqa: E402

logging.getLogger("pmc.follow").setLevel(logging.ERROR)


WARNING_TEXT = (
    "Polar is the boat's VPP certificate table, not measured. "
    "VPP polars are optimistic in light air. "
    "Elapsed times are relative comparisons, not predictions."
)


def _load_wind(path: Path) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Wind store not found: {path}")
    return xr.open_zarr(path, consolidated=True)


def _load_polar(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Polar file not found: {path}")
    return polar_mod.load_polar(path)


def _august_start_times(wind: xr.Dataset, start_template: datetime) -> list[datetime]:
    times = pd.to_datetime(wind["time"].values, utc=True)
    august = times[times.month == 8]
    unique_days = (
        pd.Series(august.date).drop_duplicates().sort_values().tolist()
        if len(august) > 0
        else []
    )
    starts: list[datetime] = []
    for day in unique_days:
        starts.append(
            datetime(
                year=day.year,
                month=day.month,
                day=day.day,
                hour=start_template.hour,
                minute=start_template.minute,
                second=start_template.second,
                tzinfo=timezone.utc,
            )
        )
    return starts


def _route_summaries(routes, results: dict[str, pd.DataFrame]) -> list[dict]:
    output: list[dict] = []
    for route in routes:
        frame = results[route.id]
        samples = frame["elapsed_hours"].dropna().to_numpy(dtype=float)
        if samples.size == 0:
            p10 = p50 = p90 = np.nan
        else:
            p10, p50, p90 = np.percentile(samples, [10, 50, 90])
        output.append(
            {
                "id": route.id,
                "label": route.label,
                "legs": [[lat, lon] for lat, lon in route.legs],
                "elapsed_hours": {
                    "p10": float(p10) if np.isfinite(p10) else None,
                    "p50": float(p50) if np.isfinite(p50) else None,
                    "p90": float(p90) if np.isfinite(p90) else None,
                    "samples": samples.tolist(),
                },
                "stall_rate": float(frame["stalled"].mean()) if len(frame) else None,
            }
        )
    return output


def main() -> int:
    wind_path = ROOT / "data/wind/analysis-august-public.zarr"
    polar_path = ROOT / "config/polar/chocolate3.pol"
    course_path = ROOT / "config/course.yaml"
    routes_path = ROOT / "config/routes.yaml"
    output_path = ROOT / "dashboard/data.json"

    wind = _load_wind(wind_path)
    try:
        polar = _load_polar(polar_path)
        course = load_course(course_path)
        routes = load_routes(routes_path)

        starts = _august_start_times(wind, course.start_time_utc)
        if not starts:
            raise ValueError("No August timesteps found in wind cube.")

        results: dict[str, pd.DataFrame] = {}
        for route in routes:
            rows = []
            for start in starts:
                rows.append(follow_mod.follow(route, wind, polar, start).as_row())
            results[route.id] = pd.DataFrame(rows)

        climatology = stats_mod.climatology(wind, [8])
        transects = stats_mod.cross_shore_transects(climatology)
        h2h = stats_mod.head_to_head(results)
        route_summaries = _route_summaries(routes, results)

        times = pd.to_datetime(wind["time"].values, utc=True)
        august_years = sorted(int(y) for y in np.unique(times[times.month == 8].year))

        meta = {
            "generated_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "display_timezone": course.display_timezone,
            "course": {
                "start": [course.start[0], course.start[1]],
                "gate": [course.gate[0], course.gate[1]],
                "finish": [course.finish[0], course.finish[1]],
            },
            "polar_name": "Chocolate3",
            "polar_is_validated": False,
            "warnings": [
                WARNING_TEXT,
                f"August years present in wind cube: {', '.join(str(y) for y in august_years)}",
            ],
        }

        payload_path = report_mod.emit(
            climatology_ds=climatology,
            routes_summary=route_summaries,
            head_to_head_df=h2h,
            skill_rows=[],
            meta=meta,
            output_path=output_path,
            extra_sections={
                "transects": transects.to_dict(orient="records"),
            },
        )
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        validate_dashboard_payload(payload)
        print(payload_path)
        print(f"august_years={','.join(str(y) for y in august_years)}")
        return 0
    finally:
        wind.close()


if __name__ == "__main__":
    raise SystemExit(main())
