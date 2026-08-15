"""Build production dashboard payload from real fetched data."""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from contracts.schemas import (  # noqa: E402
    Route,
    angular_difference,
    load_course,
    load_routes,
    uv_to_tws_twd,
    validate_dashboard_payload,
)
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
VIZ_ONLY_WARNING = (
    "Data here is not guaranteed. It is for visualization and comparison only — "
    "not as the sole basis for a race decision."
)
BONIFACIO_PANEL_LABEL = (
    "Channel choice inside the Maddalena archipelago is below model resolution. "
    "These are the conditions on arrival - the choice is local knowledge."
)


def _as_utc_timestamp(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _load_wind(path: Path) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Wind store not found: {path}")
    return xr.open_zarr(path, consolidated=True)


def _find_live_wind_store() -> Path | None:
    candidates = sorted((ROOT / "data/wind").glob("live-*.zarr"))
    if not candidates:
        return None
    return candidates[-1]


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


def _tag_value(route: Route, key: str) -> str:
    prefix = f"{key}="
    for tag in route.tags:
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    raise ValueError(f"Route '{route.id}' missing tag {key}=...")


def _common_prefix_len(a: tuple[tuple[float, float], ...], b: tuple[tuple[float, float], ...]) -> int:
    n = min(len(a), len(b))
    for idx in range(n):
        if a[idx] != b[idx]:
            return idx
    return n


def _build_leg_routes(routes: list[Route]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Route]] = defaultdict(dict)
    for route in routes:
        offset = _tag_value(route, "leg1_offset_nm")
        fork = _tag_value(route, "leg2_fork")
        grouped[offset][fork] = route

    legs: dict[str, dict[str, Any]] = {}
    for offset, forks in grouped.items():
        if set(forks.keys()) != {"east_of_corsica", "bonifacio_west_corsica"}:
            raise ValueError(f"Offset {offset} does not contain the two required leg2 forks.")
        east_route = forks["east_of_corsica"]
        bon_route = forks["bonifacio_west_corsica"]
        prefix_len = _common_prefix_len(east_route.legs, bon_route.legs)
        if prefix_len < 3:
            raise ValueError(f"Offset {offset} routes do not share a usable leg1 prefix.")

        gate_proxy = east_route.legs[prefix_len - 2]
        post_gate_join = east_route.legs[prefix_len - 1]
        leg1_route = Route(
            id=f"leg1_{offset}",
            label=f"Leg1 {offset}",
            description=f"Leg1 route for offset {offset}",
            legs=east_route.legs[: prefix_len - 1],
            tags=(f"leg1_offset_nm={offset}",),
        )

        leg2_routes: dict[str, Route] = {}
        for fork_id, full_route in forks.items():
            leg2_routes[fork_id] = Route(
                id=f"leg2_{offset}_{fork_id}",
                label=f"Leg2 {offset} {fork_id}",
                description=f"Leg2 fork {fork_id} for offset {offset}",
                legs=(gate_proxy,) + full_route.legs[prefix_len - 1 :],
                tags=(f"leg1_offset_nm={offset}", f"leg2_fork={fork_id}"),
            )

        bon_entry_point = bon_route.legs[prefix_len]
        bon_entry_route = Route(
            id=f"bon_entry_{offset}",
            label=f"Bonifacio entry {offset}",
            description=f"Gate proxy to Bonifacio entry for offset {offset}",
            legs=(gate_proxy, post_gate_join, bon_entry_point),
            tags=(f"leg1_offset_nm={offset}", "leg2_fork=bonifacio_west_corsica"),
        )

        legs[offset] = {
            "full_routes": forks,
            "leg1_route": leg1_route,
            "leg2_routes": leg2_routes,
            "bon_entry_route": bon_entry_route,
            "bon_entry_point": bon_entry_point,
        }
    return legs


def _combine_full_row(
    route_id: str,
    start_time: datetime,
    leg1: dict[str, Any],
    leg2: dict[str, Any],
) -> dict[str, Any]:
    elapsed = float(leg1["elapsed_hours"]) + float(leg2["elapsed_hours"])
    dist = float(leg1["distance_nm"]) + float(leg2["distance_nm"])
    mean_tws = np.nan
    if elapsed > 0:
        mean_tws = (
            float(leg1["mean_tws_kt"]) * float(leg1["elapsed_hours"])
            + float(leg2["mean_tws_kt"]) * float(leg2["elapsed_hours"])
        ) / elapsed
    max_stall = max(float(leg1["max_stall_hours"]), float(leg2["max_stall_hours"]))
    return {
        "start_time": _as_utc_timestamp(start_time),
        "route_id": route_id,
        "elapsed_hours": elapsed,
        "distance_nm": dist,
        "mean_tws_kt": mean_tws,
        "hours_below_5kt": float(leg1["hours_below_5kt"]) + float(leg2["hours_below_5kt"]),
        "hours_upwind": float(leg1["hours_upwind"]) + float(leg2["hours_upwind"]),
        "land_fill_fraction": max(float(leg1["land_fill_fraction"]), float(leg2["land_fill_fraction"])),
        "stalled": bool(leg1["stalled"] or leg2["stalled"]),
        "max_stall_hours": max_stall,
    }


def _route_summaries(routes: list[Route], results: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
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


def _sample_uv(wind: xr.Dataset, when: datetime, lat: float, lon: float) -> tuple[float, float]:
    ts = np.datetime64(_as_utc_timestamp(when).to_datetime64(), "ns")
    u10, v10, _ = follow_mod._interpolate_uv(wind, ts, float(lat), float(lon))
    if np.isfinite([u10, v10]).all():
        return float(u10), float(v10)
    nearest = wind.sel(time=ts, lat=lat, lon=lon, method="nearest")
    return float(nearest["u10"].values), float(nearest["v10"].values)


def _nw_component_kt(tws_kt: float, twd_deg: float, center_bearing: float = 310.0) -> float:
    # Positive contribution when wind direction is in the requested NW sector.
    delta = float(abs(angular_difference(twd_deg, center_bearing)))
    if delta > 20.0:
        return 0.0
    return max(0.0, tws_kt * float(np.cos(np.radians(delta))))


def _wind_metrics_from_uv(u10: float, v10: float) -> tuple[float, float, float]:
    tws, twd = uv_to_tws_twd(u10, v10)
    tws_kt = float(np.asarray(tws))
    twd_deg = float(np.asarray(twd))
    nw_component = _nw_component_kt(tws_kt, twd_deg)
    return tws_kt, twd_deg, nw_component


def _build_hourly_climatology_at_point(
    wind: xr.Dataset, lat: float, lon: float, display_tz: str
) -> dict[int, dict[str, Any]]:
    point = wind.sel(lat=lat, lon=lon, method="nearest")
    df = point[["u10", "v10"]].to_dataframe().reset_index()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    tws, twd = uv_to_tws_twd(df["u10"].to_numpy(), df["v10"].to_numpy())
    df["tws_kt"] = np.asarray(tws, dtype=float)
    df["twd_deg"] = np.asarray(twd, dtype=float)
    df["nw_component_kt"] = [
        _nw_component_kt(float(s), float(d))
        for s, d in zip(df["tws_kt"].tolist(), df["twd_deg"].tolist())
    ]
    df["local_hour"] = df["time"].dt.tz_convert(display_tz).dt.hour

    stats: dict[int, dict[str, Any]] = {}
    for hour, group in df.groupby("local_hour"):
        stats[int(hour)] = {
            "n": int(len(group)),
            "tws_kt": {
                "p10": float(np.nanpercentile(group["tws_kt"], 10)),
                "p50": float(np.nanpercentile(group["tws_kt"], 50)),
                "p90": float(np.nanpercentile(group["tws_kt"], 90)),
            },
            "twd_deg": {
                "p10": float(np.nanpercentile(group["twd_deg"], 10)),
                "p50": float(np.nanpercentile(group["twd_deg"], 50)),
                "p90": float(np.nanpercentile(group["twd_deg"], 90)),
            },
            "nw_component_kt": {
                "p10": float(np.nanpercentile(group["nw_component_kt"], 10)),
                "p50": float(np.nanpercentile(group["nw_component_kt"], 50)),
                "p90": float(np.nanpercentile(group["nw_component_kt"], 90)),
            },
        }
    return stats


# IFS analysis is ~9 km; same-day AROME comparison showed Sardinia samples
# inside ~10 nm collapse to one IFS cell. Do not present that flat as a finding.
IFS_TRANSECT_BELOW_RESOLUTION_NM = 10.0
IFS_TRANSECT_RESOLUTION_NOTE = (
    "Shaded band (0–10 nm): below IFS 9 km model resolution — "
    "inner distances sample one grid cell, not a coastal gradient."
)


def _format_transects_for_dashboard(transects: pd.DataFrame) -> list[dict[str, Any]]:
    if transects.empty:
        return []
    output: list[dict[str, Any]] = []
    for transect_id, group in transects.groupby("transect_id"):
        by_hour: list[dict[str, Any]] = []
        dist = sorted(float(v) for v in group["distance_offshore_nm"].drop_duplicates().tolist())
        for hour, hgroup in group.groupby("hour"):
            values = (
                hgroup.sort_values("distance_offshore_nm")
                .set_index("distance_offshore_nm")["mean_tws_kt"]
            )
            by_hour.append(
                {
                    "hour": int(hour),
                    "mean_tws_kt": [float(values.get(d, np.nan)) for d in dist],
                }
            )
        output.append(
            {
                "id": str(transect_id),
                "name": str(group["coast_name"].iloc[0]),
                "distance_nm": dist,
                "by_hour": sorted(by_hour, key=lambda r: r["hour"]),
                "below_model_resolution_nm": IFS_TRANSECT_BELOW_RESOLUTION_NM,
                "resolution_note": IFS_TRANSECT_RESOLUTION_NOTE,
                "source_model": "ecmwf_ifs_analysis_9km",
            }
        )
    return sorted(output, key=lambda t: t["id"])


def _load_skill_rows() -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    path = ROOT / "data/skill/skill_rows.json"
    if not path.exists():
        return [], None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, None
    rows = payload.get("skill") or []
    meta = payload.get("meta")
    return list(rows), meta if isinstance(meta, dict) else None


def main() -> int:
    wind_path = ROOT / "data/wind/analysis-august.zarr"
    live_wind_path = _find_live_wind_store()
    polar_path = ROOT / "config/polar/boat.pol"
    course_path = ROOT / "config/course.yaml"
    routes_path = ROOT / "config/routes.yaml"
    output_path = ROOT / "dashboard/data.json"

    wind = _load_wind(wind_path)
    live_wind = _load_wind(live_wind_path) if live_wind_path is not None else None
    try:
        polar = _load_polar(polar_path)
        course = load_course(course_path)
        routes = load_routes(routes_path)
        route_legs = _build_leg_routes(routes)

        starts = _august_start_times(wind, course.start_time_utc)
        if not starts:
            raise ValueError("No August timesteps found in wind cube.")

        full_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        leg2_hourly_margins: dict[int, list[float]] = defaultdict(list)
        entry_hourly_wind: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for offset, spec in route_legs.items():
            for start in starts:
                leg1_result = follow_mod.follow(spec["leg1_route"], wind, polar, start).as_row()
                gate_start = start + timedelta(hours=float(leg1_result["elapsed_hours"]))
                gate_hour_local = (
                    _as_utc_timestamp(gate_start)
                    .tz_convert(course.display_timezone)
                    .hour
                )

                leg2_east = follow_mod.follow(
                    spec["leg2_routes"]["east_of_corsica"],
                    wind,
                    polar,
                    gate_start,
                ).as_row()
                leg2_bon = follow_mod.follow(
                    spec["leg2_routes"]["bonifacio_west_corsica"],
                    wind,
                    polar,
                    gate_start,
                ).as_row()

                east_id = spec["full_routes"]["east_of_corsica"].id
                bon_id = spec["full_routes"]["bonifacio_west_corsica"].id
                full_rows[east_id].append(_combine_full_row(east_id, start, leg1_result, leg2_east))
                full_rows[bon_id].append(_combine_full_row(bon_id, start, leg1_result, leg2_bon))

                margin_hours = float(leg2_bon["elapsed_hours"]) - float(leg2_east["elapsed_hours"])
                leg2_hourly_margins[int(gate_hour_local)].append(margin_hours)

                entry_result = follow_mod.follow(spec["bon_entry_route"], wind, polar, gate_start).as_row()
                entry_time = gate_start + timedelta(hours=float(entry_result["elapsed_hours"]))
                u10, v10 = _sample_uv(wind, entry_time, *spec["bon_entry_point"])
                tws_kt, twd_deg, nw_component_kt = _wind_metrics_from_uv(u10, v10)
                entry_hourly_wind[offset].append(
                    {
                        "entry_time_utc": entry_time,
                        "tws_kt": tws_kt,
                        "twd_deg": twd_deg,
                        "nw_component_kt": nw_component_kt,
                    }
                )

        results: dict[str, pd.DataFrame] = {}
        for route in routes:
            frame = pd.DataFrame(full_rows[route.id])
            frame["start_time"] = pd.to_datetime(frame["start_time"], utc=True)
            results[route.id] = frame

        climatology = stats_mod.climatology(wind, [8])
        transects = stats_mod.cross_shore_transects(climatology)
        h2h = stats_mod.head_to_head(results)
        route_summaries = _route_summaries(routes, results)

        leg2_hour_rows: list[dict[str, Any]] = []
        for hour in sorted(leg2_hourly_margins):
            margins = np.asarray(leg2_hourly_margins[hour], dtype=float)
            if margins.size == 0:
                continue
            leg2_hour_rows.append(
                {
                    "arrival_hour_local": int(hour),
                    "n": int(margins.size),
                    "east_of_corsica_wins_pct": float(np.mean(margins > 0.0) * 100.0),
                    "bonifacio_west_corsica_wins_pct": float(np.mean(margins < 0.0) * 100.0),
                    "median_margin_hours": float(np.median(margins)),
                    "p10_margin_hours": float(np.quantile(margins, 0.10)),
                    "p90_margin_hours": float(np.quantile(margins, 0.90)),
                }
            )

        bon_entry_point = next(iter(route_legs.values()))["bon_entry_point"]
        climo_at_entry_hour = _build_hourly_climatology_at_point(
            wind, bon_entry_point[0], bon_entry_point[1], course.display_timezone
        )
        bonifacio_panel_rows: list[dict[str, Any]] = []
        for offset, hist_rows in sorted(entry_hourly_wind.items(), key=lambda kv: float(kv[0])):
            leg1_route = route_legs[offset]["leg1_route"]
            bon_entry_route = route_legs[offset]["bon_entry_route"]
            race_entry_time = None
            race_forecast = None
            if live_wind is not None:
                race_leg1 = follow_mod.follow(leg1_route, live_wind, polar, course.start_time_utc).as_row()
                race_gate = course.start_time_utc + timedelta(hours=float(race_leg1["elapsed_hours"]))
                race_entry = follow_mod.follow(bon_entry_route, live_wind, polar, race_gate).as_row()
                race_entry_time = race_gate + timedelta(hours=float(race_entry["elapsed_hours"]))
                u10, v10 = _sample_uv(live_wind, race_entry_time, *bon_entry_point)
                tws_kt, twd_deg, nw_component_kt = _wind_metrics_from_uv(u10, v10)
                race_forecast = {
                    "tws_kt": tws_kt,
                    "twd_deg": twd_deg,
                    "nw_component_kt": nw_component_kt,
                }

            if race_entry_time is not None:
                local_hour = int(_as_utc_timestamp(race_entry_time).tz_convert(course.display_timezone).hour)
                local_time = _as_utc_timestamp(race_entry_time).tz_convert(course.display_timezone)
                modeled_entry_local = local_time.strftime("%Y-%m-%d %H:%M %Z")
            else:
                local_hours_hist = [
                    int(_as_utc_timestamp(r["entry_time_utc"]).tz_convert(course.display_timezone).hour)
                    for r in hist_rows
                ]
                local_hour = int(np.median(local_hours_hist)) if local_hours_hist else 0
                modeled_entry_local = "unavailable"

            bonifacio_panel_rows.append(
                {
                    "leg1_offset_nm": float(offset),
                    "modeled_entry_time_local": modeled_entry_local,
                    "arrival_hour_local": local_hour,
                    "race_day_forecast": race_forecast,
                    "august_climatology_at_hour": climo_at_entry_hour.get(local_hour),
                }
            )

        times = pd.to_datetime(wind["time"].values, utc=True)
        august_years = sorted(int(y) for y in np.unique(times[times.month == 8].year))
        skill_rows, skill_meta = _load_skill_rows()
        meta = {
            "generated_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "display_timezone": course.display_timezone,
            "course": {
                "start": [course.start[0], course.start[1]],
                "gate": [course.gate[0], course.gate[1]],
                "finish": [course.finish[0], course.finish[1]],
            },
            "polar_name": "boat_vpp",
            "polar_is_validated": False,
            "climatology_years": august_years,
            "warnings": [
                VIZ_ONLY_WARNING,
                WARNING_TEXT,
            ],
        }
        if skill_meta:
            meta["skill"] = skill_meta
            bias_warning = (
                "Model skill scores ECMWF IFS / AIFS against IFS analysis "
                "(reference-biased). Independent models are comparable to each other; "
                "biased rows are not."
            )
            if bias_warning not in meta["warnings"]:
                meta["warnings"].append(bias_warning)

        payload_path = report_mod.emit(
            climatology_ds=climatology,
            routes_summary=route_summaries,
            head_to_head_df=h2h,
            skill_rows=skill_rows,
            meta=meta,
            output_path=output_path,
            extra_sections={
                "transects": _format_transects_for_dashboard(transects),
                "leg2_win_rate_by_arrival_hour_local": leg2_hour_rows,
                "bonifacio_decision_support": {
                    "label": BONIFACIO_PANEL_LABEL,
                    "entry_point": [bon_entry_point[0], bon_entry_point[1]],
                    "rows": bonifacio_panel_rows,
                },
            },
        )
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        validate_dashboard_payload(payload)
        print(payload_path)
        print(f"august_years={','.join(str(y) for y in august_years)}")
        return 0
    finally:
        wind.close()
        if live_wind is not None:
            live_wind.close()


if __name__ == "__main__":
    raise SystemExit(main())
