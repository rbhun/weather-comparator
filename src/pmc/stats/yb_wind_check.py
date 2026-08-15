"""YB historical tracks as an independent check on IFS analysis wind.

At each timed track segment, compare observed SOG with the boat speed the
analysis wind + polar would allow on that COG. Positive residual means the
fleet sailed faster than the analysis field permits — candidate evidence that
the analysis under-represents local wind (e.g. coastal thermal).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xarray as xr

from contracts.schemas import (
    MS_TO_KT,
    Polar,
    angular_difference,
    haversine_nm,
    initial_bearing_deg,
    uv_to_tws_twd,
)
from pmc.geo import distance_to_land_nm
from pmc.io.yb import Boat, DEFAULT_YEARS, load_edition_full_tracks

OFFSHORE_BINS_NM = (0.0, 5.0, 10.0, 20.0, 40.0, np.inf)
OFFSHORE_LABELS = ("0-5nm", "5-10nm", "10-20nm", "20-40nm", "40nm+")
TWS_BINS_KT = (0.0, 6.0, 12.0, 20.0, np.inf)
TWS_LABELS = ("0-6kt", "6-12kt", "12-20kt", "20kt+")
DEFAULT_INTERVAL_MIN = 20
MIN_INTERVAL_MIN = 15
MAX_INTERVAL_MIN = 30
MIN_SOG_KT = 0.3
MAX_SOG_KT = 35.0
AFTERNOON_HOURS_LOCAL = frozenset(range(12, 18))
NEAR_COAST_NM = 10.0


@dataclass(frozen=True)
class WindCheckConfig:
    interval_min: int = DEFAULT_INTERVAL_MIN
    min_interval_min: int = MIN_INTERVAL_MIN
    max_interval_min: int = MAX_INTERVAL_MIN
    display_timezone: str = "Europe/Rome"
    min_sog_kt: float = MIN_SOG_KT
    max_sog_kt: float = MAX_SOG_KT


def _parse_times(times: Sequence[str]) -> np.ndarray:
    parsed: list[np.datetime64] = []
    for raw in times:
        text = raw.replace("Z", "+00:00") if isinstance(raw, str) else str(raw)
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        parsed.append(np.datetime64(dt, "ns"))
    return np.asarray(parsed, dtype="datetime64[ns]")


def track_motion_samples(
    boat: Boat,
    *,
    cfg: WindCheckConfig = WindCheckConfig(),
) -> pd.DataFrame:
    """Emit mid-segment SOG/COG samples at ~cfg.interval_min spacing."""

    if len(boat.lat) < 2 or len(boat.lon) < 2:
        return pd.DataFrame()
    if not boat.times or len(boat.times) != len(boat.lat):
        return pd.DataFrame()

    times = _parse_times(boat.times)
    lats = np.asarray(boat.lat, dtype=float)
    lons = np.asarray(boat.lon, dtype=float)
    order = np.argsort(times)
    times = times[order]
    lats = lats[order]
    lons = lons[order]

    target = np.timedelta64(int(cfg.interval_min), "m")
    min_dt = np.timedelta64(int(cfg.min_interval_min), "m")
    max_dt = np.timedelta64(int(cfg.max_interval_min), "m")

    rows: list[dict[str, Any]] = []
    last_idx = 0
    for idx in range(1, len(times)):
        dt = times[idx] - times[last_idx]
        if dt < min_dt and idx < len(times) - 1:
            continue
        if dt <= np.timedelta64(0, "s"):
            last_idx = idx
            continue
        # Prefer ~target interval; allow up to max when track is sparse.
        if dt < target and idx < len(times) - 1:
            continue
        if dt > max_dt * 3:
            # Huge gap (mooring / hole) — skip and restart.
            last_idx = idx
            continue

        dist_nm = float(haversine_nm(lats[last_idx], lons[last_idx], lats[idx], lons[idx]))
        hours = float(dt / np.timedelta64(1, "h"))
        if hours <= 0:
            last_idx = idx
            continue
        sog_kt = dist_nm / hours
        if sog_kt < cfg.min_sog_kt or sog_kt > cfg.max_sog_kt:
            last_idx = idx
            continue
        if dt > max_dt and sog_kt < 1.0:
            last_idx = idx
            continue

        cog = initial_bearing_deg(lats[last_idx], lons[last_idx], lats[idx], lons[idx])
        mid_lat = 0.5 * (lats[last_idx] + lats[idx])
        mid_lon = 0.5 * (lons[last_idx] + lons[idx])
        mid_time = times[last_idx] + dt / 2
        rows.append(
            {
                "year": boat.year,
                "boat": boat.name,
                "time_utc": mid_time,
                "lat": float(mid_lat),
                "lon": float(mid_lon),
                "sog_kt": float(sog_kt),
                "cog_deg": float(cog),
                "segment_min": float(dt / np.timedelta64(1, "m")),
            }
        )
        last_idx = idx

    return pd.DataFrame(rows)


def collect_motion_samples(
    boats: Iterable[Boat],
    *,
    cfg: WindCheckConfig = WindCheckConfig(),
) -> pd.DataFrame:
    frames = [track_motion_samples(boat, cfg=cfg) for boat in boats]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
                "year",
                "boat",
                "time_utc",
                "lat",
                "lon",
                "sog_kt",
                "cog_deg",
                "segment_min",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _interp_1d(values: np.ndarray, x: float) -> tuple[int, int, float]:
    idx_hi = int(np.searchsorted(values, x, side="right"))
    if idx_hi <= 0:
        return 0, 0, 0.0
    if idx_hi >= values.size:
        i = values.size - 1
        return i, i, 0.0
    idx_lo = idx_hi - 1
    x0 = float(values[idx_lo])
    x1 = float(values[idx_hi])
    if x1 == x0:
        return idx_lo, idx_hi, 0.0
    return idx_lo, idx_hi, float(np.clip((x - x0) / (x1 - x0), 0.0, 1.0))


def _bilinear_uv(
    field_u: np.ndarray,
    field_v: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    lat: float,
    lon: float,
) -> tuple[float, float]:
    lat_c = float(np.clip(lat, lat_vals[0], lat_vals[-1]))
    lon_c = float(np.clip(lon, lon_vals[0], lon_vals[-1]))
    i0, i1, wa = _interp_1d(lat_vals, lat_c)
    j0, j1, wb = _interp_1d(lon_vals, lon_c)

    def _sample(field: np.ndarray) -> float:
        corners = np.array(
            [field[i0, j0], field[i1, j0], field[i0, j1], field[i1, j1]], dtype=float
        )
        valid = np.isfinite(corners)
        if valid.sum() == 0:
            return float("nan")
        if valid.sum() < 4:
            return float(np.nanmean(corners))
        f00, f10, f01, f11 = corners
        return float(
            (1.0 - wa) * (1.0 - wb) * f00
            + wa * (1.0 - wb) * f10
            + (1.0 - wa) * wb * f01
            + wa * wb * f11
        )

    return _sample(field_u), _sample(field_v)


def sample_analysis_wind(wind: xr.Dataset, time_utc: np.datetime64, lat: float, lon: float) -> tuple[float, float]:
    """Bilinear-in-space, linear-in-time sample of analysis u/v (m/s)."""

    times = wind["time"].values.astype("datetime64[ns]")
    time_ns = times.astype(np.int64)
    t_ns = int(np.datetime64(time_utc, "ns").astype(np.int64))
    i0, i1, wt = _interp_1d(time_ns, t_ns)
    lat_vals = wind["lat"].values.astype(float)
    lon_vals = wind["lon"].values.astype(float)
    u0, v0 = _bilinear_uv(wind["u10"].values[i0], wind["v10"].values[i0], lat_vals, lon_vals, lat, lon)
    if i0 == i1:
        return u0, v0
    u1, v1 = _bilinear_uv(wind["u10"].values[i1], wind["v10"].values[i1], lat_vals, lon_vals, lat, lon)
    if not np.isfinite([u0, v0, u1, v1]).all():
        u_c = [v for v in (u0, u1) if np.isfinite(v)]
        v_c = [v for v in (v0, v1) if np.isfinite(v)]
        if not u_c or not v_c:
            return float("nan"), float("nan")
        return float(np.mean(u_c)), float(np.mean(v_c))
    return (1.0 - wt) * u0 + wt * u1, (1.0 - wt) * v0 + wt * v1


def predicted_speed_kt(polar: Polar, twa_deg: float, tws_kt: float) -> float:
    """Polar boat speed available at the observed track TWA / analysis TWS."""

    return float(np.asarray(polar.speed(abs(float(twa_deg)), float(tws_kt))))


def annotate_samples_with_wind(
    samples: pd.DataFrame,
    wind: xr.Dataset,
    polar: Polar,
    *,
    cfg: WindCheckConfig = WindCheckConfig(),
    compute_offshore: bool = True,
) -> pd.DataFrame:
    """Attach analysis wind, polar prediction, residual, and bin labels."""

    if samples.empty:
        return samples.copy()

    out = samples.copy()
    out["time_utc"] = pd.to_datetime(out["time_utc"], utc=True)
    u_vals: list[float] = []
    v_vals: list[float] = []
    for row in out.itertuples(index=False):
        t = np.datetime64(pd.Timestamp(row.time_utc).tz_convert("UTC").tz_localize(None).to_datetime64())
        u, v = sample_analysis_wind(wind, t, float(row.lat), float(row.lon))
        u_vals.append(u)
        v_vals.append(v)

    out["u10_ms"] = np.asarray(u_vals, dtype=float)
    out["v10_ms"] = np.asarray(v_vals, dtype=float)
    tws, twd = uv_to_tws_twd(out["u10_ms"].to_numpy(), out["v10_ms"].to_numpy())
    out["analysis_tws_kt"] = tws
    out["analysis_twd_deg"] = twd
    twa = angular_difference(out["analysis_twd_deg"].to_numpy(), out["cog_deg"].to_numpy())
    out["twa_deg"] = twa
    out["predicted_sog_kt"] = np.asarray(
        [predicted_speed_kt(polar, float(a), float(s)) for a, s in zip(out["twa_deg"], out["analysis_tws_kt"])],
        dtype=float,
    )
    out["residual_kt"] = out["sog_kt"] - out["predicted_sog_kt"]

    tz = ZoneInfo(cfg.display_timezone)
    local = out["time_utc"].dt.tz_convert(tz)
    out["hour_local"] = local.dt.hour.astype(int)
    out["hour_utc"] = out["time_utc"].dt.hour.astype(int)

    if compute_offshore:
        out["distance_offshore_nm"] = distance_to_land_nm(
            out["lat"].to_numpy(), out["lon"].to_numpy()
        )
    elif "distance_offshore_nm" not in out.columns:
        out["distance_offshore_nm"] = np.nan

    out["offshore_bin"] = pd.cut(
        out["distance_offshore_nm"],
        bins=list(OFFSHORE_BINS_NM),
        labels=list(OFFSHORE_LABELS),
        right=False,
        include_lowest=True,
    )
    out["tws_bin"] = pd.cut(
        out["analysis_tws_kt"],
        bins=list(TWS_BINS_KT),
        labels=list(TWS_LABELS),
        right=False,
        include_lowest=True,
    )
    out["afternoon_local"] = out["hour_local"].isin(AFTERNOON_HOURS_LOCAL)
    out["near_coast"] = out["distance_offshore_nm"] < NEAR_COAST_NM
    out["polar_name"] = polar.name
    return out


def _bin_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=group_cols
            + [
                "n",
                "residual_median_kt",
                "residual_p10_kt",
                "residual_p90_kt",
                "sog_median_kt",
                "predicted_median_kt",
                "analysis_tws_median_kt",
            ]
        )
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(group_cols, dropna=True, observed=True)
    for keys, chunk in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        resid = chunk["residual_kt"].to_numpy(dtype=float)
        resid = resid[np.isfinite(resid)]
        if resid.size == 0:
            continue
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "n": int(resid.size),
                "residual_median_kt": round(float(np.median(resid)), 2),
                "residual_p10_kt": round(float(np.percentile(resid, 10)), 2),
                "residual_p90_kt": round(float(np.percentile(resid, 90)), 2),
                "sog_median_kt": round(float(np.nanmedian(chunk["sog_kt"])), 2),
                "predicted_median_kt": round(float(np.nanmedian(chunk["predicted_sog_kt"])), 2),
                "analysis_tws_median_kt": round(float(np.nanmedian(chunk["analysis_tws_kt"])), 2),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=group_cols
            + [
                "n",
                "residual_median_kt",
                "residual_p10_kt",
                "residual_p90_kt",
                "sog_median_kt",
                "predicted_median_kt",
                "analysis_tws_median_kt",
            ]
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def summarise_residuals(samples: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Median residual tables by the requested bin axes (plus key cross-tabs)."""

    valid = samples.dropna(subset=["residual_kt", "analysis_tws_kt"]).copy()
    valid = valid[np.isfinite(valid["residual_kt"]) & np.isfinite(valid["analysis_tws_kt"])]
    return {
        "by_offshore": _bin_summary(valid, ["offshore_bin"]),
        "by_hour_local": _bin_summary(valid, ["hour_local"]),
        "by_tws": _bin_summary(valid, ["tws_bin"]),
        "by_offshore_hour": _bin_summary(valid, ["offshore_bin", "hour_local"]),
        "by_coast_afternoon": _bin_summary(
            valid.assign(
                coast_window=np.where(
                    valid["near_coast"] & valid["afternoon_local"],
                    "near_coast_afternoon",
                    np.where(
                        (~valid["near_coast"]) & valid["afternoon_local"],
                        "offshore_afternoon",
                        np.where(
                            valid["near_coast"] & (~valid["afternoon_local"]),
                            "near_coast_other",
                            "offshore_other",
                        ),
                    ),
                )
            ),
            ["coast_window"],
        ),
    }


def load_years_full_tracks(
    years: Sequence[int] = DEFAULT_YEARS,
    cache_root: Path | None = None,
) -> list[Boat]:
    from pmc.io.yb import DEFAULT_CACHE

    root = cache_root or DEFAULT_CACHE
    boats: list[Boat] = []
    for year in years:
        boats.extend(load_edition_full_tracks(int(year), cache_root=root))
    return boats


def run_wind_check(
    boats: Sequence[Boat],
    wind: xr.Dataset,
    polar: Polar,
    *,
    cfg: WindCheckConfig = WindCheckConfig(),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """End-to-end: motion samples → wind/polar residuals → binned medians."""

    samples = collect_motion_samples(boats, cfg=cfg)
    annotated = annotate_samples_with_wind(samples, wind, polar, cfg=cfg)
    return annotated, summarise_residuals(annotated)


def format_summary_markdown(
    summaries: dict[str, pd.DataFrame],
    *,
    meta: dict[str, Any] | None = None,
) -> str:
    """Human-readable report of residual medians with sample counts."""

    lines: list[str] = [
        "# YB tracks vs analysis wind",
        "",
        "Residual = observed SOG − polar speed at (analysis TWS, track TWA).",
        "Positive residual ⇒ fleet faster than analysis+polar allows.",
        "",
    ]
    if meta:
        lines.append("## Meta")
        lines.append("")
        for key, value in meta.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    def _emit(title: str, frame: pd.DataFrame) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if frame.empty:
            lines.append("_No samples._")
            lines.append("")
            return
        lines.append(frame.to_markdown(index=False))
        lines.append("")

    _emit("By distance offshore", summaries["by_offshore"])
    _emit("By local hour (Europe/Rome unless overridden)", summaries["by_hour_local"])
    _emit("By analysis wind speed", summaries["by_tws"])
    _emit("Coast × afternoon window", summaries["by_coast_afternoon"])
    _emit("Offshore × local hour (cross-tab)", summaries["by_offshore_hour"])

    coast = summaries.get("by_coast_afternoon")
    if coast is not None and not coast.empty:
        lines.append("## Coastal-thermal question")
        lines.append("")
        keyed = {str(r.coast_window): r for r in coast.itertuples()}
        near = keyed.get("near_coast_afternoon")
        off = keyed.get("offshore_afternoon")
        if near is not None and off is not None:
            delta = float(near.residual_median_kt) - float(off.residual_median_kt)
            lines.append(
                f"Near-coast afternoon median residual **{near.residual_median_kt:+.2f} kt** "
                f"(n={near.n}) vs offshore afternoon **{off.residual_median_kt:+.2f} kt** "
                f"(n={off.n}). Difference (near − offshore) = **{delta:+.2f} kt**."
            )
            if delta > 0.5 and near.residual_median_kt > 0.5:
                lines.append(
                    "Interpretation: fleet systematically faster than analysis allows "
                    "near the coast in the afternoon — consistent with analysis "
                    "under-representing coastal thermal enhancement."
                )
            elif near.residual_median_kt > 0.5:
                lines.append(
                    "Interpretation: positive near-coast afternoon residual, but not "
                    "clearly larger than offshore afternoon — thermal signal mixed with "
                    "other biases (polar, current, pointing)."
                )
            else:
                lines.append(
                    "Interpretation: no clear positive near-coast afternoon surplus "
                    "in this sample."
                )
        lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Mixed fleet; generic 50–60 ft / fabricated 52 ft polar when boat polar unknown."
    )
    lines.append("- SOG includes current; polar is through-water.")
    lines.append("- Track COG over 15–30 min is made-good, not instantaneous heading.")
    lines.append("- Analysis sampled bilinearly; coastal gradients may still be under-resolved.")
    lines.append("")
    return "\n".join(lines)
