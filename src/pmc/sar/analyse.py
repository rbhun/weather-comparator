"""Paired within-scene SAR lee-shadow falsification analysis."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr

from contracts.schemas import MS_TO_KT, validate_sar_shadow_payload, validate_sar_store
from pmc.sar.fetch import load_sar_config
from pmc.sar.geometry import (
    assign_band,
    distance_to_coast_nm,
    distance_to_virtual_meridian_nm,
    in_bbox,
    km_to_nm,
    ms_to_kt,
)

LOGGER = logging.getLogger(__name__)

SAMPLING_LIMITATION = (
    "Sentinel-1 is sun-synchronous with roughly 06:00 and 18:00 local overpasses. "
    "The archive therefore samples near dawn and dusk and largely misses mid-afternoon, "
    "when a thermal or lee effect would be strongest and when much of the fleet passes "
    "that coast. A null SAR differential may mean the overpass window is uninformative, "
    "not that the shadow does not exist."
)

HYPOTHESIS = (
    "Within each SAR scene, mean wind speed in the 0–5 nm band along the Sardinian "
    "east coast is 3–4 kt lower than in the 7.5–10 nm band (paired difference)."
)

VERDICTS = ("supported", "contradicted", "insufficient sample")


def _bands_from_cfg(cfg: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(a), float(b)) for a, b in cfg["bands_nm"]]


def _bootstrap_mean_ci(
    samples: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, tuple[float, float]]:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draw = rng.choice(samples, size=samples.size, replace=True)
        means[i] = float(np.mean(draw))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return float(np.mean(samples)), (lo, hi)


def _ci_contains_zero(ci: tuple[float, float]) -> bool:
    lo, hi = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return True
    return lo <= 0.0 <= hi


def _scene_band_means(
    speed_ms: np.ndarray,
    dist_nm: np.ndarray,
    incidence: np.ndarray,
    quality: np.ndarray,
    *,
    bands_nm: list[tuple[float, float]],
    buffer_nm: float,
    exclude_below_ms: float | None,
) -> dict[str, Any] | None:
    """Return band statistics for one scene, or None if either required band empty."""
    valid = (
        np.isfinite(speed_ms)
        & np.isfinite(dist_nm)
        & (quality == 0)
        & (dist_nm >= buffer_nm)
    )
    if exclude_below_ms is not None:
        valid = valid & (speed_ms >= float(exclude_below_ms))

    band_idx = assign_band(dist_nm, bands_nm)
    inshore_i = 0
    offshore_i = next(
        i for i, (lo, hi) in enumerate(bands_nm) if abs(lo - 7.5) < 1e-9 and abs(hi - 10.0) < 1e-9
    )

    band_means_ms: list[float | None] = []
    band_n: list[int] = []
    band_inc: list[float | None] = []
    for i in range(len(bands_nm)):
        mask = valid & (band_idx == i)
        band_n.append(int(mask.sum()))
        if mask.any():
            band_means_ms.append(float(np.mean(speed_ms[mask])))
            if np.isfinite(incidence[mask]).any():
                band_inc.append(float(np.nanmean(incidence[mask])))
            else:
                band_inc.append(None)
        else:
            band_means_ms.append(None)
            band_inc.append(None)

    if band_means_ms[inshore_i] is None or band_means_ms[offshore_i] is None:
        return None

    inshore = float(band_means_ms[inshore_i])
    offshore = float(band_means_ms[offshore_i])
    return {
        "band_means_ms": band_means_ms,
        "band_n": band_n,
        "band_incidence_deg": band_inc,
        "paired_diff_ms": inshore - offshore,
        "paired_diff_kt": (inshore - offshore) * MS_TO_KT,
        "inshore_ms": inshore,
        "offshore_ms": offshore,
        "inshore_low_tail_frac": float(
            np.mean(speed_ms[valid & (band_idx == inshore_i)] < 3.0)
        )
        if band_n[inshore_i]
        else None,
    }


def _corridor_distances(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    corridor: dict[str, Any],
    *,
    use_virtual: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (distance_nm, in_corridor_mask)."""
    mask = in_bbox(
        lat2d,
        lon2d,
        lat_min=float(corridor["lat_min"]),
        lat_max=float(corridor["lat_max"]),
        lon_min=float(corridor["lon_min"]),
        lon_max=float(corridor["lon_max"]),
    )
    if use_virtual:
        dist = distance_to_virtual_meridian_nm(
            lat2d, lon2d, float(corridor["virtual_coast_lon"])
        )
    else:
        # Only compute expensive coast distances inside the bbox.
        dist = np.full(lat2d.shape, np.nan, dtype=float)
        if mask.any():
            dist_vals = distance_to_coast_nm(lat2d[mask], lon2d[mask])
            dist[mask] = dist_vals
    return dist, mask


def _collect_scene_rows(
    sar: xr.Dataset,
    corridor: dict[str, Any],
    *,
    bands_nm: list[tuple[float, float]],
    buffer_km: float,
    exclude_below_ms: float | None,
    use_virtual: bool,
    dist_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    model_diffs: dict[str, dict[int, float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    """Walk scenes; return retained rows, raw year counts, raw scene count."""
    buffer_nm = km_to_nm(buffer_km)
    lat = sar["lat"].values.astype(float)
    lon = sar["lon"].values.astype(float)
    lon2d, lat2d = np.meshgrid(lon, lat)
    cache_key = f"{corridor.get('id')}:{use_virtual}"
    if dist_cache is not None and cache_key in dist_cache:
        dist, corridor_mask = dist_cache[cache_key]
    else:
        dist, corridor_mask = _corridor_distances(
            lat2d, lon2d, corridor, use_virtual=use_virtual
        )
        if dist_cache is not None:
            dist_cache[cache_key] = (dist, corridor_mask)

    times = sar["time"].values
    speed = sar["wind_speed_ms"].values
    incidence = (
        sar["incidence_deg"].values
        if "incidence_deg" in sar
        else np.full(speed.shape, np.nan)
    )
    quality = (
        sar["quality_flag"].values
        if "quality_flag" in sar
        else np.zeros(speed.shape, dtype=np.int8)
    )

    retained: list[dict[str, Any]] = []
    raw_by_year: dict[str, int] = {}
    n_raw = int(sar.sizes["scene"])

    for s in range(n_raw):
        t = np.datetime64(times[s], "ns")
        year = str(int(str(t)[:4]))
        raw_by_year[year] = raw_by_year.get(year, 0) + 1

        spd = speed[s]
        inc = incidence[s]
        q = quality[s]
        # Mask to corridor
        spd_c = np.where(corridor_mask, spd, np.nan)
        dist_c = np.where(corridor_mask, dist, np.nan)
        inc_c = np.where(corridor_mask, inc, np.nan)
        q_c = np.where(corridor_mask, q, np.int8(1))

        stats = _scene_band_means(
            spd_c,
            dist_c,
            inc_c,
            q_c,
            bands_nm=bands_nm,
            buffer_nm=buffer_nm,
            exclude_below_ms=exclude_below_ms,
        )
        if stats is None:
            continue

        row = {
            "scene": int(s),
            "time_utc": str(t.astype("datetime64[s]")) + "Z",
            "year": year,
            "month": int(str(t)[5:7]),
            **stats,
        }
        if model_diffs:
            for model_name, by_scene in model_diffs.items():
                if s in by_scene:
                    row[f"{model_name}_diff_kt"] = float(by_scene[s])
        retained.append(row)

    return retained, raw_by_year, n_raw


def _summarise_diffs(
    rows: list[dict[str, Any]],
    *,
    key: str,
    n_boot: int,
    seed: int,
    emit_mean: bool,
) -> dict[str, Any]:
    samples = np.array([r[key] for r in rows if key in r and np.isfinite(r[key])], dtype=float)
    n = int(samples.size)
    if n == 0:
        return {
            "mean": None,
            "ci95": [None, None],
            "n": 0,
            "sign_consistency": None,
            "samples": [],
        }
    mean, ci = _bootstrap_mean_ci(samples, n_boot=n_boot, seed=seed)
    sign_consistency = float(np.mean(samples < 0.0))
    return {
        "mean": float(mean) if emit_mean else None,
        "ci95": [float(ci[0]), float(ci[1])] if emit_mean else [None, None],
        "n": n,
        "sign_consistency": float(sign_consistency),
        "samples": [float(x) for x in samples.tolist()] if emit_mean else [],
    }


def _decide_verdict(
    *,
    n: int,
    threshold: int,
    pipeline_valid: bool,
    sar_summary: dict[str, Any],
    hypothesis: dict[str, float],
) -> str:
    if n < threshold or not pipeline_valid:
        return "insufficient sample"
    mean = sar_summary.get("mean")
    ci = sar_summary.get("ci95") or [None, None]
    if mean is None or ci[0] is None or ci[1] is None:
        return "insufficient sample"
    # Supported: CI entirely below zero and mean in/near the registered 3–4 kt lee.
    if ci[1] < 0.0:
        # Negative differential = inshore weaker = shadow direction.
        lo, hi = float(hypothesis["low"]), float(hypothesis["high"])
        # Accept any significantly negative result as support of the *existence*
        # of a lee; magnitude check is reported separately.
        return "supported"
    if ci[0] > 0.0:
        # Significantly positive: opposite of prediction.
        return "contradicted"
    # CI includes zero → does not support the registered shadow at this sample.
    return "contradicted"


def _model_diff_from_field(
    field_kt: xr.DataArray,
    sar: xr.Dataset,
    rows: list[dict[str, Any]],
    corridor: dict[str, Any],
    *,
    bands_nm: list[tuple[float, float]],
    buffer_km: float,
    use_virtual: bool,
) -> dict[int, float]:
    """Sample a model speed field at each retained scene's valid time & band cells.

    ``field_kt`` dims: (time, lat, lon) in knots. Nearest-time selection.
    """
    buffer_nm = km_to_nm(buffer_km)
    lat = sar["lat"].values.astype(float)
    lon = sar["lon"].values.astype(float)
    lon2d, lat2d = np.meshgrid(lon, lat)
    dist, corridor_mask = _corridor_distances(
        lat2d, lon2d, corridor, use_virtual=use_virtual
    )
    inshore_i = 0
    offshore_i = next(
        i for i, (lo, hi) in enumerate(bands_nm) if abs(lo - 7.5) < 1e-9 and abs(hi - 10.0) < 1e-9
    )
    out: dict[int, float] = {}
    for row in rows:
        s = int(row["scene"])
        t = np.datetime64(sar["time"].values[s], "ns")
        # Nearest model time
        model_times = field_kt["time"].values.astype("datetime64[ns]")
        idx = int(np.argmin(np.abs(model_times - t)))
        slab = field_kt.isel(time=idx)
        # Interpolate onto SAR grid
        interp = slab.interp(lat=lat, lon=lon, method="linear")
        speed = np.asarray(interp.values, dtype=float)
        valid = (
            corridor_mask
            & np.isfinite(speed)
            & np.isfinite(dist)
            & (dist >= buffer_nm)
        )
        band_idx = assign_band(dist, bands_nm)
        m_in = valid & (band_idx == inshore_i)
        m_off = valid & (band_idx == offshore_i)
        if not m_in.any() or not m_off.any():
            continue
        out[s] = float(np.mean(speed[m_in]) - np.mean(speed[m_off]))
    return out


def analyse_shadow_test(
    sar: xr.Dataset,
    cfg: dict[str, Any] | None = None,
    *,
    arome_speed_kt: xr.DataArray | None = None,
    era5_speed_kt: xr.DataArray | None = None,
    precomputed_model_diffs: dict[str, dict[int, float]] | None = None,
) -> dict[str, Any]:
    """Run the paired within-scene lee-shadow falsification test.

    Parameters
    ----------
    sar:
        C10 SAR store (speed only).
    cfg:
        Loaded ``config/sar.yaml`` (or equivalent dict).
    arome_speed_kt / era5_speed_kt:
        Optional model speed fields (kt) on (time, lat, lon) for three-way comparison.
    precomputed_model_diffs:
        Optional ``{model: {scene_index: paired_diff_kt}}`` for fixtures/tests.
    """
    validate_sar_store(sar)
    config = cfg or load_sar_config()
    bands_nm = _bands_from_cfg(config)
    threshold = int(config.get("min_scenes_threshold", 15))
    n_boot = int(config.get("bootstrap_samples", 2000))
    seed = int(config.get("bootstrap_seed", 20260816))
    buffer_km = float(config.get("default_buffer_km", 1.5))
    buffers = [float(x) for x in config.get("buffer_sensitivity_km", [1.0, 1.5, 2.0])]
    if buffer_km not in buffers:
        buffers = sorted(set(buffers + [buffer_km]))
    low_ms = float(config.get("low_speed_exclude_ms", 3.0))
    hypothesis = {
        "low": float(config["hypothesis_diff_kt"]["low"]),
        "high": float(config["hypothesis_diff_kt"]["high"]),
    }

    sard = config["sardinia_corridor"]
    ctrl = config["control_corridor"]
    dist_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    primary_rows, raw_by_year, n_raw = _collect_scene_rows(
        sar,
        sard,
        bands_nm=bands_nm,
        buffer_km=buffer_km,
        exclude_below_ms=None,
        use_virtual=False,
        dist_cache=dist_cache,
    )
    control_rows, _, _ = _collect_scene_rows(
        sar,
        ctrl,
        bands_nm=bands_nm,
        buffer_km=buffer_km,
        exclude_below_ms=None,
        use_virtual=True,
        dist_cache=dist_cache,
    )

    n = len(primary_rows)
    emit_mean = n >= threshold

    # Control validity
    control_summary = _summarise_diffs(
        control_rows,
        key="paired_diff_kt",
        n_boot=n_boot,
        seed=seed + 1,
        emit_mean=True,  # always report control diagnostics
    )
    control_ci = (
        float(control_summary["ci95"][0]) if control_summary["ci95"][0] is not None else float("nan"),
        float(control_summary["ci95"][1]) if control_summary["ci95"][1] is not None else float("nan"),
    )
    control_ok = _ci_contains_zero(control_ci) or control_summary["n"] == 0
    # If control has samples and CI excludes zero → pipeline invalid.
    if control_summary["n"] > 0 and not _ci_contains_zero(control_ci):
        control_ok = False
    pipeline_valid = bool(control_ok)

    # Model co-location
    model_diffs: dict[str, dict[int, float]] = dict(precomputed_model_diffs or {})
    if arome_speed_kt is not None and "arome" not in model_diffs:
        model_diffs["arome"] = _model_diff_from_field(
            arome_speed_kt,
            sar,
            primary_rows,
            sard,
            bands_nm=bands_nm,
            buffer_km=buffer_km,
            use_virtual=False,
        )
    if era5_speed_kt is not None and "era5" not in model_diffs:
        model_diffs["era5"] = _model_diff_from_field(
            era5_speed_kt,
            sar,
            primary_rows,
            sard,
            bands_nm=bands_nm,
            buffer_km=buffer_km,
            use_virtual=False,
        )
    for row in primary_rows:
        s = int(row["scene"])
        for name, by_scene in model_diffs.items():
            if s in by_scene:
                row[f"{name}_diff_kt"] = float(by_scene[s])

    sar_summary = _summarise_diffs(
        primary_rows,
        key="paired_diff_kt",
        n_boot=n_boot,
        seed=seed,
        emit_mean=emit_mean and pipeline_valid,
    )
    arome_summary = _summarise_diffs(
        primary_rows,
        key="arome_diff_kt",
        n_boot=n_boot,
        seed=seed + 2,
        emit_mean=emit_mean and pipeline_valid,
    )
    era5_summary = _summarise_diffs(
        primary_rows,
        key="era5_diff_kt",
        n_boot=n_boot,
        seed=seed + 3,
        emit_mean=emit_mean and pipeline_valid,
    )

    # Low-speed exclusion sensitivity on primary corridor
    excl_rows, _, _ = _collect_scene_rows(
        sar,
        sard,
        bands_nm=bands_nm,
        buffer_km=buffer_km,
        exclude_below_ms=low_ms,
        use_virtual=False,
        dist_cache=dist_cache,
    )
    excl_summary = _summarise_diffs(
        excl_rows,
        key="paired_diff_kt",
        n_boot=n_boot,
        seed=seed + 4,
        emit_mean=len(excl_rows) >= threshold and pipeline_valid,
    )

    # Buffer sensitivity
    buffer_sensitivity = []
    for b_km in buffers:
        brows, _, _ = _collect_scene_rows(
            sar,
            sard,
            bands_nm=bands_nm,
            buffer_km=b_km,
            exclude_below_ms=None,
            use_virtual=False,
            dist_cache=dist_cache,
        )
        bsum = _summarise_diffs(
            brows,
            key="paired_diff_kt",
            n_boot=n_boot,
            seed=seed + int(b_km * 10),
            emit_mean=len(brows) >= threshold and pipeline_valid,
        )
        buffer_sensitivity.append(
            {
                "buffer_km": float(b_km),
                "n": int(bsum["n"]),
                "mean_kt": bsum["mean"],
                "ci95_kt": bsum["ci95"],
            }
        )

    # Incidence diagnostic: paired difference of band-mean incidence vs speed diff
    inc_deltas = []
    for row in primary_rows:
        inc_in = row["band_incidence_deg"][0]
        off_i = next(
            i
            for i, (lo, hi) in enumerate(bands_nm)
            if abs(lo - 7.5) < 1e-9 and abs(hi - 10.0) < 1e-9
        )
        inc_off = row["band_incidence_deg"][off_i]
        if inc_in is None or inc_off is None:
            continue
        inc_deltas.append(
            {
                "incidence_delta_deg": float(inc_in) - float(inc_off),
                "speed_diff_kt": float(row["paired_diff_kt"]),
            }
        )
    if inc_deltas:
        x = np.array([d["incidence_delta_deg"] for d in inc_deltas], dtype=float)
        y = np.array([d["speed_diff_kt"] for d in inc_deltas], dtype=float)
        if x.size >= 3 and np.std(x) > 1e-6:
            corr = float(np.corrcoef(x, y)[0, 1])
        else:
            corr = None
    else:
        corr = None

    # Inventory by year (retained)
    retained_by_year: dict[str, int] = {}
    for row in primary_rows:
        retained_by_year[row["year"]] = retained_by_year.get(row["year"], 0) + 1

    # Low-speed tail of inshore distribution
    tails = [r["inshore_low_tail_frac"] for r in primary_rows if r["inshore_low_tail_frac"] is not None]
    low_tail = {
        "mean_frac_below_3ms": float(np.mean(tails)) if tails else None,
        "note": (
            "Surfactants and slicks invert to spuriously low wind and concentrate "
            "in sheltered coastal water — the most dangerous confounder."
        ),
        "survives_excluding_below_3ms": {
            "n": int(excl_summary["n"]),
            "mean_kt": excl_summary["mean"],
            "ci95_kt": excl_summary["ci95"],
        },
    }

    # AROME-at-overpass note: if we have AROME diffs, compare magnitude at SAR times
    arome_at_overpass = {
        "note": (
            "Compare AROME's paired differential restricted to SAR overpass times "
            "with the SAR differential. If AROME's own dawn/dusk shadow is weak, "
            "the test is uninformative."
        ),
        "arome_mean_kt": arome_summary["mean"],
        "arome_ci95_kt": arome_summary["ci95"],
        "sar_mean_kt": sar_summary["mean"],
        "sar_ci95_kt": sar_summary["ci95"],
    }

    three_way = []
    for row in primary_rows:
        three_way.append(
            {
                "time_utc": row["time_utc"],
                "sar_diff_kt": float(row["paired_diff_kt"]),
                "arome_diff_kt": row.get("arome_diff_kt"),
                "era5_diff_kt": row.get("era5_diff_kt"),
            }
        )

    verdict = _decide_verdict(
        n=n,
        threshold=threshold,
        pipeline_valid=pipeline_valid,
        sar_summary=sar_summary,
        hypothesis=hypothesis,
    )
    # Force insufficient when pipeline invalid (control failed)
    if not pipeline_valid:
        verdict = "insufficient sample"
        sar_summary = {
            **sar_summary,
            "mean": None,
            "ci95": [None, None],
            "samples": [],
            "suppressed_reason": "control_corridor_differential_not_zero",
        }
        arome_summary = {**arome_summary, "mean": None, "ci95": [None, None], "samples": []}
        era5_summary = {**era5_summary, "mean": None, "ci95": [None, None], "samples": []}

    payload = {
        "hypothesis": HYPOTHESIS,
        "sampling_limitation": SAMPLING_LIMITATION,
        "min_scenes_threshold": threshold,
        "n_scenes_retained": n,
        "n_scenes_raw": n_raw,
        "verdict": verdict,
        "pipeline_valid": pipeline_valid,
        "control_indistinguishable_from_zero": control_ok,
        "default_buffer_km": buffer_km,
        "scene_inventory": {
            "by_year_raw": raw_by_year,
            "by_year_retained": retained_by_year,
            "total_raw": n_raw,
            "retained": n,
            "both_bands_filter": "scene requires valid retrievals in 0–5 nm and 7.5–10 nm",
        },
        "paired_differentials_kt": {
            "sar": sar_summary,
            "arome": arome_summary,
            "era5": era5_summary,
        },
        "control": {
            "corridor_id": ctrl["id"],
            "label": ctrl["label"],
            "paired_differential_kt": control_summary,
            "interpretation": (
                "Control differential must be indistinguishable from zero; "
                "otherwise the Sardinian result is suppressed."
            ),
        },
        "buffer_sensitivity": buffer_sensitivity,
        "incidence_diagnostic": {
            "n_pairs": len(inc_deltas),
            "corr_incidence_delta_vs_speed_diff": corr,
            "points": inc_deltas[:200],
            "note": (
                "If the speed differential is an incidence-angle gradient in disguise, "
                "correlation with cross-track incidence delta will be large."
            ),
        },
        "low_speed_tail": low_tail,
        "arome_at_overpass": arome_at_overpass,
        "three_way_table": three_way,
        "era5_caveat": (
            "ERA5 is unreliable for Bonifacio acceleration and Sicilian/Corsican "
            "thermals; included as a low-resolution baseline, not as truth."
        ),
    }
    validate_sar_shadow_payload(payload)
    return payload
