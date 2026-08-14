"""Generate deterministic FIXTURE-ONLY artifacts for contract-first development."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numcodecs import Blosc

from contracts.schemas import (
    MS_TO_KT,
    Polar,
    directional_constancy,
    load_course,
    load_domain,
    load_routes,
    tws_twd_to_uv,
    uv_to_tws_twd,
    validate_climatology,
    validate_dashboard_payload,
    validate_wind_store,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "contracts" / "fixtures"
FIXED_SEED = 20260814


def _grid(min_v: float, max_v: float, step: float) -> np.ndarray:
    count = int(round((max_v - min_v) / step)) + 1
    return np.round(np.linspace(min_v, max_v, count), 6)


def _generate_wind_fixture() -> xr.Dataset:
    domain = load_domain(ROOT / "config" / "domain.yaml")
    lat = _grid(domain.lat_min, domain.lat_max, domain.fixture_resolution).astype(
        np.float32
    )
    lon = _grid(domain.lon_min, domain.lon_max, domain.fixture_resolution).astype(
        np.float32
    )
    time = pd.date_range("2025-08-01T00:00:00Z", periods=domain.fixture_days * 24, freq="h")

    # FIXTURE ONLY: synthetic signal with plausible large-scale flow + sea breeze.
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    dist_nm = np.minimum.reduce(
        [
            np.abs(xx - 8.9) * 60.0,
            np.abs(xx - 9.8) * 60.0,
            np.abs(xx - 13.7) * 60.0,
            np.abs(yy - 43.6) * 60.0,
            np.abs(yy - 38.0) * 60.0,
        ]
    )
    coastal_decay = np.exp(-dist_nm / 28.0)

    land_mask = (
        ((yy >= 38.7) & (yy <= 41.3) & (xx >= 8.0) & (xx <= 9.8))
        | ((yy >= 41.2) & (yy <= 43.2) & (xx >= 8.3) & (xx <= 9.7))
        | ((yy <= 38.5) & (xx >= 12.0))
        | ((yy >= 43.5) & (xx >= 8.2))
    )

    base_tws_kt = 11.0
    base_twd_deg = 305.0
    base_u, base_v = tws_twd_to_uv(base_tws_kt, base_twd_deg)

    rng = np.random.default_rng(FIXED_SEED)
    ntime = len(time)
    u10 = np.empty((ntime, lat.size, lon.size), dtype=np.float32)
    v10 = np.empty_like(u10)

    for idx in range(ntime):
        hod = idx % 24
        diurnal = np.sin(2.0 * np.pi * (hod - 7.0) / 24.0)
        sea_breeze_kt = 5.8 * coastal_decay * diurnal
        # Coastal flow rotates slightly over the day to mimic veer/back.
        rot = 0.15 * np.cos(2.0 * np.pi * (hod - 12.0) / 24.0)
        u_sig = base_u + sea_breeze_kt / MS_TO_KT
        v_sig = base_v + rot * sea_breeze_kt / MS_TO_KT
        noise_u = rng.normal(0.0, 0.22, size=u_sig.shape)
        noise_v = rng.normal(0.0, 0.22, size=v_sig.shape)
        u_frame = u_sig + noise_u
        v_frame = v_sig + noise_v
        u_frame[land_mask] = np.nan
        v_frame[land_mask] = np.nan
        u10[idx] = u_frame.astype(np.float32)
        v10[idx] = v_frame.astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            "u10": (("time", "lat", "lon"), u10),
            "v10": (("time", "lat", "lon"), v10),
        },
        coords={"time": time.values.astype("datetime64[ns]"), "lat": lat, "lon": lon},
        attrs={
            "source": "ifs_analysis_9km",
            "fetched_utc": "2026-08-14T00:00:00Z",
            "api_version": "fixture-v1",
            "omissions": [],
        },
    )
    validate_wind_store(ds)
    return ds


def _generate_polar_fixture() -> Polar:
    tws = np.array([2, 4, 6, 8, 10, 12, 14, 16, 20, 25], dtype=float)
    twa = np.array([0, 30, 40, 50, 60, 75, 90, 110, 130, 150, 170, 180], dtype=float)

    # FIXTURE ONLY: fabricated 52-footer polar (do not present as real performance).
    twa_term = np.sin(np.radians(np.clip(twa, 0, 180)))
    tws_term = 0.58 * np.power(tws, 0.82)
    bsp = np.outer(twa_term, tws_term)
    bsp *= np.where(twa[:, None] < 35, 0.35, 1.0)  # poor very-high mode
    bsp *= np.where(twa[:, None] > 155, 0.88, 1.0)
    bsp = np.clip(bsp, 0.0, tws[None, :] * 1.2)
    polar = Polar(
        tws_kt=tws.astype(float),
        twa_deg=twa.astype(float),
        bsp_kt=bsp.astype(float),
        name="fixture_52ft_fabricated",
        source_file="contracts/fixtures/polar_52ft.pol",
    )
    polar.validate()
    return polar


def _climatology_from_wind(wind: xr.Dataset) -> xr.Dataset:
    tws_kt, _ = uv_to_tws_twd(wind["u10"].values, wind["v10"].values)
    tws = xr.DataArray(
        tws_kt, coords=wind["u10"].coords, dims=wind["u10"].dims, name="tws_kt"
    )

    grouped = wind.groupby("time.hour")
    tws_grouped = tws.groupby("time.hour")
    n_samples = grouped["u10"].count(dim="time")
    mean_tws = tws_grouped.mean(dim="time", skipna=True)
    mean_u = grouped["u10"].mean(dim="time", skipna=True)
    mean_v = grouped["v10"].mean(dim="time", skipna=True)
    p_below_5 = (tws_grouped < 5.0).mean(dim="time", skipna=True)
    p_below_8 = (tws_grouped < 8.0).mean(dim="time", skipna=True)
    p_above_20 = (tws_grouped > 20.0).mean(dim="time", skipna=True)

    const = xr.apply_ufunc(
        directional_constancy,
        grouped["u10"],
        grouped["v10"],
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[]],
        vectorize=True,
    )
    const = const.transpose("hour", "lat", "lon")

    climo = xr.Dataset(
        data_vars={
            "mean_tws_kt": mean_tws.astype(np.float32),
            "vector_mean_u": mean_u.astype(np.float32),
            "vector_mean_v": mean_v.astype(np.float32),
            "p_below_5kt": p_below_5.astype(np.float32),
            "p_below_8kt": p_below_8.astype(np.float32),
            "p_above_20kt": p_above_20.astype(np.float32),
            "directional_const": const.astype(np.float32),
            "n_samples": n_samples.astype(np.int32),
        },
        coords={"hour": mean_tws["hour"], "lat": wind["lat"], "lon": wind["lon"]},
        attrs={
            "source": "fixture_from_wind_small",
            "years_used": [2025],
            "months_used": [8],
        },
    )

    low_samples = climo["n_samples"] < 200
    for name in (
        "mean_tws_kt",
        "vector_mean_u",
        "vector_mean_v",
        "p_below_5kt",
        "p_below_8kt",
        "p_above_20kt",
        "directional_const",
    ):
        climo[name] = climo[name].where(~low_samples)
    validate_climatology(climo)
    return climo


def _payload_from_climatology(climo: xr.Dataset) -> dict[str, object]:
    course = load_course(ROOT / "config" / "course.yaml")
    routes = load_routes(ROOT / "config" / "routes.yaml")
    rng = np.random.default_rng(FIXED_SEED + 1)

    route_items: list[dict[str, object]] = []
    for route in routes:
        samples = np.round(rng.normal(loc=70.0, scale=8.0, size=30), 2).tolist()
        p10, p50, p90 = np.percentile(samples, [10, 50, 90]).round(2).tolist()
        route_items.append(
            {
                "id": route.id,
                "label": route.label,
                "legs": [[round(a, 3), round(b, 3)] for a, b in route.legs],
                "elapsed_hours": {
                    "p10": float(p10),
                    "p50": float(p50),
                    "p90": float(p90),
                    "samples": samples,
                },
                "stall_rate": float(np.round(rng.uniform(0.0, 0.18), 2)),
            }
        )

    pair_rows: list[dict[str, object]] = []
    ids = [r.id for r in routes]
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            margins = rng.normal(loc=0.5, scale=2.2, size=30)
            pair_rows.append(
                {
                    "a": a,
                    "b": b,
                    "a_wins_pct": float(np.round((margins > 0.0).mean() * 100.0, 2)),
                    "median_margin_hours": float(np.round(np.median(margins), 2)),
                    "p10_margin_hours": float(np.round(np.percentile(margins, 10), 2)),
                    "p90_margin_hours": float(np.round(np.percentile(margins, 90), 2)),
                    "n": int(margins.size),
                }
            )

    skill = [
        {
            "model": "ecmwf_ifs",
            "lead_days": 3,
            "wind_bin": "0-6kt",
            "vec_rmse_kt": 2.34,
            "speed_bias_kt": 0.48,
            "dir_mae_deg": 28.12,
            "reference_biased": True,
        },
        {
            "model": "aifs",
            "lead_days": 3,
            "wind_bin": "0-6kt",
            "vec_rmse_kt": 2.21,
            "speed_bias_kt": 0.40,
            "dir_mae_deg": 27.50,
            "reference_biased": True,
        },
        {
            "model": "gfs_global",
            "lead_days": 3,
            "wind_bin": "0-6kt",
            "vec_rmse_kt": 2.98,
            "speed_bias_kt": 0.64,
            "dir_mae_deg": 35.40,
            "reference_biased": False,
        },
    ]

    by_hour = []
    for hr in climo["hour"].values.astype(int).tolist():
        by_hour.append(
            {
                "hour": int(hr),
                "p_below_5kt": np.round(
                    climo["p_below_5kt"].sel(hour=hr).fillna(np.nan).values, 2
                ).tolist(),
                "mean_tws_kt": np.round(
                    climo["mean_tws_kt"].sel(hour=hr).fillna(np.nan).values, 2
                ).tolist(),
            }
        )

    payload: dict[str, object] = {
        "meta": {
            "generated_utc": "2026-08-14T00:00:00Z",
            "display_timezone": course.display_timezone,
            "course": {
                "start": [course.start[0], course.start[1]],
                "gate": [course.gate[0], course.gate[1]],
                "finish": [course.finish[0], course.finish[1]],
            },
            "polar_name": "fixture_52ft_fabricated",
            "polar_is_validated": False,
            "warnings": [
                "FIXTURE ONLY: synthetic payload for parallel agent development.",
                "Polar below 8 kt is unvalidated; treat elapsed times as relative.",
            ],
        },
        "climatology": {
            "grid": {
                "lat": np.round(climo["lat"].values, 3).tolist(),
                "lon": np.round(climo["lon"].values, 3).tolist(),
            },
            "by_hour": by_hour,
        },
        "routes": route_items,
        "head_to_head": pair_rows,
        "skill": skill,
    }
    validate_dashboard_payload(payload)
    return payload


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    wind = _generate_wind_fixture()
    wind_path = FIXTURES_DIR / "wind_small.zarr"
    if wind_path.exists():
        import shutil

        shutil.rmtree(wind_path)
    encoding = {
        "u10": {"compressor": Blosc(cname="zstd", clevel=5, shuffle=2)},
        "v10": {"compressor": Blosc(cname="zstd", clevel=5, shuffle=2)},
    }
    wind.to_zarr(wind_path, mode="w", consolidated=True, encoding=encoding)

    polar = _generate_polar_fixture()
    polar_path = FIXTURES_DIR / "polar_52ft.pol"
    with polar_path.open("w", encoding="utf-8") as f:
        f.write("TWA\t" + "\t".join(str(int(v)) for v in polar.tws_kt) + "\n")
        for i, twa in enumerate(polar.twa_deg):
            row = [f"{twa:.0f}"] + [f"{v:.2f}" for v in polar.bsp_kt[i]]
            f.write("\t".join(row) + "\n")

    climo = _climatology_from_wind(wind)
    climo_path = FIXTURES_DIR / "climatology_small.nc"
    climo.to_netcdf(climo_path)

    payload = _payload_from_climatology(climo)
    data_json_path = FIXTURES_DIR / "data.json"
    with data_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")

    readme_path = FIXTURES_DIR / "README.md"
    readme_path.write_text(
        "# Contract fixtures (committed on purpose)\n\n"
        "These files are intentionally committed so every module can develop in\n"
        "parallel against frozen interfaces.\n\n"
        "- `wind_small.zarr`: synthetic 30-day hourly wind cube (FIXTURE ONLY).\n"
        "- `polar_52ft.pol`: fabricated Expedition-style polar (FIXTURE ONLY).\n"
        "- `climatology_small.nc`: derived from the synthetic wind cube.\n"
        "- `data.json`: C7 dashboard payload fixture.\n\n"
        "Do not delete this directory. Removing it breaks parallel development\n"
        "for the other agents.\n",
        encoding="utf-8",
    )

    print(f"Wrote {wind_path}")
    print(f"Wrote {polar_path}")
    print(f"Wrote {climo_path}")
    print(f"Wrote {data_json_path}")
    print(f"Wrote {readme_path}")


if __name__ == "__main__":
    main()
