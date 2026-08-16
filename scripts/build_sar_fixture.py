"""Build the committed synthetic C9 SAR fixture (FIXTURE ONLY)."""

from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from contracts.schemas import validate_sar_store  # noqa: E402
from pmc.sar.geometry import distance_to_coast_nm  # noqa: E402

OUT = ROOT / "contracts" / "fixtures" / "sar_scenes_small.zarr"
SEED = 20260816


def build_sar_fixture(
    *,
    n_scenes: int = 24,
    lee_kt: float = -3.5,
    control_bias_kt: float = 0.0,
    include_incomplete: bool = True,
) -> xr.Dataset:
    """Synthetic August scenes with a known Sardinian lee and null control.

    Distances use the same coastline geometry as the analysis module so the
    fixture exercises the real band assignment path.
    """
    rng = np.random.default_rng(SEED)

    lats = np.round(np.arange(39.3, 41.05, 0.05), 4).astype("float32")
    lons = np.round(np.arange(9.5, 13.15, 0.05), 4).astype("float32")
    lon2d, lat2d = np.meshgrid(lons, lats)

    print("Computing distance-to-coast on fixture grid (one-time)...")
    dist_sard = distance_to_coast_nm(lat2d, lon2d)
    dist_ctrl = (lon2d - 11.8) * 60.0 * np.cos(np.radians(lat2d))

    years = list(range(2018, 2026))
    times = []
    for i in range(n_scenes):
        year = years[i % len(years)]
        day = 2 + (i * 3) % 28
        hour = 4 if (i % 2 == 0) else 16
        times.append(np.datetime64(f"{year}-08-{day:02d}T{hour:02d}:00:00"))
    times = np.array(times, dtype="datetime64[ns]")

    n_lat, n_lon = lat2d.shape
    speed = np.full((n_scenes, n_lat, n_lon), np.nan, dtype="float32")
    incidence = np.full((n_scenes, n_lat, n_lon), np.nan, dtype="float32")
    quality = np.ones((n_scenes, n_lat, n_lon), dtype="int8")

    sard_bbox = (
        (lat2d >= 39.2)
        & (lat2d <= 41.15)
        & (lon2d >= 9.4)
        & (lon2d <= 10.6)
    )
    ctrl_mask = (
        (lat2d >= 39.5)
        & (lat2d <= 41.0)
        & (lon2d >= 11.8)
        & (lon2d <= 13.2)
        & (dist_ctrl >= 0.5)
        & (dist_ctrl < 20.0)
    )

    for s in range(n_scenes):
        base_kt = float(rng.uniform(6.0, 14.0))
        inc = 30.0 + 15.0 * (lon2d - lons.min()) / max(1e-6, (lons.max() - lons.min()))
        incidence[s] = inc.astype("float32")

        field_kt = np.full(lat2d.shape, np.nan, dtype=float)

        # Sardinian corridor water
        inshore = sard_bbox & (dist_sard >= 0.5) & (dist_sard < 5.0)
        offshore = sard_bbox & (dist_sard >= 7.5) & (dist_sard < 10.0)
        mid = sard_bbox & (dist_sard >= 5.0) & (dist_sard < 7.5)
        far = sard_bbox & (dist_sard >= 10.0) & (dist_sard < 20.0)
        sea_sard = inshore | offshore | mid | far

        field_kt = np.where(offshore, base_kt, field_kt)
        field_kt = np.where(inshore, base_kt + lee_kt, field_kt)
        field_kt = np.where(mid, base_kt + 0.4 * lee_kt, field_kt)
        field_kt = np.where(far, base_kt + 0.1 * lee_kt, field_kt)

        field_kt = np.where(
            ctrl_mask,
            base_kt + control_bias_kt + rng.normal(0.0, 0.15, size=lat2d.shape),
            field_kt,
        )

        if include_incomplete and s in {0, 7}:
            field_kt = np.where(inshore, np.nan, field_kt)

        noise = rng.normal(0.0, 0.25, size=lat2d.shape)
        field_kt = field_kt + np.where(np.isfinite(field_kt), noise, 0.0)

        speed[s] = (field_kt / 1.9438445).astype("float32")
        quality[s] = np.where(np.isfinite(speed[s]), np.int8(0), np.int8(1))
        if s % 5 == 1:
            quality[s, n_lat // 2, n_lon // 3] = np.int8(2)
            speed[s, n_lat // 2, n_lon // 3] = np.float32("nan")

        _ = sea_sard  # documented intent: only band water is filled

    ds = xr.Dataset(
        data_vars={
            "wind_speed_ms": (("scene", "lat", "lon"), speed),
            "incidence_deg": (("scene", "lat", "lon"), incidence),
            "quality_flag": (("scene", "lat", "lon"), quality),
        },
        coords={
            "scene": np.arange(n_scenes, dtype="int32"),
            "time": ("scene", times),
            "lat": lats,
            "lon": lons,
        },
        attrs={
            "source": "sentinel1_l3_cmems",
            "product_id": "fixture_synthetic_s1_l3",
            "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "direction_policy": "speed_only_no_direction",
            "units_note": "wind_speed_ms is SI; analysis layer converts to knots once",
            "fixture_note": (
                f"SYNTHETIC ONLY. Injected Sardinian lee={lee_kt} kt; "
                f"control_bias={control_bias_kt} kt."
            ),
        },
    )
    validate_sar_store(ds)
    return ds


def main() -> int:
    ds = build_sar_fixture()
    if OUT.exists():
        shutil.rmtree(OUT)
    ds.to_zarr(OUT, mode="w", consolidated=True)
    print(f"Wrote {OUT} scenes={ds.sizes['scene']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
