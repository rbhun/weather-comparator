"""Direction convention helpers and instrument registry."""

from __future__ import annotations

from typing import Any

import numpy as np

from contracts.schemas import INSTRUMENT_DIR_CONVENTION, tws_twd_to_uv


def wind_to_uv_ms(
    wind_speed_ms: Any,
    wind_dir_deg: Any,
    *,
    instrument: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert product wind_speed/wind_dir to meteorological u10/v10 (m/s).

    ``INSTRUMENT_DIR_CONVENTION`` is the sole source of truth. Oceanographic
    products report direction *toward*; meteorological report *from*.
    """
    convention = INSTRUMENT_DIR_CONVENTION.get(instrument)
    if convention is None:
        raise ValueError(
            f"Unknown instrument {instrument!r}. Refusing to guess direction convention. "
            f"Known: {sorted(INSTRUMENT_DIR_CONVENTION)}"
        )
    speed = np.asarray(wind_speed_ms, dtype=float)
    direction = np.asarray(wind_dir_deg, dtype=float)
    if convention == "oceanographic":
        # toward → from
        direction = (direction + 180.0) % 360.0
    elif convention != "meteorological":
        raise ValueError(f"Unsupported convention {convention!r} for {instrument}")
    # tws_twd_to_uv expects knots; convert speed ms→kt at the boundary only.
    from contracts.schemas import MS_TO_KT

    return tws_twd_to_uv(speed * MS_TO_KT, direction)


def lead_bucket_for_hours(lead_hours: float) -> str | None:
    """Map lead hours to C9 bucket label, or None if outside 0–72 h."""
    h = float(lead_hours)
    if 0.0 <= h < 12.0:
        return "0-12"
    if 12.0 <= h < 24.0:
        return "12-24"
    if 24.0 <= h < 48.0:
        return "24-48"
    if 48.0 <= h < 72.0:
        return "48-72"
    return None


def speed_bucket_for_obs_ms(obs_speed_ms: float) -> str:
    from contracts.schemas import MS_TO_KT, VERIFY_LIGHT_AIR_MS

    if obs_speed_ms < VERIFY_LIGHT_AIR_MS:
        return "sub_3ms"
    kt = obs_speed_ms * MS_TO_KT
    if kt < 8.0:
        return "3-8kt"
    if kt < 15.0:
        return "8-15kt"
    return "15+kt"


def region_for_point(lat: float, lon: float, regions: dict[str, dict[str, list[float]]]) -> str:
    for name, box in regions.items():
        lat_lo, lat_hi = box["lat"]
        lon_lo, lon_hi = box["lon"]
        if lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi:
            return name
    return "other"
