"""Vector metrics on u10/v10 components (m/s)."""

from __future__ import annotations

from typing import Any

import numpy as np

from contracts.schemas import angular_difference, uv_to_tws_twd


def vector_rmse_ms(obs_u: Any, obs_v: Any, model_u: Any, model_v: Any) -> float:
    du = np.asarray(model_u, dtype=float) - np.asarray(obs_u, dtype=float)
    dv = np.asarray(model_v, dtype=float) - np.asarray(obs_v, dtype=float)
    if du.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(du * du + dv * dv)))


def speed_bias_ms(obs_u: Any, obs_v: Any, model_u: Any, model_v: Any) -> float:
    """Signed speed bias: model − observation (m/s)."""
    obs_spd = np.hypot(np.asarray(obs_u, dtype=float), np.asarray(obs_v, dtype=float))
    model_spd = np.hypot(np.asarray(model_u, dtype=float), np.asarray(model_v, dtype=float))
    if obs_spd.size == 0:
        return float("nan")
    return float(np.mean(model_spd - obs_spd))


def direction_mae_deg(
    obs_u: Any,
    obs_v: Any,
    model_u: Any,
    model_v: Any,
    *,
    min_obs_speed_ms: float = 3.0,
) -> float:
    """Circular direction MAE in degrees; only where observed speed ≥ threshold."""
    obs_u_a = np.asarray(obs_u, dtype=float)
    obs_v_a = np.asarray(obs_v, dtype=float)
    model_u_a = np.asarray(model_u, dtype=float)
    model_v_a = np.asarray(model_v, dtype=float)
    obs_spd = np.hypot(obs_u_a, obs_v_a)
    mask = obs_spd >= min_obs_speed_ms
    if not np.any(mask):
        return float("nan")
    _, obs_dir = uv_to_tws_twd(obs_u_a[mask], obs_v_a[mask])
    _, model_dir = uv_to_tws_twd(model_u_a[mask], model_v_a[mask])
    return float(np.mean(np.abs(angular_difference(model_dir, obs_dir))))


def bootstrap_ci(
    obs_u: Any,
    obs_v: Any,
    model_u: Any,
    model_v: Any,
    *,
    statistic: str = "vec_rmse",
    n_boot: int = 500,
    seed: int = 20260816,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for vector RMSE or speed bias."""
    obs_u_a = np.asarray(obs_u, dtype=float)
    obs_v_a = np.asarray(obs_v, dtype=float)
    model_u_a = np.asarray(model_u, dtype=float)
    model_v_a = np.asarray(model_v, dtype=float)
    n = obs_u_a.size
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if statistic == "vec_rmse":
            values[i] = vector_rmse_ms(
                obs_u_a[idx], obs_v_a[idx], model_u_a[idx], model_v_a[idx]
            )
        elif statistic == "speed_bias":
            values[i] = speed_bias_ms(
                obs_u_a[idx], obs_v_a[idx], model_u_a[idx], model_v_a[idx]
            )
        else:
            raise ValueError(f"Unknown statistic {statistic!r}")
    lo = float(np.nanpercentile(values, 100.0 * (alpha / 2.0)))
    hi = float(np.nanpercentile(values, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)
