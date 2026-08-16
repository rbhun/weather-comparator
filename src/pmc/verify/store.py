"""Append-only idempotent verification store."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from contracts.schemas import (
    MODELS_ASSIMILATING_SCATTEROMETER,
    MODELS_NON_ASSIMILATING_SCATTEROMETER,
    VERIFY_LAND_DISTANCE_KM_DEFAULT,
    VERIFY_LIGHT_AIR_MS,
    VERIFY_MIN_RANK_N,
)


DEDUPE_COLS = (
    "source_file_hash",
    "cell_id",
    "model",
    "run_init",
    "lead_bucket",
)


@dataclass(frozen=True)
class VerifyConfig:
    land_distance_km: float = VERIFY_LAND_DISTANCE_KM_DEFAULT
    light_air_ms: float = VERIFY_LIGHT_AIR_MS
    thin_grid_deg: float = 0.25
    max_points_per_pass: int = 400
    time_tolerance_minutes: int = 30
    equivalent_neutral_correction: bool = False
    equivalent_neutral_delta_ms: float = 0.2
    min_rank_n: int = VERIFY_MIN_RANK_N
    default_lead_bucket: str = "48-72"
    bootstrap_samples: int = 500
    bootstrap_seed: int = 20260816
    models: tuple[str, ...] = (
        "ecmwf_ifs",
        "ecmwf_aifs025",
        "icon_global",
        "icon_eu",
        "gfs_global",
        "arome_france",
        "arpege_europe",
    )
    regions: dict[str, dict[str, list[float]]] | None = None
    store_dir: Path = Path("data/verify")

    @property
    def assimilating(self) -> tuple[str, ...]:
        return MODELS_ASSIMILATING_SCATTEROMETER

    @property
    def non_assimilating(self) -> tuple[str, ...]:
        return MODELS_NON_ASSIMILATING_SCATTEROMETER


def load_verify_config(path: Path, *, store_dir: Path | None = None) -> VerifyConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    regions = raw.get("regions") or {}
    return VerifyConfig(
        land_distance_km=float(raw.get("land_distance_km", VERIFY_LAND_DISTANCE_KM_DEFAULT)),
        light_air_ms=float(raw.get("light_air_ms", VERIFY_LIGHT_AIR_MS)),
        thin_grid_deg=float(raw.get("thin_grid_deg", 0.25)),
        max_points_per_pass=int(raw.get("max_points_per_pass", 400)),
        time_tolerance_minutes=int(raw.get("time_tolerance_minutes", 30)),
        equivalent_neutral_correction=bool(raw.get("equivalent_neutral_correction", False)),
        equivalent_neutral_delta_ms=float(raw.get("equivalent_neutral_delta_ms", 0.2)),
        min_rank_n=int(raw.get("min_rank_n", VERIFY_MIN_RANK_N)),
        default_lead_bucket=str(raw.get("default_lead_bucket", "48-72")),
        bootstrap_samples=int(raw.get("bootstrap_samples", 500)),
        bootstrap_seed=int(raw.get("bootstrap_seed", 20260816)),
        models=tuple(raw.get("models") or VerifyConfig.models),
        regions=regions,
        store_dir=store_dir or Path("data/verify"),
    )


def cell_id_for(lat: float, lon: float, obs_time: pd.Timestamp, instrument: str) -> str:
    ts = pd.Timestamp(obs_time)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    key = f"{lat:.5f}|{lon:.5f}|{ts.isoformat()}|{instrument}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


class VerifyStore:
    """Append-only parquet store with idempotent inserts."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.collocated_path = self.root / "collocated.parquet"
        self.scores_path = self.root / "scores.parquet"

    def load_collocated(self) -> pd.DataFrame:
        if not self.collocated_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.collocated_path)

    def load_scores(self) -> pd.DataFrame:
        if not self.scores_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.scores_path)

    def append_collocated(self, rows: pd.DataFrame) -> int:
        """Append rows; return count of newly inserted rows (0 if all duplicates)."""
        if rows is None or len(rows) == 0:
            return 0
        frame = rows.copy()
        existing = self.load_collocated()
        if existing.empty:
            self._write_collocated(frame)
            return int(len(frame))
        merged = pd.concat([existing, frame], ignore_index=True)
        before = len(existing)
        merged = merged.drop_duplicates(subset=list(DEDUPE_COLS), keep="first")
        self._write_collocated(merged)
        return int(len(merged) - before)

    def replace_scores(self, scores: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if scores is None or len(scores) == 0:
            if self.scores_path.exists():
                # Keep file; empty replace writes empty frame.
                pd.DataFrame().to_parquet(self.scores_path, index=False)
            return
        scores.to_parquet(self.scores_path, index=False)

    def _write_collocated(self, frame: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(self.collocated_path, index=False)


def empty_collocated_frame() -> pd.DataFrame:
    cols = [
        "pass_id",
        "obs_class",
        "instrument",
        "source_file_hash",
        "cell_id",
        "model",
        "run_init",
        "valid_time",
        "lead_hours",
        "lat",
        "lon",
        "obs_u10",
        "obs_v10",
        "model_u10",
        "model_v10",
        "lead_bucket",
        "speed_bucket",
        "region",
        "bucket_label",
        "land_dist_km",
    ]
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
