"""A2 — Polar module with content-based loader for .pol and CSV data."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from contracts.schemas import Polar


def _tokenize(line: str) -> list[str]:
    return [tok for tok in re.split(r"[,\t ]+", line.strip()) if tok]


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _normalise_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append(_tokenize(stripped))
    if len(rows) < 3:
        raise ValueError(f"Polar file {path} does not contain enough data rows.")
    return rows


def _extract_tws(header: list[str]) -> np.ndarray:
    if len(header) < 3:
        raise ValueError("Header must contain TWS bins.")
    # .pol files commonly start with TWA/TWS label; CSVs might be fully numeric.
    tws_tokens = header[1:] if not _is_number(header[0]) else header
    tws = np.array([float(x) for x in tws_tokens], dtype=float)
    if tws.ndim != 1 or tws.size < 2 or not np.all(np.diff(tws) > 0):
        raise ValueError("TWS bins must be strictly ascending.")
    return tws


def load_polar(path: Path) -> Polar:
    """Load Expedition-style .pol or plain CSV by content inspection."""
    rows = _normalise_rows(path)
    tws = _extract_tws(rows[0])

    twa_vals: list[float] = []
    bsp_rows: list[list[float]] = []
    for row in rows[1:]:
        if len(row) < tws.size + 1:
            raise ValueError("Polar row has fewer columns than TWS header.")
        twa_vals.append(float(row[0]))
        bsp_rows.append([float(x) for x in row[1 : 1 + tws.size]])

    twa = np.array(twa_vals, dtype=float)
    if twa.ndim != 1 or twa.size < 2 or not np.all(np.diff(twa) > 0):
        raise ValueError("TWA bins must be strictly ascending.")
    bsp = np.array(bsp_rows, dtype=float)
    if np.any(~np.isfinite(bsp)):
        raise ValueError("Polar speeds must be finite numeric values.")
    name = path.stem
    return Polar(
        tws_kt=tws,
        twa_deg=twa,
        bsp_kt=bsp,
        name=name,
        source_file=str(path),
    )
