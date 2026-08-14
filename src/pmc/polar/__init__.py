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


def _looks_like_expedition_pairs(rows: list[list[str]]) -> bool:
    first = rows[0][0].lower()
    if first.startswith("!expedition"):
        return True
    for row in rows:
        if not _is_number(row[0]):
            return False
        if (len(row) - 1) < 4 or (len(row) - 1) % 2 != 0:
            return False
        if not all(_is_number(tok) for tok in row[1:]):
            return False
    return True


def _parse_expedition_pairs(rows: list[list[str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_rows = rows[1:] if rows and rows[0][0].lower().startswith("!expedition") else rows
    if len(data_rows) < 2:
        raise ValueError("Expedition polar must have at least two TWS rows.")

    tws_vals: list[float] = []
    row_twas: list[np.ndarray] = []
    row_bsps: list[np.ndarray] = []
    twa_union: set[float] = set()

    for row in data_rows:
        tws = float(row[0])
        pairs = np.array([float(tok) for tok in row[1:]], dtype=float)
        if pairs.size < 4 or pairs.size % 2 != 0:
            raise ValueError("Invalid Expedition polar row: expected TWA/BSP pairs.")
        twas = pairs[0::2]
        bsps = pairs[1::2]
        order = np.argsort(twas)
        twas = twas[order]
        bsps = bsps[order]
        if np.any(np.diff(twas) <= 0):
            raise ValueError("TWA samples per TWS row must be strictly ascending.")
        tws_vals.append(tws)
        row_twas.append(twas)
        row_bsps.append(bsps)
        twa_union.update(float(x) for x in twas)

    tws_arr = np.array(tws_vals, dtype=float)
    if np.any(np.diff(tws_arr) <= 0):
        raise ValueError("TWS rows must be strictly ascending.")

    twa_arr = np.array(sorted(twa_union), dtype=float)
    if twa_arr.size < 2:
        raise ValueError("Expedition polar must define at least two TWA bins.")

    bsp_cols: list[np.ndarray] = []
    for twas, bsps in zip(row_twas, row_bsps):
        interp = np.interp(twa_arr, twas, bsps, left=bsps[0], right=bsps[-1])
        bsp_cols.append(interp)
    bsp = np.stack(bsp_cols, axis=1)
    return tws_arr, twa_arr, bsp


def load_polar(path: Path) -> Polar:
    """Load Expedition-style .pol or plain CSV by content inspection."""
    rows = _normalise_rows(path)

    if _looks_like_expedition_pairs(rows):
        tws, twa, bsp = _parse_expedition_pairs(rows)
    else:
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
