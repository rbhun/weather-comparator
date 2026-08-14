"""A2 — Geometry and polar module (real loader)."""

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


def load_polar(path: Path) -> Polar:
    """Load Expedition-style .pol or plain CSV by content inspection."""
    rows: list[list[str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append(_tokenize(stripped))
    if len(rows) < 3:
        raise ValueError(f"Polar file {path} does not contain enough data rows.")

    header = rows[0]
    if len(header) < 3:
        raise ValueError("Header must contain TWS bins.")
    if _is_number(header[0]):
        tws_tokens = header
    else:
        tws_tokens = header[1:]
    tws = np.array([float(x) for x in tws_tokens], dtype=float)

    twa_vals: list[float] = []
    bsp_rows: list[list[float]] = []
    for row in rows[1:]:
        if len(row) < tws.size + 1:
            raise ValueError("Polar row has fewer columns than TWS header.")
        twa_vals.append(float(row[0]))
        bsp_rows.append([float(x) for x in row[1 : 1 + tws.size]])

    twa = np.array(twa_vals, dtype=float)
    bsp = np.array(bsp_rows, dtype=float)
    name = path.stem
    return Polar(
        tws_kt=tws,
        twa_deg=twa,
        bsp_kt=bsp,
        name=name,
        source_file=str(path),
    )
