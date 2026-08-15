"""Boat LOA resolution for YB fleet filters (45–60 ft cross-check)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from pmc.io.yb import Boat, norm_name

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOA_PATH = REPO_ROOT / "config" / "boat_loa.yaml"

# metres → feet
M_TO_FT = 3.280839895

# Name cues: design / LOA embedded in the yacht name.
_NAME_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"\bBLACK\s*JACK\s*100\b", re.I), 100.0),
    (re.compile(r"\bCASSIOPEIA\s*68\b", re.I), 68.0),
    (re.compile(r"\bCLASS\s*40\b|\bACI\s*40\b", re.I), 40.0),
    (re.compile(r"\bTP\s*52\b|\bTP52\b", re.I), 52.0),
    (re.compile(r"\bICE\s*52\b|\bICE52\b", re.I), 52.0),
    (re.compile(r"\bFARR\s*45\b", re.I), 45.0),
    (re.compile(r"\bFARR\s*40\b", re.I), 40.0),
    (re.compile(r"\bSWAN\s*(\d{2})\b", re.I), -1.0),  # group
    (re.compile(r"\bFIRST\s*(\d{2})\b", re.I), -1.0),
    (re.compile(r"\bMYLIUS\s*(\d{2})\b", re.I), -1.0),
    (re.compile(r"\bDEHLER\s*(\d{2})\b", re.I), -1.0),
    (re.compile(r"\bJ/?(\d{2})\b", re.I), -1.0),
    (re.compile(r"\bX-?(\d{2,3})\b", re.I), -1.0),
    (re.compile(r"\bGS\s*(\d{2})\b|\bGRAND\s*SOLEIL\s*(\d{2})\b", re.I), -1.0),
)


def _load_overrides(path: Path = DEFAULT_LOA_PATH) -> dict[str, float]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    boats = raw.get("boats") or {}
    out: dict[str, float] = {}
    for name, loa in boats.items():
        out[norm_name(str(name))] = float(loa)
    return out


def loa_from_name(name: str) -> float | None:
    text = name or ""
    for pat, fixed in _NAME_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        if fixed > 0:
            return float(fixed)
        # capture last non-None group as LOA feet (design number ≈ LOA)
        for g in m.groups():
            if g is not None:
                return float(g)
    return None


def loa_from_classes(class_names: Iterable[str]) -> tuple[float | None, str | None]:
    """Class-derived LOA. Hard bounds first; ORC group as length proxy."""

    lowered = {c.strip().lower() for c in class_names}
    if "maxi" in lowered:
        return 65.0, "class_maxi"
    if "class 40" in lowered or "class40" in lowered:
        return 40.0, "class_40"
    # Palermo–Montecarlo ORC groups are length-split (Group 1 larger).
    if "orc group 1" in lowered or "orc 1" in lowered:
        return 50.0, "class_orc_group1"
    if "orc group 2" in lowered or "orc 2" in lowered:
        return 36.0, "class_orc_group2"
    return None, None


def resolve_loa_ft(
    boat: Boat,
    *,
    overrides: dict[str, float] | None = None,
) -> tuple[float | None, str]:
    """Return (loa_ft, source)."""

    table = overrides if overrides is not None else _load_overrides()
    key = norm_name(boat.name)
    if key in table:
        return float(table[key]), "override"
    from_name = loa_from_name(boat.name)
    if from_name is not None:
        return from_name, "name"
    from_class, class_src = loa_from_classes(c.class_name for c in boat.classes)
    if from_class is not None and class_src is not None:
        return from_class, class_src
    return None, "unknown"


def attach_loa(
    boats: Iterable[Boat],
    *,
    path: Path = DEFAULT_LOA_PATH,
) -> dict[tuple[int, str], dict[str, Any]]:
    """Map (year, boat_name) → {loa_ft, source}."""

    overrides = _load_overrides(path)
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for boat in boats:
        loa, source = resolve_loa_ft(boat, overrides=overrides)
        out[(boat.year, boat.name)] = {"loa_ft": loa, "loa_source": source}
    return out


def in_loa_range(loa_ft: float | None, lo: float = 45.0, hi: float = 60.0) -> bool:
    if loa_ft is None:
        return False
    return lo <= float(loa_ft) <= hi
