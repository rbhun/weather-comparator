#!/usr/bin/env python3
"""Write gitignored secrets onto disk for local / cloud builds.

Reads:
  OPENMETEO_API_KEY  -> .env (if missing)
  BOAT_POLAR         -> config/polar/boat.pol (if missing)

Does not overwrite existing files. Safe to run on every agent boot.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLAR_PATH = ROOT / "config/polar/boat.pol"


def _write_if_missing(path: Path, content: str, label: str) -> None:
    if path.exists():
        print(f"keep existing {label}: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
    path.chmod(0o600)
    print(f"wrote {label}: {path}")


def main() -> int:
    api_key = (os.getenv("OPENMETEO_API_KEY") or "").strip()
    if api_key:
        _write_if_missing(ROOT / ".env", f"OPENMETEO_API_KEY={api_key}\n", "API key")
    else:
        print("OPENMETEO_API_KEY not set; skip .env")

    polar = (os.getenv("BOAT_POLAR") or "").strip()
    if polar:
        _write_if_missing(POLAR_PATH, polar, "boat polar")
    else:
        print("BOAT_POLAR not set; skip polar file")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
