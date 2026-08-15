"""Resume previous-runs pulls for all discovered models."""

from __future__ import annotations

import datetime as dt
import sys

from pmc.io.cli import main

MODELS = [
    "ecmwf_ifs",
    "ecmwf_aifs025",
    "gfs_global",
    "icon_global",
    "gem_global",
    "arpege_europe",
]


def run() -> int:
    end = dt.date(2026, 8, 15)
    for model in MODELS:
        out = f"data/wind/previous_runs-{model}.zarr"
        print(f"[batch] start model={model} output={out}", flush=True)
        sys.argv = [
            "pmc.io.cli",
            "--source",
            "previous_runs",
            "--start",
            "2024-01-01",
            "--end",
            end.isoformat(),
            "--force-model",
            model,
            "--output-path",
            out,
            "--max-workers",
            "4",
            "--request-jitter-seconds",
            "0.5",
            "--retries",
            "8",
        ]
        rc = main()
        if rc != 0:
            return rc
        print(f"[batch] done model={model}", flush=True)
    print("[batch] all previous_runs models complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
