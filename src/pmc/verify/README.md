# pmc.verify — live observation verification (Current weather)

## What it does

Scores currently-running Open-Meteo forecasts against live observations for the
**Current weather** dashboard tab. Three observation classes are scored
**separately and never pooled**:

| Class | Role |
|---|---|
| Scatterometer (ASCAT / HSCAT / RFSCAT) | Offshore vector skill → Expedition calibration |
| Land stations (METAR/SYNOP) | MSLP + thermal/gradient timing only |
| Sentinel-1 | Opportunistic 1 km speed display — never scored |

## How to run

```bash
# Parse an operator file (auto-detect format; dry-run prints a report only)
PYTHONPATH=src:. python3 -m pmc.verify ingest path/to/file.nc --dry-run

# Run a pass (obs files + forecast table)
PYTHONPATH=src:. python3 -m pmc.verify pass \
  --pass-id pass-2026-08-10T12:00:00Z \
  --obs contracts/fixtures/verify/synthetic_ascat.json \
  --forecasts /tmp/forecasts.parquet \
  --store data/verify

# Emit dashboard section
PYTHONPATH=src:. python3 -m pmc.verify emit-payload \
  --store data/verify \
  --output /tmp/current_weather.json
```

Rebuild fixtures:

```bash
PYTHONPATH=src:. python3 contracts/make_verify_fixtures.py
```

## Conventions

- Wind stored as `u10`/`v10` in **m/s**. Knots only in the dashboard display layer.
- Direction convention is **per-instrument** (`contracts.schemas.INSTRUMENT_DIR_CONVENTION`).
  Ingest refuses unknown instruments rather than guessing.
- Persistence is append-only and idempotent on
  `(source_file_hash, cell_id, model, run_init, lead_bucket)`.
- Expedition calibration uses **scatterometer 48–72 h headline cells only**,
  stratified by instrument. Land stations and Sentinel-1 cannot contribute.

## What it gets wrong

- Land-distance uses a sparse coastline proxy, not full GSHHG geometry — coastal
  bucket boundaries are approximate.
- BUFR/GRIB ingest requires `eccodes`/`cfgrib` and still expects an explicit
  `instrument` attribute; convert to NetCDF when unsure.
- Equivalent-neutral correction is a constant 0.2 m/s scale and defaults **off**.
- Sentinel-1 queries need a Copernicus client; without one the tab shows
  `no_acquisition` (clean no-op, not an error).
- 0–12 h scores for IFS/ICON/AROME/ARPEGE measure assimilation circularity, not
  independent skill — the UI marks them contaminated.
