# C9 verify fixtures

Synthetic live-verification artifacts for contract-first development.

| Path | Purpose |
|---|---|
| `instruments/*_mistral.nc` | Per-instrument NetCDF with known mistral cell |
| `instruments/*.ref.json` | Independently computed `u10`/`v10` reference |
| `synthetic_ascat.json` | 40-cell ASCAT pass with +1.5 m/s / +8° model offset |
| `synthetic_hscat_lown.json` | 10-cell HSCAT pass (n&lt;30 → not rankable) |
| `synthetic_land.json` | Land-station MSLP/timing rows (never calibration) |
| `current_weather.json` | Full C9 dashboard section |
| `EXPECTED_OFFSET.json` | Known bias/twist for acceptance tests |
| `store/*.parquet` | Idempotent append-only store from the synthetic pass |

Rebuild with:

```bash
PYTHONPATH=src:. python3 contracts/make_verify_fixtures.py
```
