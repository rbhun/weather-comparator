# Model skill — post-fix light-air table

Verification window: 2024-01-01 → 2024-12-31.  
Grid: 00/06/12/18 UTC. Unit: `m/s` asserted.  
Forecast columns: `wind_speed_10m_previous_dayN`, `wind_direction_10m_previous_dayN`.

Independent models only, wind bin `0-6kt`, leads 2 and 3:

| model | lead | vec_rmse_kt | speed_bias_kt | dir_mae_deg | n |
|---|---:|---:|---:|---:|---:|
| gem_global | 2 | 4.22 | 0.86 | 43.82 | 6440 |
| gfs_global | 2 | 4.24 | 1.03 | 44.25 | 6440 |
| icon_global | 2 | 3.58 | 0.37 | 41.12 | 6440 |
| gem_global | 3 | 4.70 | 1.10 | 46.27 | 6436 |
| gfs_global | 3 | 4.64 | 1.18 | 47.12 | 6436 |
| icon_global | 3 | 4.22 | 0.61 | 45.18 | 6436 |

## Sanity gates (passed)

- Observed kt mean/median/p90 = 8.72 / 7.02 / 18.02
- `0-6kt` n=6445 > `20kt+` n=1104 (GFS lead 1)
- RMSE increases with lead for all models (ICON max lead=5; no day-7 column)
- `ecmwf_ifs025` lead-1 RMSE ≈ 4.3 kt (non-zero)

## Notes

- `ukmo` dropped (HTTP 400)
- `arpege_europe` omitted (no `previous_dayN` coverage on Previous Runs API)
- ECMWF forecast skill uses `ecmwf_ifs025` ( `ecmwf_ifs` previous_dayN empty historically); still tagged `reference_biased`
