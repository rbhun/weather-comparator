# Model skill — extended window + month facets

Verification: **2024-01-01 → 2026-08-15** (Previous Runs full record).  
Grid: 00/06/12/18 UTC. Unit: `m/s` asserted.  
Paired hours: all=1 216 457 · summer=415 701 · august=101 024.

## Sanity gates

All four passed on the `all` facet (RMSE↑ with lead, ECMWF ≠ 0, 0-6kt n > 20kt+, lead-1 RMSE in range).

## Light-air tables (0-6kt, leads 2–3, independent only)

### All months

| model | lead | vec_rmse_kt | speed_bias_kt | dir_mae_deg | n |
|---|---:|---:|---:|---:|---:|
| gem_global | 2 | 4.22 | 0.88 | 43.49 | 17452 |
| gfs_global | 2 | 4.28 | 1.08 | 45.07 | 17759 |
| icon_global | 2 | 3.61 | 0.41 | 40.74 | 17759 |
| gem_global | 3 | 4.71 | 1.09 | 46.60 | 17625 |
| gfs_global | 3 | 4.65 | 1.21 | 47.90 | 17755 |
| icon_global | 3 | 4.16 | 0.61 | 44.96 | 17755 |

### Summer (Jun–Sep)

| model | lead | vec_rmse_kt | speed_bias_kt | dir_mae_deg | n |
|---|---:|---:|---:|---:|---:|
| gem_global | 2 | 4.01 | 0.69 | 44.21 | 7026 |
| gfs_global | 2 | 4.20 | 0.98 | 47.18 | 7333 |
| icon_global | 2 | 3.51 | 0.30 | 41.76 | 7333 |
| gem_global | 3 | 4.33 | 0.79 | 47.51 | 7203 |
| gfs_global | 3 | 4.49 | 1.06 | 50.16 | 7333 |
| icon_global | 3 | 4.00 | 0.45 | 45.54 | 7333 |

### August only

| model | lead | vec_rmse_kt | speed_bias_kt | dir_mae_deg | n |
|---|---:|---:|---:|---:|---:|
| gem_global | 2 | 3.87 | 0.61 | 41.52 | 1922 |
| gfs_global | 2 | 4.19 | 0.96 | 46.04 | 1922 |
| icon_global | 2 | 3.43 | 0.32 | 38.63 | 1922 |
| gem_global | 3 | 3.99 | 0.65 | 44.28 | 1922 |
| gfs_global | 3 | 4.29 | 0.91 | 48.74 | 1922 |
| icon_global | 3 | 3.73 | 0.40 | 41.82 | 1922 |

## Does ICON still lead?

Yes, in every window at leads 2 and 3.

| window | lead 2 margin vs next | lead 3 margin vs next |
|---|---|---|
| all | +0.61 kt vs GEM | +0.49 kt vs GFS |
| summer | +0.50 kt vs GEM | +0.33 kt vs GEM |
| august | +0.44 kt vs GEM | +0.26 kt vs GEM |

## Speed-bias ordering

**ICON lowest, GFS highest** in every cell above (GEM in between). Holds for all / summer / august at leads 2 and 3.

## Dashboard

Month facets shipped on Model skill tab: All months / Summer (default) / August only.  
Stability line + n prominence + grey-out when n&lt;300.
