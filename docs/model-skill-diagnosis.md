# Model skill diagnosis

Numbers only. No fixes applied.

## 1. Lead time is not varying — confirmed

`previous_day=N` as a query param does **not** change the returned series on the Previous Runs API. Leads 1 and 7 are bit-identical.

Gate `41.13N, 9.55E`, valid time `2024-08-15T12:00`, model `icon_global`, columns we actually pair:

| side | lead | u | v | units |
|---|---|---|---|---|
| REF analysis `ecmwf_ifs` | — | −4.7 | 5.0 | km/h |
| PRED `icon_global` | 1 | −10.1 | 4.0 | km/h |
| PRED `icon_global` | 7 | −10.1 | 4.0 | km/h |

`lead1 == lead7`: **True**

Shipped `icon_global` rows are identical at every lead: `vec_rmse=9.56`, `n=5408` for `0-6kt` at leads 1–7 (same for all other bins).

The API’s lead-varying fields are the `*_previous_dayN` series (e.g. `wind_speed_10m_previous_day1` = 12.6 km/h vs `wind_speed_10m` = 10.8 at that same hour). We did not use those.

---

## 2. `ecmwf_ifs` = 0.00 — confirmed, self-comparison

### Columns we used

| role | endpoint | model | columns |
|---|---|---|---|
| reference | `archive-api .../v1/archive` | `ecmwf_ifs` | `wind_u_component_10m`, `wind_v_component_10m` |
| forecast | `previous-runs-api .../v1/forecast` + `previous_day=N` | `ecmwf_ifs` | **same** `wind_u_component_10m`, `wind_v_component_10m` |

We did **not** use `wind_speed_10m_previous_dayN` / `wind_u_component_10m_previous_dayN`.

At the same gate/time: analysis u/v = (−4.7, 5.0); `ecmwf_ifs` + `previous_day=1` u/v = (−4.7, 5.0). **Identical.** Shipped `ecmwf_ifs` `vec_rmse_kt` unique values: **`[0.0]`**.

---

## 3. Wind bins inverted — confirmed; unit bug

API default unit for our skill fetch: **km/h** (no `wind_speed_unit` set).  
`model_skill()` then does `hypot(u,v) * 1.9438445` as if values were m/s.

Observed speeds used for binning (2024, 11 corridor points, analysis, n=96624):

| treatment | mean | median | p90 |
|---|---|---|---|
| raw `hypot(u,v)` (km/h) | 16.64 | 13.45 | 34.39 |
| **current code** (×1.9438445) | **32.34 kt** | **26.14 kt** | **66.86 kt** |
| correct km/h→kt (÷1.852) | 8.98 | 7.26 | 18.57 |

Median **26.14 > 12** → bug confirmed. Inflation factor ≈ **3.6** (km/h treated as m/s), not a double m/s→kt (that would be ×1.94²).

Bin counts under current code: `0-6kt=5408`, `6-12=14037`, `12-20=17219`, `20kt+=59960`.  
If km/h→kt: `0-6=39880`, `6-12=30638`, `12-20=18652`, `20+=7454`.

---

## 4. Climatology contamination — not contaminated

Production IO path sets `wind_speed_unit: "ms"` (`openmeteo.py`). Climatology does one `hypot(u,v) * 1.9438445` from m/s.

Dashboard `mean_tws_kt` (all hour×cell): mean **7.04**, median **7.53**, p90 **9.69**.  
`p_below_5kt` domain mean **0.411**.

Sanity check, archive at gate 2024 with `wind_speed_unit=ms`: median **7.85 kt**, fraction &lt;5 kt **0.309**.

Calm-risk map path looks fine. Skill path is the broken one.

---

## 5. Models + window

| model | in script list? | in shipped skill? | why |
|---|---|---|---|
| `arpege_europe` | yes | yes | |
| `gfs_global` | yes | yes | |
| `icon_global` | yes | yes | |
| `ecmwf_ifs` / `ecmwf_aifs025` | yes | yes (biased) | |
| `gem_global` | yes | **no** | our `previous_day`+u/v path returned all-null (`units=undefined`); `wind_speed_10m_previous_dayN` **does** return data for GEM |
| `ukmo_global` | **not requested** | no | not in `DEFAULT_MODELS`; direct probe → HTTP 400 |
| `ukmo_seamless` | no | no | `previous_dayN` all null |

Verification window **2024-01-01 → 2024-12-31** is intentional (script args / SPEC common Previous Runs window from Jan 2024). Not a bug; only one year.

---

## Root cause summary

1. Wrong Previous Runs lead mechanism (`previous_day` query vs `*_previous_dayN` columns).
2. Missing `wind_speed_unit=ms` on the skill fetch.

Awaiting go-ahead before fixing.
