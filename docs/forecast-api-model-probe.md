# Forecast API model probe — race domain

Probe of Open-Meteo **Forecast API** (`api.open-meteo.com/v1/forecast`), not Previous Runs, for Palermo–Montecarlo domain points:

| Label | Lat | Lon |
|---|---|---|
| Sardinian east coast | 40.5°N | 9.7°E |
| Ligurian | 43.4°N | 7.8°E |
| Mid Tyrrhenian | 39.5°N | 11.5°E |

Date of probe: 2026-08-15.

## Verdict

On the Forecast API, all three race points get live wind from ICON-2i, AROME HD, and ICON-EU. Ensemble IDs such as `ecmwf_ifs_ens` / `gfs_ensemble` are **not** Forecast models — they live on the **Ensemble API** under different names. `previous_dayN` with real data is **Previous Runs only**; Forecast accepts the column names but returns nulls.

---

## Priority models (Forecast API)

| Model ID | Res (docs) | Domain | 40.5N 9.7E | 43.4N 7.8E | 39.5N 11.5E | Horizon (approx) |
|---|---|---|---|---|---|---|
| `italia_meteo_arpae_icon_2i` | ~2 km (0.02°) | Southern Europe | **yes** | **yes** | **yes** | ~3 d (85/192 h when asking 8 d) |
| `meteofrance_arome_france_hd` | ~1.5 km (0.01°) | France + neighbours | **yes** | **yes** | **yes** | ~2–2.5 d |
| `icon_eu` | ~7 km | Europe | **yes** | **yes** | **yes** | ≥5 d (120/120) |

AROME HD is not France-only in practice: Corsica, Sardinia E, Liguria, and mid-Tyrrhenian all return non-null, distinct from ARPEGE. Malta (`35.9N 14.5E`) is empty — domain stops before there. Liguria matches non-HD AROME hour-for-hour; Sardinia/Tyrrhenian differ slightly from non-HD.

---

## Ensembles (not on Forecast API)

Those IDs return HTTP 400 on Forecast. Use `ensemble-api.open-meteo.com/v1/ensemble`:

| Requested name | Working Ensemble API ID | Members @ domain | All 3 points |
|---|---|---|---|
| `ecmwf_ifs_ens` | `ecmwf_ifs025` or `ecmwf_ifs025_ensemble` | 51 | yes |
| `ecmwf_aifs025_ens` | `ecmwf_aifs025` or `ecmwf_aifs025_ensemble` | 51 | yes |
| `gfs_ensemble` | `gfs025` or `gfs_seamless` | 31 | yes |

Also available over our domain:

- `icon_eu` / `icon_eu_eps` (40 members)
- `icon_global` / `icon_global_eps` (40)
- `gem_global` (21)
- `ukmo_global_ensemble_20km` (18)

Not for us: `icon_d2`, UK 2 km, BOM ACCESS (empty).

---

## Skill: `previous_dayN` vs forecast-only

| Model | Previous Runs `previous_dayN` with data | Notes |
|---|---|---|
| `italia_meteo_arpae_icon_2i` | **1–2** | Horizon-limited |
| `meteofrance_arome_france_hd` | **1 only** | Short AROME horizon |
| `meteofrance_arome_france` | **1** | |
| `icon_eu` | **1–4** | |
| `icon_global` | **1–6** | |
| `ecmwf_ifs025`, `gfs_global`, `gem_global` | **1–7** | Full skill leads |
| Ensemble member models | **no** usable `previous_dayN` | Members returned; lead columns empty |
| Forecast API `*_previous_dayN` | keys present, **all null** | Do not use for skill |

Forecast `past_days` only gives a short recent archive (seamless timeseries), not fixed lead offsets.

---

## Other Forecast models that hit all 3 points

- `icon_global`
- `ecmwf_ifs025`
- `gfs_global`
- `gem_global`
- `meteofrance_arpege_europe` / `meteofrance_arpege_world`
- `ukmo_global_deterministic_10km`
- `cma_grapes_global`
- `jma_gsm`
- `knmi_harmonie_arome_europe`
- `dmi_harmonie_arome_europe`
- `icon_seamless`
- `best_match`

**Out of domain / fail:** `icon_d2`, `gfs_hrrr`, `jma_msm`, `ukmo_uk_deterministic_2km`, `metno_nordic`.

**Anomaly:** `ecmwf_aifs025` returned empty on Forecast and Historical Forecast on probe day (0 valid hours); Ensemble API still had AIFS members. Treat AIFS deterministic as currently unavailable until that recovers.

---

## Practical takeaway for PMC

- **Live / race-time high-res:** ICON-2i (~2 km) + AROME HD (~1.5 km) + ICON-EU (~7 km) all cover the course.
- **Skill scoring high-res:** possible but shallow — ICON-2i to day 2, AROME HD to day 1 only; ICON-EU to day 4. Days 5–7 stay on globals (`ecmwf_ifs025`, GFS, GEM, ICON global).
- **Ensembles:** separate endpoint; rename IDs; no `previous_dayN` skill path like the deterministic Previous Runs pipeline.
