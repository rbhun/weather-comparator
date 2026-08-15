# Historical Forecast API — high-res coverage probe

Probe of Open-Meteo **Historical Forecast API**
(`historical-forecast-api.open-meteo.com/v1/forecast`) — not Forecast, not
Archive/ERA5 — for high-resolution models over the PMC race domain.

Points:

| Label | Lat | Lon |
|---|---|---|
| Sardinian east coast | 40.5°N | 9.7°E |
| Ligurian | 43.4°N | 7.8°E |
| Mid Tyrrhenian | 39.5°N | 11.5°E |

Probe date: 2026-08-15. August 2026 checked through 2026-08-15 only.

---

## Headline

**Do not replace the IFS 9 km analysis climatology backbone with ICON-2i.**

ICON-2i Historical Forecast starts **2025-04-13**. That is **one complete
August (2025)** plus a partial August 2026 — not 3+ Augusts. It fails the
bar for a multi-year coastal climatology.

ICON-EU (~7 km) and AROME HD (~1.5 km) each have **three complete Augusts
(2023–2025)** at all three race points, plus partial 2026. Useful as
high-res supplements or coastal checks; still much shorter than IFS
analysis Augusts 2017–2025.

---

## Summary table

| Model | Earliest date (API) | Complete Augusts | Partial Augusts | All 3 race points |
|---|---|---|---|---|
| `italia_meteo_arpae_icon_2i` | **2025-04-13** | **2025 only** | 2026 (through 15th) | yes |
| `meteofrance_arome_france_hd` | **2022-11-13** | **2023, 2024, 2025** | 2026 (through 15th) | yes |
| `icon_eu` | **2022-11-23** | **2023, 2024, 2025** | 2026 (through 15th) | yes |

Docs “Available Since” matches the probe for ICON-2i (2025-04-13) and
ICON-EU (docs 2022-11-24; first data day found 2022-11-23). AROME France HD
docs say 2022-11-13; probe confirms that day.

---

## August completeness (hourly wind_speed_10m)

Expected full August = 31 × 24 = **744** hours. Partial Aug 2026 (1–15) =
**360** hours.

### `italia_meteo_arpae_icon_2i` (~2 km)

| Year | Sardinia E | Ligurian | Tyrrhenian | Complete? |
|---|---|---|---|---|
| 2025 | 744/744 | 744/744 | 744/744 | yes |
| 2026 (1–15) | 360/360 | 360/360 | 360/360 | partial (in progress) |

No August 2017–2024. **August count for climatology: 1.**

### `meteofrance_arome_france_hd` (~1.5 km)

| Year | Sardinia E | Ligurian | Tyrrhenian | Complete? |
|---|---|---|---|---|
| 2023 | 744/744 | 744/744 | 744/744 | yes |
| 2024 | 744/744 | 744/744 | 744/744 | yes |
| 2025 | 744/744 | 744/744 | 744/744 | yes |
| 2026 (1–15) | 360/360 | 360/360 | 360/360 | partial |

**August count: 3** (plus current). All three race points return data for
every probed August — including mid-Tyrrhenian and Sardinia E (same caveat
as the live Forecast probe: published domain is France+neighbours; API
still fills these points).

### `icon_eu` (~7 km)

| Year | Sardinia E | Ligurian | Tyrrhenian | Complete? |
|---|---|---|---|---|
| 2023 | 744/744 | 744/744 | 744/744 | yes |
| 2024 | 744/744 | 744/744 | 744/744 | yes |
| 2025 | 744/744 | 744/744 | 744/744 | yes |
| 2026 (1–15) | 360/360 | 360/360 | 360/360 | partial |

**August count: 3** (plus current). Full domain coverage at all three points.

---

## Point coverage (spot checks)

Near each model’s earliest day, mid-August 2025, and 2026-08-10: all three
points return non-null wind for all three models.

---

## Implication for climatology backbone

| Candidate | August depth | Coastal resolution | Verdict |
|---|---|---|---|
| ECMWF IFS analysis 9 km (Archive / analysis mode) | Aug 2017–2025 (~9) | coarse near headlands | **Keep as C6 backbone** |
| ICON-2i Historical Forecast | 1 complete Aug | excellent (~2 km) | **Too short** — live/coastal case studies only |
| AROME HD Historical Forecast | 3 complete Augs | excellent (~1.5 km) where domain holds | Supplement / coastal check, not backbone |
| ICON-EU Historical Forecast | 3 complete Augs | good (~7 km) | Best hi-res multi-August option here; still thin vs IFS |

SPEC C6 asks for Augusts 2017–2025 and n > 200 per cell-hour. ICON-2i
cannot satisfy that. AROME HD / ICON-EU give at most ~3 × 31 ≈ 93
day-samples per UTC hour before multi-year stacking math — far below the
IFS analysis depth.

**Do not build an ICON-2i climatology replacement on this archive.**
