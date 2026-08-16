# PMC-2026 — Palermo–Montecarlo Weather Analysis

**Every agent working on this repo receives this file in full.** Read it before
writing any code. If something here conflicts with your task brief, this file
wins — raise the conflict rather than resolving it yourself.

---

## 1. What we are building and why

A 52-foot boat is racing Palermo–Montecarlo (~500 nm, start Golfo di Mondello,
mandatory gate off Porto Cervo, finish Monaco). Expected elapsed time 2.5–3.5
days. The maxis will finish in under 40 hours, so **our boat experiences a
different weather sequence than the leaders and cannot copy their strategy.**

We are building a decision-support system that answers questions the live
routing does not:

1. **Where and when does this course go calm?** Routing optimises expected time
   and hides tail risk. We want the distribution, not the mean.
2. **What does the diurnal cycle actually do?** Coarse global models damp the
   sea breeze badly. Climatology can restore what the GRIB removed.
3. **Which strategic option wins, historically, and by how much?** Especially
   the gate-exit decision: Strait of Bonifacio vs east of Corsica vs west of
   Corsica.
4. **How much do I trust today's forecast?** Model skill by lead time and by
   wind regime, over this specific water.

The output is a **single offline HTML dashboard** read by a tactician who is
tired, on a boat, possibly without internet. Clarity beats sophistication
everywhere.

### Hard deadline

The race starts on or about **18 August 2026**. Anything not working by
17 August is worthless. Prefer a crude thing that runs to an elegant thing that
doesn't.

---

## 2. Course geometry

| Point | Approx position | Notes |
|---|---|---|
| Start | 38.20 N, 13.32 E | Golfo di Mondello, Palermo |
| Gate | 41.13 N, 9.55 E | Off Porto Cervo, NE Sardinia — **mandatory** |
| Finish | 43.73 N, 7.42 E | Monaco |

**VERIFY THESE AGAINST THE SAILING INSTRUCTIONS BEFORE THE RACE.** They are
best-effort estimates and the gate position in particular must be confirmed.
Put the confirmed values in `config/course.yaml`; nothing may hardcode them.

### Strategic structure

- **Leg 1, Mondello → gate (~250 nm).** Frequently upwind. Sub-decisions:
  which side of Ustica; how far to stand off the Sardinian east coast; whether
  the gate must be approached from the east (costly extra distance).
- **Leg 2, gate → Corsica.** The main fork: Strait of Bonifacio, or leave
  Corsica to the east, or to the west.
- **Leg 3, Ligurian approach to Monaco.** Historically the light-air lottery.

The 5–15 nm coastal decisions matter as much as the big fork for a boat our
size, because we are in that water longer than the leaders. Do not build
anything that can only reason about the big fork.

---

## 3. Analysis domain

```
lat:  37.5 .. 44.0 N
lon:   6.5 .. 14.5 E
resolution: 0.1 deg  (~9 km, matching the ECMWF IFS analysis)
```

Roughly 65 × 80 = 5,200 nodes, about half sea. This is a *field*, not a set of
waypoints. Any analysis that only works at a handful of points is wrong.

---

## 4. Data sources

Open-Meteo, paid subscription. API key from env var `OPENMETEO_API_KEY`.
Never commit it. Use the `customer-*` endpoints when the key is present.

| Purpose | Endpoint | Coverage |
|---|---|---|
| **Climatology / truth** | Historical Weather API | ECMWF IFS analysis 9 km from 2017; ERA5 0.25° from 1940 |
| **Forecast skill** | Previous Runs API | from Jan 2024 (GFS from Mar 2021), lead times 1–7 d |
| **Live forecast** | Forecast API | all models incl. AIFS |

### Choose the right one

- **Climatology and calm probability → IFS 9 km analysis (2017–2025).** Nine
  Augusts. Do NOT use 0.25° ERA5 for diurnal work: at 25 km the sea breeze is
  under-resolved and you will conclude it does not exist.
- **Deep climate baselines / SST anomaly → ERA5**, where the long record
  matters more than resolution.
- **Model skill → Previous Runs.** Common window across all models is Jan 2024
  onward, which contains only two Augusts. Therefore stratify skill by *wind
  regime*, not by calendar month — light, weak-gradient, coastal conditions
  occur year-round and give us adequate sample size.

### Known bias, must be stated in any skill output

If ECMWF IFS analysis is the reference, then IFS and AIFS are being scored
against their own analysis, and AIFS was trained on ECMWF reanalysis. Their
scores are **not comparable** to GFS/ICON/ARPEGE/GEM/UKMO. Every skill table
and chart must carry this warning inline. Do not bury it in a footnote.

---

## 5. Conventions — these are not negotiable

### Time

**Storage and computation: UTC, always. Display: local, always.**

The split is deliberate. Mixing zones inside the pipeline produces off-by-two-
hour bugs that look like real meteorological findings — a sea breeze that
appears to fill at 0800 when it actually fills at 1000. Those are very hard to
catch. But UTC on screen at 3am is a different kind of error, made by a tired
human, and that one is on us to prevent.

- On disk and in every calculation: `numpy.datetime64[ns]`, UTC, ISO 8601 in
  JSON with an explicit `Z`.
- Display timezone comes from `config/course.yaml`:
  ```yaml
  display_timezone: "Europe/Rome"   # CEST, UTC+2 in August
  ```
  Both Palermo and Monaco are in UTC+2 during the race, so one zone covers the
  whole course. Do not hardcode +2 — use `zoneinfo`, so the code survives a
  different race or a DST boundary.
- **Every timestamp rendered to a human is local and carries its label**:
  `21 Aug 14:30 CEST`. No bare times. No UTC on screen except in a small
  "also UTC" secondary line where a tactician might be cross-checking a GRIB.
- The dashboard has a **global UTC/local toggle** in a fixed position, with the
  current mode always visible. Default is local. It must switch every timestamp
  on the page at once, including axis labels.
- Variable naming: any variable holding a local-time value must be suffixed
  `_local`. Anything unsuffixed is UTC. A function that takes or returns local
  time says so in its name. This one convention prevents most of the damage.

### Hour-of-day in the climatology

The diurnal cycle is driven by the sun, not by the clock, so the `hour` axis in
the climatology grids (C6) is **computed in UTC and displayed in local**. In
August the offset is a constant +2 across the entire race window, so this is a
pure relabelling with no analytical consequence.

One caveat to note in the module README, not to engineer around: the domain
spans 6.5°E to 14.5°E, which is about 32 minutes of solar time. The sea breeze
at Monaco is on a genuinely later solar clock than at Palermo. That is smaller
than the hourly resolution and can be ignored, but do not be surprised by a
systematic west-to-east phase drift in the transects — it is real.

### Wind
- **Store as `u10`, `v10` in m/s.** Never store direction on disk. This kills
  the entire class of from/to convention bugs.
- Derived on read:
  ```python
  tws_kt = np.hypot(u, v) * 1.9438445
  twd_deg = (np.degrees(np.arctan2(-u, -v))) % 360   # meteorological, FROM
  ```
- Meteorological convention throughout: **direction is where the wind comes
  from**. 0° = from the north.
- **Exception — Sentinel-1 SAR (C9):** store scalar `wind_speed_ms` only. Do not
  fabricate `u10`/`v10` from the model prior used in dual-pol inversion;
  SAR direction is not independent evidence for NWP falsification.

### Units
- **Storage layer: SI** (m/s, metres, degrees).
- **Everything above the storage layer: knots and nautical miles.**
- Conversion happens exactly once, in the loader. If you find yourself
  converting units in analysis or routing code, something is wrong.
- Angles in degrees, not radians, at every interface. Radians only inside a
  function.

### Coordinates
- Always `(lat, lon)` order, in that order, everywhere. Degrees, north and east
  positive.
- Array dims always `(time, lat, lon)`.
- Latitude ascending, longitude ascending.

### Missing data
- **Never silently fill.** NaN is NaN. If an analysis needs complete data it
  must assert completeness and fail loudly.
- Land nodes are NaN in wind fields, not zero.

---

## 6. Repository layout

```
pmc2026/
  config/
    course.yaml          # start, gate, finish, race window
    domain.yaml          # grid bounds, resolution
    models.yaml          # Open-Meteo model ids, verified live
  contracts/             # FROZEN. See CONTRACTS.md. Changes need agreement.
    schemas.py           # dataclasses + validators
    fixtures/            # synthetic data, committed, small
  src/pmc/
    io/                  # A1: Open-Meteo client, zarr cache
    geo/                 # A2: coastline, gate, distance, bearing
    polar/               # A2: polar loader, interpolation
    follow/              # A3: fixed-track route follower
    route/               # A4: isochrone optimiser
    stats/               # A5: climatology, calm probability, skill
    report/              # A6: JSON emit
  dashboard/             # A6: single-file HTML
  tests/
  scripts/
    walking_skeleton.py  # end-to-end on fixtures. MUST always pass.
```

---

## 7. Technical choices

- Python 3.11+. `numpy`, `pandas`, `xarray`, `zarr`, `requests`, `shapely`,
  `pyyaml`, `pytest`. Nothing else without asking.
- No async, no framework, no ORM, no cloud. This runs on one laptop.
- Dashboard: **one self-contained HTML file** plus one JSON. Plotly from a
  vendored local copy, not a CDN — it must work with no internet.
- Caching is mandatory. Every API response goes to disk keyed by request
  parameters. Re-running an analysis must not re-download.

---

## 8. Definition of done — applies to every task

A task is done when all of these are true:

1. `pytest` passes.
2. It runs end-to-end on `contracts/fixtures/` with no network access.
3. Its outputs validate against the schema in `contracts/schemas.py`.
4. It has a CLI entry point with `--help`.
5. `scripts/walking_skeleton.py` still passes.
6. The README section for your module says what it does, how to run it, and
   **what it gets wrong**.

Point 6 is not optional. An honest limitations note is worth more to us than a
feature.

---

## 9. Working rules for agents

- **Do not modify anything in `contracts/`.** If you believe a contract is
  wrong, stop and report it. Changing it breaks every other agent silently.
- **Do not modify another agent's module.** Report the bug instead.
- Commit small and often. Never leave the walking skeleton broken.
- If you are blocked waiting for another module, use the fixtures. That is
  what they are for. Blocking is never the right answer.
- If a task is underspecified, choose the simplest defensible option, implement
  it, and write down the assumption in the module README. Do not stall on
  clarification.
- **Do not invent meteorology.** If you need a physical assumption (wind shear
  exponent, sea-breeze scaling), write it as a named, documented constant in
  one place so it can be challenged.

---

## 10. What we already know is hard

Flagging these so nobody rediscovers them at 4am:

- **AIFS is 6-hourly**, other models hourly. Any multi-model comparison must
  resample to a common 00/06/12/18 UTC grid or AIFS will look artificially bad.
- **Models give 10 m wind; the masthead sees more** — typically 5–10% higher,
  and more under the stable nocturnal stratification common in the summer
  Tyrrhenian. Apply shear correction in one documented place. It does not
  affect model *ranking*; it does affect absolute boat speed prediction.
- **Our polar is not validated in light air.** We have one day with the boat and
  it is not enough. Below ~8 kt TWS the polar is extrapolated and probably
  optimistic. Every elapsed-time output must be treated as a *relative*
  comparison, never an absolute prediction. Say so on the dashboard.
- **Coastline resolution matters.** At 0.1° a naive land mask will let a route
  cut a corner through Capo Testa. Use real coastline geometry.
