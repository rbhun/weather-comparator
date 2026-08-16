# CONTRACTS — frozen interfaces

**Committed before any agent starts work. Do not change without agreement from
the project owner.** These exist so that six agents can work simultaneously
without ever talking to each other.

Every contract ships with a synthetic fixture in `contracts/fixtures/` that has
the correct shape and plausible values. **Develop against the fixture.** Never
wait for another agent's real output.

---

## C1 — Wind field store

Zarr, on disk at `data/wind/{source}.zarr`.

```
dims:      time, lat, lon
coords:
  time     datetime64[ns], UTC, ascending, no duplicates
  lat      float32, 37.5 .. 44.0, ascending, step 0.1
  lon      float32,  6.5 .. 14.5, ascending, step 0.1
vars:
  u10      float32 (time, lat, lon)   m/s, eastward
  v10      float32 (time, lat, lon)   m/s, northward
attrs:
  source          "ifs_analysis_9km" | "era5" | "forecast:{model}:{lead}"
  fetched_utc     ISO 8601
  api_version     str
  omissions       list of time ranges known to be missing
```

Land nodes are NaN. Chunking: `(720, 32, 40)`.

Fixture: `contracts/fixtures/wind_small.zarr` — 30 days hourly, full domain,
synthetic but with a real diurnal cycle and a land mask, so downstream code
exercises realistic paths.

---

## C2 — Polar

```python
@dataclass(frozen=True)
class Polar:
    tws_kt: np.ndarray        # (n_tws,)  ascending, e.g. [2,4,6,8,10,12,14,16,20,25]
    twa_deg: np.ndarray       # (n_twa,)  ascending, 0..180
    bsp_kt: np.ndarray        # (n_twa, n_tws)
    name: str
    source_file: str

    def speed(self, twa_deg, tws_kt) -> np.ndarray:
        """Bilinear interpolation. Vectorised. TWA is symmetric about 0:
        the caller may pass -180..180 and it is folded to abs().
        Outside the TWS range: clamp, do NOT extrapolate.
        Returns 0.0 where interpolation gives <= 0."""

    def vmg_optimum(self, tws_kt, upwind: bool) -> tuple[float, float]:
        """Returns (best_twa_deg, bsp_kt) maximising VMG. Cached per tws."""
```

Loader must read **Expedition .pol format** (tab or space delimited, first row
TWS, first column TWA) and plain CSV. Detect by content, not extension.

Fixture: `contracts/fixtures/polar_52ft.pol` — a plausible 52-footer polar.
**It is fabricated.** Never present results from it as real without a banner.

---

## C3 — Candidate route

`config/routes.yaml`, one entry per strategic option.

```yaml
- id: bonifacio_direct
  label: "Bonifacio Strait, direct"
  description: "Through the strait, shortest distance"
  legs:
    - [38.20, 13.32]    # start
    - [38.62, 13.10]    # west of Ustica
    - [40.10, 11.20]
    - [41.13,  9.55]    # GATE - mandatory, must appear in every route
    - [41.32,  9.18]    # Bonifacio Strait
    - [42.60,  8.40]
    - [43.73,  7.42]    # finish
  tags: [strait, direct]
```

Rules:
- Every route must pass within 2 nm of the gate. The loader asserts this.
- Legs are great-circle segments between consecutive points.
- Minimum 8 routes covering: Ustica east/west; Sardinian coast tight/offshore;
  and the three Corsica options. Coastal variants matter as much as the big
  fork — see SPEC §2.

---

## C4 — Route follower result

Parquet, `data/results/follow/{route_id}.parquet`, one row per historical day:

```
start_time        datetime64[ns]  UTC, the simulated start
route_id          str
elapsed_hours     float32         NaN if the run stalled (see below)
distance_nm       float32         actual sailed distance
mean_tws_kt       float32
hours_below_5kt   float32         time spent in < 5 kt
hours_upwind      float32         TWA < 60
stalled           bool            True if boat made < 0.5 kt for > 6 h
max_stall_hours   float32
```

A stalled run is **not** an error. Record it. The tail is the point.

---

## C5 — Optimiser result

Same columns as C4, plus:

```
track             list[[lat, lon, iso_time]]   the achieved path
n_manoeuvres      int32
gate_time         datetime64[ns]
corridor          str    which of the strategic options it chose, classified
                         post hoc by which side of Corsica the track passed
```

---

## C6 — Climatology grids

NetCDF, `data/climatology/{name}.nc`.

```
dims: hour (0..23), lat, lon        # or (month, hour, lat, lon)
vars:
  mean_tws_kt          float32
  vector_mean_u        float32      # for vector-mean direction
  vector_mean_v        float32
  p_below_5kt          float32      # 0..1
  p_below_8kt          float32
  p_above_20kt         float32
  directional_const    float32      # |vector mean| / mean speed, 0..1
  n_samples            int32        # per cell. Assert > 200 before trusting.
attrs:
  source, years_used, months_used
```

`directional_const` is the useful one nobody computes: near 1 means the wind is
reliably from one direction; near 0 means it is a lottery. It separates "light
but steady" from "light and random", which are completely different tactically.

---

## C7 — Dashboard payload

`dashboard/data.json`. Single file, the only thing the HTML reads.

```jsonc
{
  "meta": {
    "generated_utc": "...",
    "display_timezone": "Europe/Rome",  // dashboard renders local by default
    "course": { "start": [lat,lon], "gate": [lat,lon], "finish": [lat,lon] },
    "polar_name": "...",
    "polar_is_validated": false,     // drives a warning banner
    "warnings": ["..."]              // rendered prominently, always visible
  },
  "climatology": {
    "grid": { "lat": [...], "lon": [...] },
    "by_hour": [ { "hour": 0, "p_below_5kt": [[...]], "mean_tws_kt": [[...]] } ]
  },
  "routes": [
    { "id": "...", "label": "...", "legs": [[lat,lon]],
      "elapsed_hours": { "p10": 0, "p50": 0, "p90": 0, "samples": [...] },
      "stall_rate": 0.0 }
  ],
  "head_to_head": [
    { "a": "bonifacio_direct", "b": "east_corsica",
      "a_wins_pct": 62.0, "median_margin_hours": 1.4,
      "p10_margin_hours": -6.2, "n": 214 }
  ],
  "skill": [
    { "model": "gfs_global", "lead_days": 3, "wind_bin": "0-6kt",
      "vec_rmse_kt": 0.0, "speed_bias_kt": 0.0, "dir_mae_deg": 0.0,
      "reference_biased": false }
  ],
  "current_weather": { /* see C9 */ }
}
```

`reference_biased: true` for every ECMWF/AIFS row. The dashboard must render
those rows visually distinct and annotated.

`current_weather` is required (C9). Live verification is distinct from
historical `skill`.

Numbers are rounded to 2 decimals on emit. Keep the file under 20 MB.

---

## C8 — Module entry points

Each agent exposes exactly this, so the skeleton can call it:

```python
# io
def fetch_wind(source: str, start: date, end: date, cfg: Domain) -> Path
# geo
def is_sea(lat, lon) -> np.ndarray
def crosses_land(lat0, lon0, lat1, lon1) -> bool
# polar
def load_polar(path: Path) -> Polar
# follow
def follow(route: Route, wind: xr.Dataset, polar: Polar, start: datetime) -> FollowResult
# route
def optimise(wind, polar, start, course: Course) -> RouteResult
# stats
def climatology(wind: xr.Dataset, months: list[int]) -> xr.Dataset
def head_to_head(results: dict[str, pd.DataFrame]) -> pd.DataFrame
# sar
def analyse_shadow_test(sar: xr.Dataset, cfg: dict | None = None) -> dict
# verify (live observation verification — Current weather tab)
def ingest_observations(path: Path, *, dry_run: bool = False) -> IngestReport
def run_verification_pass(pass_id: str, cfg: VerifyConfig) -> PassSummary
def build_current_weather_payload(store_dir: Path) -> dict
# report
def emit(...) -> Path
```

Signatures are frozen. Add keyword arguments with defaults if you must; never
change or reorder existing ones.

---

## C9 — Live observation verification (Current weather)

Distinct from historical model skill (C7 `skill`). Scores **currently-running**
forecasts against live observations, refreshed ~6-hourly until race start.

### Observation classes — never pooled

| Class | Verifies | Feeds Expedition calibration? |
|---|---|---|
| `scatterometer` | Offshore 10 m wind vector | **Yes** — sole operational input |
| `land_station` | MSLP + thermal/gradient timing | **No** |
| `sentinel1` | Opportunistic 1 km wind-speed snapshot | **No** — display only |

### Persistence (append-only, idempotent)

`data/verify/collocated.parquet` — raw collocated pairs (recompute metrics
without re-fetching). Re-ingesting the same source file must insert zero new
rows (dedupe key below).

```
pass_id           str
obs_class         "scatterometer" | "land_station" | "sentinel1"
instrument        str                 # first-class; never pool across instruments
source_file_hash  str                 # sha256 of raw bytes
cell_id           str                 # stable hash of (lat,lon,obs_time,instrument)
model             str
run_init          datetime64[ns] UTC
valid_time        datetime64[ns] UTC  # per-cell observation time
lead_hours        float32
lat, lon          float32
obs_u10, obs_v10  float32             # m/s eastward/northward
model_u10, model_v10  float32         # m/s
lead_bucket       "0-12" | "12-24" | "24-48" | "48-72"
speed_bucket      "3-8kt" | "8-15kt" | "15+kt" | "sub_3ms"
region            str
bucket_label      "headline" | "coastal" | "light_air" | "qc_reject"
land_dist_km      float32
```

Dedupe key: `(source_file_hash, cell_id, model, run_init, lead_bucket)`.

Aggregate score rows (derived, regenerable) live alongside at
`data/verify/scores.parquet`, keyed by
`(pass_id, instrument, model, run_init, lead_bucket, region, speed_bucket)`.

### Direction convention

Per-instrument, explicit, tested. Store only `u10`/`v10` after conversion.
Meteorological FROM is the internal convention (SPEC §5). Fixtures under
`contracts/fixtures/verify/instruments/` carry an independently computed
reference `(u10, v10)` for one unambiguous cell; ingest must match.

### Metrics (components only)

On `u10`/`v10`, never speed/direction pairs between modules:

- Vector RMSE (primary), speed bias (model − obs), direction MAE (circular,
  only where obs speed ≥ 3 m/s), `n`, bootstrap 95% CI on RMSE and bias.

Display conversion to knots happens exactly once in the dashboard layer.

### Dashboard payload section

`current_weather` key inside C7 `dashboard/data.json` (also via
`emit(..., extra_sections={"current_weather": ...})`):

```jsonc
{
  "meta": {
    "generated_utc": "...Z",
    "equivalent_neutral_correction": false,
    "min_rank_n": 30,
    "default_lead_bucket": "48-72",
    "circularity_lead_buckets": ["0-12"],
    "models_assimilating_scatterometer": ["ecmwf_ifs", "icon_global", "icon_eu", "arome_france", "arpege_europe"],
    "models_non_assimilating": ["gfs_global", "ecmwf_aifs025"],
    "warnings": ["..."]
  },
  "scorecard": [
    {
      "pass_id": "...", "instrument": "ascat_metop_b", "model": "gfs_global",
      "lead_bucket": "48-72", "region": "all", "speed_bucket": "all",
      "vec_rmse_ms": 0.0, "vec_rmse_ci95": [0.0, 0.0],
      "speed_bias_ms": 0.0, "speed_bias_ci95": [0.0, 0.0],
      "dir_mae_deg": 0.0, "n": 0,
      "rankable": true, "circularity_contaminated": false
    }
  ],
  "trend": [
    { "pass_id": "...", "pass_time_utc": "...Z", "model": "...",
      "instrument": "...", "lead_bucket": "48-72", "vec_rmse_ms": 0.0, "n": 0 }
  ],
  "residuals": [
    { "pass_id": "...", "model": "...", "lat": 0.0, "lon": 0.0,
      "du_ms": 0.0, "dv_ms": 0.0, "obs_time_utc": "...Z" }
  ],
  "pressure": {
    "stations": [
      { "id": "LIET", "name": "Arbatax", "lat": 0.0, "lon": 0.0 }
    ],
    "mslp_scores": [
      { "station_id": "LIET", "model": "...", "lead_bucket": "48-72",
        "bias_hpa": 0.0, "rmse_hpa": 0.0, "n": 0 }
    ],
    "onset_lags": [
      { "station_id": "LIET", "day_utc": "2026-08-10", "model": "...",
        "obs_onset_utc": "...Z", "model_onset_utc": "...Z",
        "lag_minutes": 0, "kind": "thermal" }
    ]
  },
  "sentinel": {
    "status": "no_acquisition" | "available",
    "acquisition_utc": null,
    "footprint": null,
    "speed_field": null,
    "model_speed_fields": []
  },
  "expedition_calibration": [
    {
      "model": "gfs_global",
      "tws_scale_pct": 100.0,
      "twd_twist_deg": 0.0,
      "n": 0,
      "tws_scale_ci95": [100.0, 100.0],
      "twd_twist_ci95": [0.0, 0.0],
      "source": "scatterometer:ascat_metop_b:48-72"
    }
  ],
  "bucket_counts": {
    "headline": 0, "coastal": 0, "light_air": 0, "qc_reject": 0
  }
}
```

Rules enforced by validators and tests:

1. `expedition_calibration[].source` must start with `scatterometer:`.
2. Land-station and Sentinel paths must be structurally unable to write
   calibration rows.
3. Rows with `n < min_rank_n` have `rankable: false`.
4. Lead bucket `0-12` has `circularity_contaminated: true` for assimilating
   models; default UI view is `48-72`.

Fixture: `contracts/fixtures/verify/current_weather.json` plus per-instrument
NetCDF cells under `contracts/fixtures/verify/instruments/`.

---

## C10 — SAR scalar wind speed store (historical lee-shadow test)

Sentinel-1 L3 wind from Copernicus Marine Wind TAC. **Speed only.** The
dual-pol inversion uses a model-derived a priori wind to constrain direction,
so SAR direction is not independent evidence and must never appear in this
store or be used to test a model-derived hypothesis.

Zarr (or NetCDF) on disk at `data/sar/{product}.zarr`.

```
dims:      scene, lat, lon
coords:
  scene     int32, 0..n-1
  time      datetime64[ns], UTC, one valid time per scene (aligned on scene)
  lat       float32, ascending
  lon       float32, ascending
vars:
  wind_speed_ms   float32 (scene, lat, lon)   m/s, scalar 10 m equivalent
  incidence_deg   float32 (scene, lat, lon)   SAR incidence angle
  quality_flag    int8    (scene, lat, lon)   0 = usable; non-zero = discard
attrs:
  source            "sentinel1_l3_cmems"
  product_id        str
  fetched_utc       ISO 8601
  direction_policy  "speed_only_no_direction"
  units_note        "wind_speed_ms is SI; analysis layer converts to knots once"
```

Land / invalid retrievals are NaN in `wind_speed_ms`. Do **not** invent `u10` /
`v10` from a model prior — that would reintroduce the dependency this store
exists to break.

Fixture: `contracts/fixtures/sar_scenes_small.zarr` — synthetic August scenes
over the Sardinian east-coast corridor and a Tyrrhenian control corridor, with
known paired differentials for offline tests.

Dashboard payload optional section: `sar_shadow_test` (see module README). The
C7 validator accepts it as an optional top-level key; absence is valid.

This is distinct from C9 live `sentinel1` verification (opportunistic race-week
snapshots). C10 is the multi-year August falsification study for the AROME
east-Sardinia lee hypothesis.

