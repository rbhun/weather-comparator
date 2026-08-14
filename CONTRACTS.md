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
  ]
}
```

`reference_biased: true` for every ECMWF/AIFS row. The dashboard must render
those rows visually distinct and annotated.

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
# report
def emit(...) -> Path
```

Signatures are frozen. Add keyword arguments with defaults if you must; never
change or reorder existing ones.
