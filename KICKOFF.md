# Cursor agent kickoff prompts

Paste `SPEC.md`, `CONTRACTS.md` and `AGENTS.md` into the repo root first and
commit them. Then work through the prompts below in order.

**P0 must be finished and merged to `main` before anything else starts.**
Everything after it depends on the contracts and fixtures existing. Running
Wave 1 before P0 lands will produce five agents building against nothing.

Each prompt below is self-contained. Copy one block, paste into a fresh Cursor
agent, let it run.

---

## P0 — Hour zero (run alone, first)

> Branch: `p0-contracts`

```
Read SPEC.md, CONTRACTS.md and AGENTS.md in the repo root, in full, before
writing any code.

Your job is to build "hour zero": the frozen contracts, synthetic fixtures,
stub modules and walking skeleton that let five other agents work in parallel
without blocking each other. You are NOT implementing any real functionality.

Create:

1. config/course.yaml
   Start 38.20N 13.32E (Golfo di Mondello). Gate 41.13N 9.55E (Porto Cervo),
   tolerance 2.0 nm. Finish 43.73N 7.42E (Monaco). start_time_utc
   2026-08-18T09:55:00Z. Climatology months [8], years 2017-2025.
   display_timezone "Europe/Rome".

2. config/domain.yaml
   lat 37.5-44.0, lon 6.5-14.5, resolution 0.1. Also fixture_resolution 0.25
   and fixture_days 30 so the committed fixture stays small.

3. config/routes.yaml
   At least 8 candidate routes as described in CONTRACTS.md C3. They must span
   three independent decisions: east/west of Ustica; tight/offshore on the
   Sardinian coast; and Bonifacio vs east-of-Corsica vs west-of-Corsica. Every
   route must include the gate waypoint.

4. contracts/schemas.py
   Implement every contract in CONTRACTS.md: constants, uv_to_tws_twd,
   tws_twd_to_uv, angular_difference, validate_wind_store, the Polar
   dataclass with speed()/vmg_optimum()/validate(), Route with
   assert_passes_gate(), the C4/C5 column dicts, validate_climatology,
   directional_constancy, validate_dashboard_payload, and the geometry helpers
   haversine_nm / initial_bearing_deg / advance_position.

   Polar.speed must be vectorised bilinear interpolation with NO Python loop,
   and must CLAMP outside the TWS range rather than extrapolate.

5. contracts/make_fixtures.py
   Deterministic (fixed seed, so fixtures do not churn in git). Generates:
     - contracts/fixtures/wind_small.zarr : 30 days hourly at 0.25 deg over the
       full domain, u10/v10 in m/s, with a genuine diurnal sea-breeze signal
       that decays with distance offshore, and a crude rectangular land mask
       where land cells are NaN. Mark it clearly as FIXTURE ONLY in comments.
     - contracts/fixtures/polar_52ft.pol : fabricated 52-footer polar in
       Expedition format (TWS header row, TWA first column, tab separated).
     - contracts/fixtures/climatology_small.nc : derived from the fixture wind.
     - contracts/fixtures/data.json : a C7 dashboard payload.
   Every generated fixture must be validated against its schema before writing.
   Also write contracts/fixtures/README.md explaining that these are committed
   on purpose and must not be deleted.

6. src/pmc/{io,geo,polar,follow,route,stats,report}/__init__.py
   Working STUBS with the exact signatures in CONTRACTS.md C8. Each module
   docstring must start by naming its owning agent and saying "STUB" if it is
   one. io.fetch_wind returns the fixture path. geo uses crude boxes, clearly
   marked as fixture-grade. polar.load_polar is real. follow.follow returns a
   schema-valid fixed row. stats.head_to_head is real (percentiles, not just
   means). report.emit is real. route.optimise raises NotImplementedError.

7. scripts/walking_skeleton.py
   Runs the whole pipeline end to end on fixtures with no network: load config,
   assert all routes pass the gate, open the wind store and validate C1, check
   the uv round trip, check geometry, load and validate the polar, benchmark
   1e6 polar interpolations, run the follower over all routes for 12 start
   days, compute head-to-head, compute climatology, emit the dashboard payload.
   Print a clear pass/fail per stage and list which stubs remain.
   Warn if median elapsed time falls outside 40-110 hours - that band catches a
   transposed polar array, which otherwise looks superficially plausible.

8. tests/test_contracts.py
   Include at minimum:
     - a northerly has negative v
     - a westerly has positive u
     - uv round trip is exact
     - averaging 350 and 10 degrees via components gives north, not 180
       (this is the test that justifies storing u/v at all - label it clearly)
     - directional_constancy is 1.0 for steady wind, near 0 for random
     - Palermo to Monaco great circle is 380-430 nm
     - every configured route passes the gate
     - a route missing the gate raises
     - the fixture wind satisfies C1 and has land as NaN
     - the fixture wind has a real diurnal cycle
     - the polar clamps rather than extrapolates
     - the polar is symmetric about zero TWA
     - upwind VMG angle is between 35 and 55 degrees

Also write .gitignore: ignore data/** and .env, but explicitly un-ignore
contracts/fixtures/** with a comment saying deleting them breaks five agents.

Acceptance, all three must pass and you must show me the output:
    python contracts/make_fixtures.py
    python -m pytest tests/ -q
    python scripts/walking_skeleton.py

The fixture directory must be under 6 MB. Do not commit anything to data/.
```

---

## Wave 1 — run these three in parallel once P0 is merged

### A1 — Data pipeline

> Branch: `a1-data`

```
Read SPEC.md and CONTRACTS.md in full, then the "A1 - Data pipeline" section
of AGENTS.md. That section is your task. Do not read or modify other agents'
sections or modules.

You own src/pmc/io/ only. contracts/ is frozen - if you think a contract is
wrong, STOP and tell me rather than editing it.

Build the real Open-Meteo client replacing the stub. The three most important
properties, in order:

1. Disk cache keyed by request parameters. Re-running must not re-download.
2. Resume after interruption. A killed 3-hour download must not restart at zero.
3. Runtime model discovery. Do not trust a hardcoded model id list - probe each
   candidate, log which respond with data, write survivors to config/models.yaml.
   Stale model ids fail silently and that is how a night gets lost.

The API key is in the OPENMETEO_API_KEY env var. Use the customer-* endpoints
when it is present. Never write the key into a tracked file.

Evaluate self-hosting EARLY, not on night three: Open-Meteo publishes its
server on GitHub under AGPLv3 and supports syncing archived data from AWS Open
Data locally. If grid pulls are slow or rate limited, that removes the API
bottleneck entirely. Tell me what you conclude.

Done when all three modes fill a C1-conformant zarr, a second run of the same
command is 100% cache hits, and a killed run resumes cleanly. Keep
scripts/walking_skeleton.py passing throughout.
```

### A2 — Geometry and polar

> Branch: `a2-geo`

```
Read SPEC.md and CONTRACTS.md in full, then the "A2 - Geometry and polar"
section of AGENTS.md. That section is your task. Do not read or modify other
agents' sections or modules.

You own src/pmc/geo/ and src/pmc/polar/ only. contracts/ is frozen.

The current geo stub uses crude rectangles. Replace it with real coastline
geometry from GSHHG full resolution (Natural Earth 10m as a fallback), using
prepared shapely geometry with a spatial index.

crosses_land is the critical function. At 0.1 degree a naive mask lets a track
cut through Capo Testa, through the Maddalena archipelago, and across the
corner of Corsica. Test the segment against actual polygons, with a
configurable safety buffer defaulting to 0.5 nm so routes do not shave
headlands.

Write a test asserting that a straight line from the gate (41.13N 9.55E) to
Monaco (43.73N 7.42E) is detected as crossing Corsica. If that test does not
fail before you write the real mask, your test is wrong - check that first.

For the polar: harden format detection (Expedition .pol and CSV, detected by
content not extension) and make sure speed() interpolation of 1e6 points stays
under one second, since the follower calls it millions of times.

Extend validate() to report the TWS range where the polar is interpolated
rather than measured, and warn loudly below 8 kt. Our polar is unvalidated
there and every downstream number inherits that uncertainty.

Keep scripts/walking_skeleton.py passing throughout.
```

### A6 — Dashboard

> Branch: `a6-dashboard`

```
Read SPEC.md and CONTRACTS.md in full, then the "A6 - Dashboard" section of
AGENTS.md. That section is your task. Do not read or modify other agents'
sections or modules.

You own dashboard/ and src/pmc/report/ only. contracts/ is frozen.

You are not blocked by anyone - build against contracts/fixtures/data.json
from the first minute.

Deliver ONE self-contained HTML file plus data.json. Plotly vendored locally in
dashboard/vendor/, never from a CDN. It must open from the filesystem with no
internet and no server. Test that by actually disconnecting, not by assuming.

Views in priority order: calm risk map (domain heatmap of p_below_5kt with an
hour-of-day slider and the course overlaid - this is the flagship); route
comparison (elapsed-time distributions plus the head-to-head table with
margins and percentiles); point detail on click (diurnal wind rose, mean speed
by hour, directional constancy, sample count); cross-shore transects; model
skill table.

The reader is a tired tactician on a moving boat at night. Therefore:
- Local time by default from meta.display_timezone, with a fixed-position
  UTC/local toggle that switches every timestamp on the page at once,
  including axis labels. Every rendered time carries its zone label, e.g.
  "21 Aug 14:30 CEST". Never a bare time.
- meta.warnings renders as a persistent banner, not a dismissible toast. If
  polar_is_validated is false that warning is always on screen.
- Every number carries its uncertainty. No bare point estimates.
- Skill rows with reference_biased true must be visually distinct and
  annotated - ECMWF and AIFS are scored against their own analysis and are not
  comparable to the other models.
- Dark background, high contrast, large type. Viridis or cividis for maps,
  never red-green to mean good-bad.
- No animation, no hover-only information. A hover tooltip is useless on a
  tablet in the rain. Everything important is visible or one tap away.

Done when it opens offline on a machine that has never seen the project and
someone who has not read the spec can find the calm-risk map unaided.
```

---

## Wave 2 — start once Wave 1 has landed

### A3 — Route follower (this is the core deliverable)

> Branch: `a3-follow`

```
Read SPEC.md and CONTRACTS.md in full, then the "A3 - Route follower" section
of AGENTS.md. That section is your task. Do not read or modify other agents'
sections or modules.

You own src/pmc/follow/ only. contracts/ is frozen.

March a boat along a FIXED polyline through a historical wind field. No
optimisation. Default timestep 10 minutes. At each step: interpolate u,v at
the current position and time (bilinear in space, linear in time), derive TWS
and TWD, compute TWA from the bearing to the next waypoint, look up boat speed
in the polar, advance.

Exactly two refinements, no more:

1. Beating. If the required TWA is inside the polar's upwind VMG angle the
   boat cannot sail the bearing - sail at the VMG optimum on the favoured tack
   and credit VMG toward the bearing. Same for running. Without this, upwind
   legs will be badly wrong, and most of leg 1 is upwind.

2. Stall detection. If speed is under 0.5 kt for over 6 hours, set
   stalled=True, keep going, record max_stall_hours. Do NOT abort. The stall
   is the finding, not an error.

Run every route against every historical August day, seeded at the real race
start hour. Emit C4 parquet.

Why fixed tracks rather than an optimiser: we are comparing strategies, not
predicting elapsed time. Holding the track fixed isolates the strategic
variable. An optimiser would tactically compensate within each corridor and
partially mask the difference we are trying to measure.

Done when 3000 day-by-route combinations run in under a minute and elapsed
times land in the 50-90 hour range for a 52-footer over 500 nm. If they come
out near 20 hours your polar indexing is transposed - check that first.
```

### A5 — Statistics and climatology

> Branch: `a5-stats`

```
Read SPEC.md and CONTRACTS.md in full, then the "A5 - Statistics and
climatology" section of AGENTS.md. That section is your task. Do not read or
modify other agents' sections or modules.

You own src/pmc/stats/ only. contracts/ is frozen.

Three deliverables.

1. Climatology (C6) per grid cell and UTC hour over Augusts 2017-2025 from the
   ECMWF IFS 9 km analysis. Mean speed, vector-mean wind, calm probabilities,
   directional constancy, sample count. Mask any cell with under 200 samples
   rather than reporting it.

   Do NOT use 0.25 degree ERA5 for the diurnal work. At 25 km the sea breeze
   is under-resolved and you will conclude it does not exist.

   The finding we are hunting: the offshore distance at which coastal thermal
   enhancement dies, along the Sardinian and Ligurian coasts. Produce a
   cross-shore transect analysis - mean wind versus distance from shore, by
   hour - at several points along each coast. That number directly answers
   "how tight do we go".

2. Head-to-head. For every route pair: fraction of days A beats B, median
   margin, and 10th/90th percentile margins on matched start times. Report
   percentiles, never just means. A route that wins 60% of the time but loses
   by six hours when it loses is a bad route and a mean hides that completely.

3. Model skill from the Previous Runs data: vector RMSE, speed bias, direction
   MAE, stratified by lead time and by observed wind bin.

   Critical: if ECMWF IFS analysis is the reference, then IFS and AIFS are
   being scored against their own analysis, and AIFS was trained on ECMWF
   reanalysis. Tag every ECMWF/AIFS row reference_biased=true. State the
   caveat inline in any output, never in a footnote.

   The common multi-model window is Jan 2024 onward, containing only two
   Augusts. So stratify skill by WIND REGIME, not calendar month - light,
   weak-gradient, coastal conditions occur year-round and give adequate sample
   size.

Keep scripts/walking_skeleton.py passing throughout.
```

---

## Wave 3 — only if Waves 1 and 2 are solid

### A4 — Isochrone optimiser (CUTTABLE)

> Branch: `a4-route`

```
Read SPEC.md and CONTRACTS.md in full, then the "A4 - Isochrone optimiser"
section of AGENTS.md. You own src/pmc/route/ only. contracts/ is frozen.

This module is explicitly cuttable. If the rest of the project is not solid,
tell me and stop - shipping without you is the plan, not a failure.

Standard isochrone advance: fan out over headings from the current front,
advance one timestep, prune to the outer envelope in bearing-to-destination
bins. Hard gate constraint at Porto Cervo - optimise start-to-gate and
gate-to-finish separately. Land avoidance via pmc.geo. Manoeuvre penalty
(default 60 s plus a speed reduction) so it does not tack every timestep.

Do not try to beat Expedition. Validate by running both on ten identical cases
and checking they agree on the STRATEGIC DECISION, not the ETA. If it picks
the same side of Corsica 9 times out of 10 it is good enough.

Document honestly where it is weak: sail changes, current handling, and
land-avoidance quality. Put that in the module README.

Done when it beats the best fixed route from A3 on the same day - it must, or
something is broken - and the ten-case Expedition comparison is written up.
```

---

## Merge order

Merge each branch to `main` when its walking-skeleton run passes. The modules
touch disjoint directories so merges should be near-trivial, but only if every
branch was cut from the same committed contracts.

If two agents ever need to change the same file, that is a signal the split is
wrong. Stop and re-cut the boundary rather than letting them fight over it.
