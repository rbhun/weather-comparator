# Cursor agent kickoff — cluster edition

P0 is complete. What remains is grouped into **three clusters** rather than six
agents, because Cursor agents spawn subagents and six branches is more than one
person can supervise from a phone.

The clusters follow real data coupling, not an arbitrary split:

| Cluster | Branch | Owns | Why grouped |
|---|---|---|---|
| **D — Data** | `cluster-d-data` | `src/pmc/io/`, `src/pmc/stats/` | stats consumes exactly what io produces; they share the zarr layout, cache model and model-id list |
| **B — Boat** | `cluster-b-boat` | `src/pmc/geo/`, `src/pmc/polar/`, `src/pmc/follow/` | the follower is the main consumer of polar and geometry; if interpolation is slow or `crosses_land` is wrong, the follower finds out first |
| **V — View** | `cluster-v-view` | `dashboard/`, `src/pmc/report/` | genuinely independent — only ever reads `data.json`, works off the fixture from minute one |

D and B and V run **in parallel**. All three start now.

The optimiser (`src/pmc/route/`) stays cut unless everything else is solid.

---

## The subagent problem — read this before dispatching

Subagents receive a task summary, not the parent's context. So if a subagent
needs to know that wind is stored as u/v with meteorological direction, it
either gets told explicitly or it invents something plausible and wrong.

**Frozen contracts matter more with subagents, not less.** Each cluster prompt
below contains a mandatory handoff block that the parent must pass verbatim to
every subagent it spawns. Do not let a parent paraphrase it.

---

## Cluster D — Data

> Branch: `cluster-d-data`

```
Read SPEC.md and CONTRACTS.md in the repo root, in full, before writing code.
Then read the "A1 - Data pipeline" and "A5 - Statistics and climatology"
sections of AGENTS.md. Both are yours.

You own src/pmc/io/ and src/pmc/stats/ ONLY. contracts/ is frozen — if you
believe a contract is wrong, STOP and tell me rather than editing it. Do not
touch src/pmc/geo/, polar/, follow/, report/ or dashboard/; other agents are
working there right now and will overwrite you.

Keep scripts/walking_skeleton.py passing at every commit.

=== PART 1: the data pipeline (src/pmc/io/) ===

Replace the stub with a real Open-Meteo client. Three properties matter most,
in this order:

1. Disk cache keyed by request parameters. Re-running must never re-download.
2. Resume after interruption. A killed three-hour download must not restart
   from zero.
3. Runtime model discovery. Do NOT trust a hardcoded model-id list. Probe each
   candidate, log which respond with actual data, write the survivors to
   config/models.yaml. Stale model ids fail silently and that is how a night
   gets lost.

Three fetch modes:
  - analysis: ECMWF IFS 9 km analysis, Augusts 2017-2025, hourly. The
    climatology backbone.
  - previous_runs: all surviving models, lead 1-7 days, Jan 2024 to now,
    resampled to a common 00/06/12/18 UTC grid.
  - live: current forecast, all models.

API key is in OPENMETEO_API_KEY. Use customer-* endpoints when present. Never
write the key into a tracked file.

Evaluate self-hosting EARLY, not on night three. Open-Meteo publishes its
server on GitHub under AGPLv3 and supports syncing archived data from AWS Open
Data to a local instance. If grid pulls are slow or rate-limited that removes
the bottleneck entirely. Report your conclusion to me explicitly.

=== PART 2: statistics (src/pmc/stats/) ===

1. Climatology (contract C6), per grid cell and UTC hour, over Augusts
   2017-2025 from the IFS 9 km analysis. Mean speed, vector-mean wind, calm
   probabilities, directional constancy, sample count. Mask cells with under
   200 samples rather than reporting them.

   Do NOT use 0.25 degree ERA5 for diurnal work. At 25 km the sea breeze is
   under-resolved and you will conclude it does not exist.

   The finding we are hunting: the offshore distance at which coastal thermal
   enhancement dies, along the Sardinian and Ligurian coasts. Produce a
   cross-shore transect analysis — mean wind versus distance from shore, by
   hour — at several points along each coast. That number directly answers
   "how tight do we sail to the coast".

2. Head-to-head. For every route pair on matched start times: fraction of days
   A beats B, median margin, and 10th/90th percentile margins. Report
   percentiles, never just means. A route that wins 60% of the time but loses
   by six hours when it loses is a bad route, and a mean hides that entirely.

3. Model skill from previous_runs: vector RMSE, speed bias, direction MAE,
   stratified by lead time and by observed wind bin.

   CRITICAL: if ECMWF IFS analysis is the reference, then IFS and AIFS are
   scored against their own analysis, and AIFS was trained on ECMWF reanalysis.
   Tag every ECMWF/AIFS row reference_biased=true. State the caveat inline in
   any output, never in a footnote.

   The common multi-model window is Jan 2024 onward and contains only two
   Augusts. So stratify by WIND REGIME, not calendar month — light,
   weak-gradient, coastal conditions occur year-round and give adequate sample
   size.

=== MANDATORY: pass this verbatim to every subagent you spawn ===

  PROJECT INVARIANTS — do not deviate, do not paraphrase:
  - Wind is stored as u10/v10 components in m/s. Direction is NEVER stored.
    Derive it on read. Meteorological convention: direction is where the wind
    comes FROM, 0 = from the north.
  - All storage and computation is UTC. Only display is local. Any variable
    holding local time carries a _local suffix; anything unsuffixed is UTC.
  - Storage layer is SI. Everything above it is knots and nautical miles.
    Conversion happens exactly once, in the loader.
  - Coordinates are always (lat, lon), degrees, north/east positive. Array
    dims are always (time, lat, lon), both axes ascending.
  - Missing data is NaN and is never silently filled. Land is NaN, not zero.
  - contracts/ is frozen. Import from it; never edit it.
  - Import validators from contracts/schemas.py and actually call them.
```

---

## Cluster B — Boat

> Branch: `cluster-b-boat`

```
Read SPEC.md and CONTRACTS.md in the repo root, in full, before writing code.
Then read the "A2 - Geometry and polar" and "A3 - Route follower" sections of
AGENTS.md. Both are yours.

You own src/pmc/geo/, src/pmc/polar/ and src/pmc/follow/ ONLY. contracts/ is
frozen — if you believe a contract is wrong, STOP and tell me rather than
editing it. Do not touch src/pmc/io/, stats/, report/ or dashboard/; other
agents are working there right now and will overwrite you.

Keep scripts/walking_skeleton.py passing at every commit.

=== PART 1: geometry (src/pmc/geo/) ===

The current stub uses crude rectangles. Replace with real coastline geometry
from GSHHG full resolution (Natural Earth 10m as fallback), using prepared
shapely geometry with a spatial index.

crosses_land is the critical function. At 0.1 degree a naive mask lets a track
cut through Capo Testa, through the Maddalena archipelago, and across the
corner of Corsica. Test segments against actual polygons, with a configurable
safety buffer defaulting to 0.5 nm so routes do not shave headlands.

Write a test asserting that a straight line from the gate (41.13N 9.55E) to
Monaco (43.73N 7.42E) is detected as crossing Corsica. If that test does not
FAIL against the crude stub before you write the real mask, your test is wrong
— check that first.

=== PART 2: polar (src/pmc/polar/) ===

Harden format detection: Expedition .pol and CSV, detected by content not
extension. Keep speed() interpolation of 1e6 points under one second — the
follower calls it millions of times.

Extend validate() to report the TWS range where the polar is interpolated
rather than measured, and warn loudly below 8 kt. Our polar is unvalidated
there and every downstream number inherits that uncertainty.

=== PART 3: the route follower (src/pmc/follow/) — THE CORE DELIVERABLE ===

March a boat along a FIXED polyline through a historical wind field. No
optimisation. Default timestep 10 minutes. At each step: interpolate u,v at
current position and time (bilinear in space, linear in time), derive TWS and
TWD, compute TWA from bearing to the next waypoint, look up boat speed in the
polar, advance.

Exactly two refinements, no more:

1. Beating. If the required TWA is inside the polar's upwind VMG angle, the
   boat cannot sail the bearing — sail at the VMG optimum on the favoured tack
   and credit VMG toward the bearing. Same for running. Without this, upwind
   legs are badly wrong, and most of leg 1 is upwind.

2. Stall detection. If speed is under 0.5 kt for over 6 hours: set
   stalled=True, KEEP GOING, record max_stall_hours. Do not abort. The stall
   is the finding, not an error.

Run every route against every historical August day, seeded at the real race
start hour. Emit contract C4 as parquet.

Why fixed tracks rather than an optimiser: we are comparing strategies, not
predicting elapsed time. Holding the track fixed isolates the strategic
variable. An optimiser would tactically compensate within each corridor and
partially mask the very difference we are trying to measure.

Sanity gate: 3000 day-by-route combinations must run in under a minute, and
elapsed times must land in the 50-90 hour range for a 52-footer over 500 nm.
If they come out near 20 hours your polar array is transposed — check that
before anything else.

=== MANDATORY: pass this verbatim to every subagent you spawn ===

  PROJECT INVARIANTS — do not deviate, do not paraphrase:
  - Wind is stored as u10/v10 components in m/s. Direction is NEVER stored.
    Derive it on read. Meteorological convention: direction is where the wind
    comes FROM, 0 = from the north.
  - All storage and computation is UTC. Only display is local. Any variable
    holding local time carries a _local suffix; anything unsuffixed is UTC.
  - Storage layer is SI. Everything above it is knots and nautical miles.
    Conversion happens exactly once, in the loader.
  - Coordinates are always (lat, lon), degrees, north/east positive. Array
    dims are always (time, lat, lon), both axes ascending.
  - Missing data is NaN and is never silently filled. Land is NaN, not zero.
  - The polar is UNVALIDATED below 8 kt TWS. Every elapsed time is a RELATIVE
    comparison, never an absolute prediction. Never present one as a forecast.
  - contracts/ is frozen. Import from it; never edit it.
```

---

## Cluster V — View

> Branch: `cluster-v-view`

```
Read SPEC.md and CONTRACTS.md in the repo root, in full, before writing code.
Then read the "A6 - Dashboard" section of AGENTS.md. That is yours.

You own dashboard/ and src/pmc/report/ ONLY. contracts/ is frozen. Do not
touch anything under src/pmc/ except report/; other agents are working there
right now and will overwrite you.

You are blocked by nobody. Build against contracts/fixtures/data.json from the
first minute.

Deliver ONE self-contained HTML file plus data.json. Plotly vendored locally in
dashboard/vendor/, never from a CDN. It must open from the filesystem with no
internet and no server. Verify by actually disconnecting, not by assuming.

Views in priority order:
1. Calm risk map — domain heatmap of p_below_5kt, hour-of-day slider, course
   overlaid. This is the flagship; build it first and build it well.
2. Route comparison — elapsed-time distributions per route, plus the
   head-to-head table with margins and percentiles.
3. Point detail on click — diurnal wind rose, mean speed by hour, directional
   constancy, sample count.
4. Cross-shore transects — mean wind versus distance offshore, by hour.
5. Model skill table by model and lead time.

The reader is a tired tactician on a moving boat at night. Therefore:
- Local time by DEFAULT from meta.display_timezone. Fixed-position UTC/local
  toggle, current mode always visible, switching every timestamp on the page at
  once including axis labels. Every rendered time carries its zone label, e.g.
  "21 Aug 14:30 CEST". Never a bare time.
- meta.warnings renders as a persistent banner, not a dismissible toast. If
  polar_is_validated is false, that warning is on screen permanently.
- Every number carries its uncertainty. No bare point estimates anywhere.
- Skill rows with reference_biased=true must be visually distinct and
  annotated. ECMWF and AIFS are scored against their own analysis and are NOT
  comparable to the other models. A reader must not be able to misread that
  table at 3am.
- Dark background, high contrast, large type. Viridis or cividis for maps.
  Never red-green to mean good-bad.
- No animation, no transitions, no hover-only information. A hover tooltip is
  useless on a tablet in the rain. Everything important is visible or one tap
  away.

Done when it opens offline on a machine that has never seen the project, and
someone who has not read the spec can find the calm-risk map unaided.

=== MANDATORY: pass this verbatim to every subagent you spawn ===

  PROJECT INVARIANTS — do not deviate, do not paraphrase:
  - The dashboard reads data.json and nothing else. No network, no server, no
    CDN, no build step required to open it.
  - Local time by default; every displayed time carries its zone label.
  - meta.warnings is always visible, never dismissible.
  - reference_biased rows are visually distinct and annotated.
  - Colour scales are colourblind-safe and never red-green for good-bad.
  - contracts/ is frozen. Validate payloads with contracts/schemas.py.
```

---

## Merge order and supervision

Merge each cluster to `main` when its walking-skeleton run passes. The clusters
touch disjoint directories, so merges should be near-trivial — but only if all
three branched from the same committed contracts.

Watch for one failure mode: an agent editing a file it does not own. That is
the isolation breaking down. Stop it and re-issue the prompt with the ownership
line first rather than buried mid-prompt.

If two clusters ever need to change the same file, the split is wrong. Stop and
re-cut the boundary rather than letting them fight over it.
