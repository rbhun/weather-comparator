# Agent briefs

Give each agent: `SPEC.md` + `CONTRACTS.md` + **only its own section below.**
Do not give an agent another agent's brief — it invites them to "helpfully" fix
someone else's module and cause merge conflicts you will debug at 3am.

## Sequencing

```
Hour 0     YOU: commit contracts/ + fixtures + walking_skeleton.py
           (nothing starts before this exists)

Wave 1     A1 data      A2 geo+polar      A6 dashboard shell
           (all three work purely against fixtures; fully parallel)

Wave 2     A3 follower  A5 stats
           (A3 needs A2's polar; A5 needs A1's cache — both landed in wave 1)

Wave 3     A4 optimiser
           (only if waves 1–2 are solid. This is the cuttable one.)
```

**A4 is optional.** If time runs short, ship without the optimiser. A3 answers
the strategic question adequately on its own — see A3's brief for why.

---

## A7 — Live observation verification (Current weather)

Own `src/pmc/verify/` and the **Current weather** dashboard tab.

Scores currently-running forecasts against scatterometer, land-station, and
Sentinel-1 observations. Contract: **C9**. Develop against
`contracts/fixtures/verify/`. Do not pool observation classes. Expedition
calibration is scatterometer 48–72 h only.

Done when: acceptance tests in `tests/test_verify.py` pass, the tab renders
offline from the fixture payload, and re-ingest is idempotent.

---

## A1 — Data pipeline

Own `src/pmc/io/`.

Build an Open-Meteo client that fetches the domain grid and writes C1 zarr
stores. Three modes:

1. `analysis` — ECMWF IFS 9 km analysis, Augusts 2017–2025, hourly. This is the
   climatology backbone.
2. `previous_runs` — all available models, lead 1–7 d, Jan 2024→now, resampled
   to the 00/06/12/18 UTC common grid.
3. `live` — current forecast, all models, for race-time use.

Requirements:

- **Discover model ids at runtime.** Do not trust a hardcoded list; probe each
  candidate, log which respond with data, write the survivors to
  `config/models.yaml`. Model ids change and stale ones fail silently.
- **Disk cache keyed by request params.** Re-running must not re-download. This
  is the single most important property of your module.
- Resume after interruption. A 3-hour download that restarts from zero is a
  dead night.
- Respect rate limits with exponential backoff. The subscription is generous
  but not infinite.
- Grid fetches: batch coordinates per request where the API allows; otherwise
  parallelise with a bounded thread pool (max 8) and a jitter delay.
- Log a per-run summary: cells fetched, cells missing, wall time, cache hits.

**Consider self-hosting.** Open-Meteo publishes its server on GitHub under
AGPLv3 and supports syncing archived data from AWS Open Data to a local
instance. If grid pulls prove slow or rate-limited, a local instance removes the
API bottleneck entirely. Evaluate this early — do not discover on night three
that you needed it.

Done when: all three modes fill a zarr that passes the C1 validator, a second
run of the same command hits 100% cache, and a killed run resumes cleanly.

---

## A2 — Geometry and polar

Own `src/pmc/geo/` and `src/pmc/polar/`.

**Geometry.** Land mask and land-crossing test from real coastline data (GSHHG
full resolution, or Natural Earth 10m as a fallback). Great-circle distance and
bearing, rhumb-line variants, cross-track distance to a leg.

The critical function is `crosses_land`. At 0.1° a naive mask lets a track cut
through Capo Testa, through the Maddalena archipelago, and across the corner of
Corsica. Use prepared shapely geometry with a spatial index and test the segment
against actual polygons. Add a configurable safety buffer (default 0.5 nm) so
routes do not shave headlands.

Write a test that asserts a straight line from the gate to Monaco is correctly
detected as crossing Corsica. If that test does not fail before you write the
mask, your test is wrong.

**Polar.** Implement C2. Bilinear interpolation, vectorised over arrays of
(twa, tws) — the follower will call this millions of times, so it must not loop
in Python. Clamp rather than extrapolate outside the TWS range.

Also provide `vmg_optimum` with caching; A4 depends on it heavily.

Add a `validate()` that reports: monotonicity violations, implausible values
(bsp > 1.5 × tws for tws > 10), and the TWS range where the polar is
interpolated rather than measured. **Print a loud warning for TWS < 8 kt.**
Our polar is not validated there and every downstream number inherits that.

Done when: `crosses_land` correctly rejects the gate→Monaco straight line and
accepts a legitimate east-of-Corsica track, and polar interpolation of 10^6
points takes under a second.

---

## A3 — Route follower

Own `src/pmc/follow/`. Depends on A2.

March a boat along a **fixed** polyline through a historical wind field. No
optimisation.

```
for each timestep dt (default 10 min):
    interpolate u,v at current position and time (bilinear in space, linear in time)
    derive tws, twd
    bearing = great-circle bearing to next waypoint
    twa = angular difference(twd, bearing)
    bsp = polar.speed(twa, tws)
    advance position by bsp * dt along bearing
    if reached waypoint: advance to next
```

Two refinements, and only these two:

- **Beating.** If |TWA| < the polar's upwind VMG angle, the boat cannot sail the
  bearing. Sail at the VMG optimum angle on the favoured tack and credit VMG
  toward the bearing. Same for running. Without this, upwind legs — which is
  most of leg 1 — will be badly wrong.
- **Stall detection.** If speed < 0.5 kt for > 6 h, mark `stalled=True`, keep
  going, record `max_stall_hours`. Do not abort; the stall *is* the finding.

Run every route against every historical August day, starting at the real race
start time of day. Emit C4.

**Why this rather than an optimiser:** we are comparing strategies, not
predicting elapsed time. Holding the track fixed isolates the strategic
variable. An optimiser would tactically compensate within each corridor and
partially mask the difference we are trying to measure. It is also ~200 lines
and nearly impossible to get subtly wrong, which matters more than realism this
week.

Done when: 3,000 day×route combinations run in under a minute, and a sanity
check shows elapsed times in the 50–90 hour range for a 52-footer over 500 nm.
If they come out at 20 hours, your polar indexing is transposed.

---

## A4 — Isochrone optimiser (CUTTABLE)

Own `src/pmc/route/`. Only start if waves 1–2 are solid.

Standard isochrone advance: from the current front, fan out over headings,
advance one timestep, prune to the outer envelope in bearing-to-destination
bins. Hard gate constraint at Porto Cervo — optimise start→gate, then
gate→finish, separately. Land avoidance via A2. Manoeuvre penalty (default 60 s
plus a speed reduction) so it does not tack every timestep.

**Do not try to beat Expedition.** Validate by running both on ten identical
cases and checking they agree on the *strategic decision*, not the ETA. If your
router picks the same side of Corsica 9 times out of 10, it is good enough for
what we need it for.

Known weak spots to document honestly: sail changes, current, and
land-avoidance quality. State them in the README.

Done when: it beats the best fixed route from A3 on the same day (it must, or
something is broken), and the ten-case Expedition comparison is written up.

---

## A5 — Statistics and climatology

Own `src/pmc/stats/`. Depends on A1.

**Climatology (C6).** Per grid cell and UTC hour, over Augusts 2017–2025 from
the IFS 9 km analysis: mean speed, vector-mean wind, calm probabilities,
directional constancy, sample count. Assert n > 200 per cell before a cell is
considered trustworthy; mask the rest.

The finding we are hunting: **the offshore distance at which the coastal
thermal enhancement dies**, along the Sardinian and Ligurian coasts. Produce a
cross-shore transect analysis — mean wind vs distance from shore, by hour — at
several points along each coast. That number directly answers "how tight do we
go".

**Head-to-head (C7).** For every pair of routes, the fraction of historical days
A beats B, the median margin, and the 10th/90th percentile margins. Report
percentiles, never just means. A route that wins 60% of the time but loses by
six hours when it loses is a bad route, and a mean hides that completely.

**Model skill.** Vector RMSE, speed bias, direction MAE, stratified by lead time
and by observed wind bin. Tag every ECMWF/AIFS row `reference_biased: true` —
see SPEC §4.

Done when: the climatology has no unmasked low-sample cells and the transect
analysis produces a defensible number for coastal enhancement range.

---

## A6 — Dashboard

Own `dashboard/` and `src/pmc/report/`.

**One self-contained HTML file** plus `data.json`. Plotly vendored locally. Must
open from the filesystem with no internet and no server. Test that by
disconnecting, not by assuming.

Start immediately against the fixture `data.json` — you are not blocked by
anyone.

Views, in priority order:

1. **Calm risk map.** Domain heatmap of `p_below_5kt`, hour-of-day slider,
   course overlaid. This is the flagship.
2. **Route comparison.** Elapsed-time distributions per route as box plots or
   violins, plus the head-to-head table with margins and percentiles.
3. **Point detail.** Click any grid cell: diurnal wind rose, mean speed by hour,
   directional constancy, sample count.
4. **Cross-shore transects.** Mean wind vs distance offshore, by hour.
5. **Model skill.** Table by model and lead time, with biased-reference rows
   visually separated and labelled.

Design constraints — the reader is exhausted, on a moving boat, at night:

- Every number carries its uncertainty. No bare point estimates.
- The `meta.warnings` array renders as a persistent banner, not a dismissible
  toast. If `polar_is_validated` is false, that warning is always on screen.
- Dark background. High contrast. Large type.
- Colour scales must be colourblind-safe and must not use red-green to mean
  good-bad. Use viridis or cividis for the maps.
- No animation, no transitions, no hover-only information — a hover tooltip is
  useless on a tablet in the rain. Everything important is visible or one tap
  away.
- **Local time by default**, from `meta.display_timezone`. A fixed-position
  UTC/local toggle, current mode always visible, switching every timestamp on
  the page at once including axis labels. Every rendered time carries its zone
  label: `21 Aug 14:30 CEST`. Never a bare time. See SPEC §5.

Done when: it opens offline on a machine that has never seen the project,
renders the fixture data, and someone who has not read this document can find
the calm-risk map without being told where it is.
