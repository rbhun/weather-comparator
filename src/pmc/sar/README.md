# SAR lee-shadow falsification (`src/pmc/sar/`)

Independent Sentinel-1 evidence for the AROME-reported Sardinian east-coast
wind shadow (0–5 nm weaker than 7.5–10 nm). This is a **falsification test**,
not an exploration: the prediction is registered in `config/sar.yaml` before
analysis.

## What it does

1. Loads Copernicus Marine Wind TAC Sentinel-1 L3 daily wind (**speed only** —
   C9; no direction, no fabricated `u10`/`v10`).
2. For each scene, bands retrievals by distance to coastline (Sardinia) or a
   virtual meridian (Tyrrhenian control).
3. **Discards** any scene lacking valid data in both the 0–5 nm and 7.5–10 nm
   bands.
4. Takes the paired difference (inshore − offshore) as the unit of analysis.
5. Requires the control corridor differential to be indistinguishable from zero;
   otherwise the pipeline marks itself invalid and suppresses the Sardinian mean.
6. Emits a dashboard section with verdict
   `supported` | `contradicted` | `insufficient sample`.

## How to run

```bash
# Offline / CI — fixture only
python -m pmc.sar.cli fetch --fixture
python -m pmc.sar.cli analyse --fixture --output data/sar/shadow_test.json

# Live (requires Copernicus Marine account env vars + copernicusmarine package)
python -m pmc.sar.cli fetch --start 2018-08-01 --end 2025-08-31
python -m pmc.sar.cli analyse --sar data/sar/s1_l3_*.zarr
```

Optional three-way comparison: pass `--arome` / `--era5` speed fields
(`wind_speed_kt` or `u10`/`v10` in m/s).

## What it gets wrong

- **Diurnal sampling.** S1 overpasses are near dawn/dusk local. Mid-afternoon
  lee/thermal peaks are largely unsampled. A null result may be
  "uninformative window", not "no shadow". The payload states this first-class.
- **Surfactants / slicks** darken sheltered water and invert to spuriously low
  wind — exactly where the hypothesis lives. Low-speed-tail and
  exclude-below-3 m/s diagnostics are reported; they are imperfect.
- **Land spillover** still contaminates the first coastal pixels; buffer
  sensitivity across ≥3 values is mandatory. If the finding survives at only
  one buffer, it is not a finding.
- **Incidence angle** correlates with cross-track position and thus with
  distance from coast; the correlation diagnostic must be checked before
  trusting a differential.
- **Sparse August archive.** After S1B failure (Dec 2021) pass counts are
  uneven. Below `min_scenes_threshold` (default 15) the verdict is forced to
  `insufficient sample` and no mean is plotted.
- **ERA5** in the three-way table is a coarse baseline only — unreliable for
  Bonifacio and island thermals.
- **Not operational for race week.** Revisit is far too sparse; this answers a
  post-race evidence question only.
- Fixture coast longitude is a cartoon polyline for offline tests; production
  analysis uses the real OSM/GSHHG coastline via `pmc.geo`.
