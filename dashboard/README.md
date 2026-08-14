# Cluster V dashboard

Offline tactical dashboard for PMC-2026.

## What it does

- Renders a single-file HTML dashboard from `dashboard/data.json`.
- Shows calm-risk heatmap (`p_below_5kt`) with route overlays.
- Shows route distributions and head-to-head margins with percentiles.
- Shows click-to-inspect point detail by hour.
- Shows model-skill table with clearly marked `reference_biased` rows.
- Renders persistent warnings from `meta.warnings` and keeps the polar
  unvalidated warning on-screen when `polar_is_validated=false`.
- Defaults to local display time from `meta.display_timezone`, with a global
  UTC/local toggle that updates all visible time labels.

## How to run

1. Prepare payload:

   ```bash
   PYTHONPATH="/workspace/src:/workspace" \
     python3 -m pmc.report \
     --input "/workspace/contracts/fixtures/data.json" \
     --output "/workspace/dashboard/data.json"
   ```

2. Open `dashboard/index.html` directly from the filesystem (no server).

3. If your browser blocks direct `file://` fetch for `data.json`, use the
   built-in file picker shown on the page to load the same local `data.json`.

## What it gets wrong / current limitations

- Wind-rose and full point-detail diagnostics depend on optional fields
  (`vector_mean_u`, `vector_mean_v`, `directional_const`, `n_samples`) that may
  be missing in minimal C7 payloads; the UI shows explicit "not available"
  states in that case.
- Cross-shore transect plots require an optional `transects` section in
  `data.json`; with core C7-only data the panel shows a clear missing-data
  message instead of inventing values.
- This is a tactical comparison dashboard, not a deterministic forecast tool.
  Elapsed-time outputs inherit polar uncertainty below 8 kt TWS.
