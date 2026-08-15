# Cluster V dashboard

Offline tactical dashboard for PMC-2026.

## What it does

- Thin header tabs: **Historical weather**, **Model skill**, **Historical
  result** (YB tracks), and **Settings**.
- **Model skill** is its own page: winners by lead, RMSE/bias/direction vs lead
  day, wind-bin and model filters, hide-biased toggle, and a sortable detail
  table. ECMWF/AIFS rows stay visually separated as reference-biased.
- **Settings** holds the UTC/local toggle, operational warnings, visitor
  stats (GoatCounter: count, country, phone vs desktop, revisits), and a
  read-only Historical weather block (downloaded years, area, race).
  Selection of those fields comes later. Create the free GoatCounter site
  `rbhun-pmc` once at goatcounter.com so the Settings link has data.
- Renders a single-file HTML dashboard from `dashboard/data.json`.
- Shows calm-risk heatmap (`p_below_5kt`) with route overlays.
- Renders a pre-cached OpenStreetMap base map (CARTO Voyager tiles) plus an
  OpenSeaMap seamark overlay beneath the calm-risk heatmap, from local raster
  tiles in `dashboard/tiles/` (no network needed at render time).
- Shows route distributions and head-to-head margins with percentiles.
- Shows click-to-inspect point detail by hour.
- Model skill lives on the **Model skill** tab (not buried under Historical
  weather). Rebuild scores with
  `python3 scripts/compute_model_skill.py --patch-dashboard` (cached under
  `data/cache/model_skill/`).
- Operational warnings live on Settings, including a visualization-only
  disclaimer. Stale follower/year-cube notes are dropped. The polar
  unvalidated note still appears there when `polar_is_validated=false`.
- Defaults to local display time from `meta.display_timezone`, with a UTC/local
  toggle on the Settings tab that updates all visible time labels.
- **Historical result** overlays YB tracks from 2017–2019 and 2021–2025. Year
  and class chips filter the fleet. **Top 3 absolute** is uncorrected line
  honours; **Top 3 per class** is the three fastest uncorrected elapsed times
  in each selected class (clock time, not IRC/ORC handicap). **Year weather**
  paints that edition's race-window mean 10 m wind from Open-Meteo IFS analysis
  when the overlay includes a `weather`
  block. Rebuild tracks with `scripts/fetch_yb_tracks.py` and weather with
  `scripts/fetch_yb_year_weather.py`.

## How to run

### From anywhere (GitHub Pages)

After Pages is enabled (Settings → Pages → Source: **GitHub Actions**), every
push to `main` that touches `dashboard/` publishes the site to:

**https://rbhun.github.io/weather-comparator/**

You can also trigger a publish manually: Actions → **Deploy dashboard to GitHub
Pages** → Run workflow.

The published site is a normal HTTPS URL — open it on a phone, tablet, or any
laptop. Offline `file://` use below still works for race-day with no network.
Visitor counts (GoatCounter) are sent only from the online Pages URL.

Caveats:

- This repo is **private**. GitHub Pages on private repos needs Pro/Team (or
  make the repo public). The Pages URL itself is typically **public** once
  published — do not put secrets in `data.json`.
- First enable requires a one-time Pages source selection; the workflow cannot
  turn Pages on by itself.

### Local / offline (`file://`)

1. Prepare payload:

   ```bash
   PYTHONPATH="/workspace/src:/workspace" \
     python3 -m pmc.report \
     --input "/workspace/contracts/fixtures/data.json" \
     --output "/workspace/dashboard/data.json"
   ```

   This also writes `dashboard/data.js` so the page can load on `file://`
   without a file picker.

2. Open `dashboard/index.html` directly from the filesystem (no server).

3. (Optional) Refresh the offline chart-tile cache:

   ```bash
   python3 /workspace/dashboard/fetch_openseamap_tiles.py
   ```

   - Downloads OpenStreetMap (CARTO Voyager) base tiles and OpenSeaMap
     seamark tiles for lon `6.5..14.5`, lat `37.5..44.0` at zoom 7-9.
   - Enforces a 30 MB cap per layer by dropping the highest zoom first.

4. If you pick a different `data.json`, the page stores that payload in
   `localStorage` and reloads it after refresh. Use **Use bundled data** to
   go back. Browsers cannot keep a live File handle after reload on `file://`;
   remembering the JSON contents is the workaround. If a browser blocks
   `localStorage` on `file://`, pick the file again.

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
- The chart underlay is crowd-sourced OSM/OpenSeaMap information and is
  explicitly not suitable for navigation.
- The calm-risk slider is **hour of day** from August climatology (UTC hours
  0–23, shown in local time). It is not a race-calendar control for 18–23 Aug.
  Day-specific forecast maps need a live/previous-run payload, which this
  fixture does not contain.
- Each graph has +, −, and A (show all) buttons. Click-drag zoom still works;
  A resets to the full domain.
- The live dashboard field is currently August 2017 IFS 9 km analysis, 10 m
  wind. The fetcher only requests sea cells, so islands are empty. Open-Meteo
  can serve 10 m and 100 m over land if asked; 100 m is not boat-level.
  Grey on the map is no data, not calm. Some sea cells are still empty
  because 2017 is not fully downloaded.
