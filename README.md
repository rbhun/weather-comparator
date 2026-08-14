# weather-comparator

Palermo–Montecarlo weather analysis: climatology, fixed-route comparison, and
an offline tactical dashboard.

## Local copy on a Mac

The repo lives at [github.com/rbhun/weather-comparator](https://github.com/rbhun/weather-comparator).
A cloud agent cannot write files onto your Mac; clone it locally instead.

```bash
cd ~
git clone https://github.com/rbhun/weather-comparator.git
cd weather-comparator
```

If you do not already have Python 3.11+:

```bash
brew install python@3.12
```

Then create the local environment:

```bash
./scripts/setup_local.sh
source .venv/bin/activate
```

Smoke-check the fixture pipeline (no network required):

```bash
pytest
python scripts/walking_skeleton.py
python -m pmc.report \
  --input contracts/fixtures/data.json \
  --output dashboard/data.json
```

Open `dashboard/index.html` from the Finder or a browser. It is a single
offline HTML file plus `data.json` — no server and no internet.

In Cursor: **File → Open Folder…** and choose `~/weather-comparator`.

## Coastline data attribution

Primary coastline polygons are derived from OpenStreetMap land polygons
(`land-polygons-complete-4326`, WGS84) and are used under the
Open Database License (ODbL).

Source: https://osmdata.openstreetmap.de/data/land-polygons.html
