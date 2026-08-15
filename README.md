# weather-comparator

## Dashboard (open from anywhere)

The tactical dashboard lives in `dashboard/`. It remains a self-contained
offline HTML app (`file://`), and is also published via GitHub Pages:

**https://rbhun.github.io/weather-comparator/**

Enable once under Settings → Pages → Source: **GitHub Actions**. Deploys run
from `.github/workflows/deploy-dashboard.yml` on pushes to `main` that change
`dashboard/`. See `dashboard/README.md` for local rebuild steps and caveats
(private-repo Pages eligibility; published site is public).

## Coastline data attribution

Primary coastline polygons are derived from OpenStreetMap land polygons
(`land-polygons-complete-4326`, WGS84) and are used under the
Open Database License (ODbL).

Source: https://osmdata.openstreetmap.de/data/land-polygons.html

## Leg-1 interpretation caveat

Leg 1 is predominantly upwind in this setup. The fixed-route follower therefore
uses polar VMG angles and does not model shift-playing tactics. As a result,
leg-1 route deltas can undervalue options that benefit from larger wind
oscillations (typically tighter coastal lanes).