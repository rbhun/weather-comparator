# weather-comparator

## Secrets (API key + real polar)

The Open-Meteo API key and the proprietary boat polar are **not** meant to be
in git. They are gitignored:

| Secret | Where it lives |
| --- | --- |
| `OPENMETEO_API_KEY` | `.env` (local) or env / Cursor / GitHub Actions secret |
| Boat polar | `config/polar/boat.pol` or secret `BOAT_POLAR` (full file body) |

```bash
cp .env.example .env   # then paste the key
python3 scripts/materialize_secrets.py   # writes .env + polar from env secrets
```

The fabricated fixture `contracts/fixtures/polar_52ft.pol` stays in the repo for
tests. The published dashboard payload does not embed the polar table.

**If this repo is public:** rotate the Open-Meteo API key. An older commit still
tracked `.env`, so the previous key is in git history until that history is
rewritten or the key is revoked. Scrub history (e.g. `git filter-repo`) if an
old polar file must not remain in past commits.

## Dashboard (open from anywhere)

The tactical dashboard lives in `dashboard/`. It remains a self-contained
offline HTML app (`file://`), and is also published via GitHub Pages:

**https://rbhun.github.io/weather-comparator/**

Enable once under Settings → Pages → Source: **GitHub Actions**. Deploys run
from `.github/workflows/deploy-dashboard.yml` on pushes to `main` that change
`dashboard/`. See `dashboard/README.md` for local rebuild steps. The Pages site
can be public while this repo stays private, or the repo can be public once the
secrets above are untracked and the API key is rotated.

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