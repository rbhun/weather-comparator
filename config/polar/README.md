# Polar files (gitignored)

The real boat polar (`boat.pol`) is **not** in git. Keep it local or inject it
from a secret before building.

## Local

```bash
cp /path/to/your.pol config/polar/boat.pol
```

`config/course.yaml` expects `config/polar/boat.pol`.

## From a secret (Cursor / CI)

Set secret `BOAT_POLAR` to the full `.pol` file contents, then:

```bash
python3 scripts/materialize_secrets.py
```

That writes `config/polar/boat.pol` when the file is missing.

## Public vs private

- `contracts/fixtures/polar_52ft.pol` — fabricated fixture; safe to publish.
- `config/polar/boat.pol` — proprietary VPP; must stay out of the public tree.
