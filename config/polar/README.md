# Polar files (gitignored)

The real Expedition polar (`chocolate3.pol`) is **not** in git. Keep it local
or inject it from a secret before building.

## Local

```bash
cp /path/to/Chocolate3.pol config/polar/chocolate3.pol
```

`config/course.yaml` expects `config/polar/chocolate3.pol`.

## From a secret (Cursor / CI)

Set secret `CHOCOLATE3_POLAR` to the full `.pol` file contents, then:

```bash
python3 scripts/materialize_secrets.py
```

That writes `config/polar/chocolate3.pol` when the file is missing.

## Public vs private

- `contracts/fixtures/polar_52ft.pol` — fabricated fixture; safe to publish.
- `config/polar/chocolate3.pol` — proprietary VPP; must stay out of the public tree.
