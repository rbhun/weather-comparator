# Contract fixtures (committed on purpose)

These files are intentionally committed so every module can develop in
parallel against frozen interfaces.

- `wind_small.zarr`: synthetic 30-day hourly wind cube (FIXTURE ONLY).
- `polar_52ft.pol`: fabricated Expedition-style polar (FIXTURE ONLY).
- `climatology_small.nc`: derived from the synthetic wind cube.
- `data.json`: C7 dashboard payload fixture.
- `sar_scenes_small.zarr`: synthetic Sentinel-1 L3 scalar-speed scenes (C10).
- `sar_shadow_test.json`: optional dashboard section from the SAR fixture.

Do not delete this directory. Removing it breaks parallel development
for the other agents.
