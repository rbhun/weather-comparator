"""Observation ingest with format auto-detect and parse reports."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .conventions import wind_to_uv_ms


@dataclass
class IngestReport:
    path: str
    detected_format: str
    obs_class: str
    instrument: str | None
    n_cells: int
    n_rejected_qc: int
    messages: list[str] = field(default_factory=list)
    dry_run: bool = False
    source_file_hash: str = ""
    cells: pd.DataFrame | None = None


_METAR_RE = re.compile(
    r"^(?P<id>[A-Z]{4})\s+(?P<dom>\d{2})(?P<hour>\d{2})(?P<minute>\d{2})Z\s+"
    r"(?P<wind>(?P<dir>\d{3}|VRB)(?P<spd>\d{2,3})(?:G\d{2,3})?KT)\s+"
    r".*?(?P<qnh>Q(?P<hpa>\d{4}))",
    re.IGNORECASE,
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_format(path: Path) -> str:
    """Auto-detect observation file format from extension and content sniff."""
    suffix = path.suffix.lower()
    if suffix in {".nc", ".nc4", ".netcdf"}:
        return "netcdf"
    if suffix in {".grib", ".grib2", ".grb", ".grb2"}:
        return "grib"
    if suffix in {".bufr", ".bfr"}:
        return "bufr"
    if suffix in {".json"}:
        return "json"
    if suffix in {".csv", ".txt", ".metar", ".synop"}:
        head = path.read_text(encoding="utf-8", errors="replace")[:4000]
        if "METAR" in head.upper() or _METAR_RE.search(head):
            return "metar"
        if "," in head and ("lat" in head.lower() or "wind" in head.lower()):
            return "csv"
        return "text"
    # Sniff magic
    raw = path.read_bytes()[:8]
    if raw.startswith(b"CDF") or raw.startswith(b"\x89HDF"):
        return "netcdf"
    if raw.startswith(b"GRIB"):
        return "grib"
    return "unknown"


def ingest_observations(path: Path, *, dry_run: bool = False) -> IngestReport:
    """Ingest an operator-supplied observation file.

    Returns a parse report. When ``dry_run`` is True, cells are parsed but the
    caller must not persist them.
    """
    path = Path(path)
    fmt = detect_format(path)
    digest = file_sha256(path)
    if fmt == "netcdf":
        return _ingest_netcdf(path, digest, dry_run=dry_run)
    if fmt == "json":
        return _ingest_json(path, digest, dry_run=dry_run)
    if fmt == "csv":
        return _ingest_csv(path, digest, dry_run=dry_run)
    if fmt == "metar":
        return _ingest_metar(path, digest, dry_run=dry_run)
    if fmt == "grib":
        return _ingest_grib(path, digest, dry_run=dry_run)
    if fmt == "bufr":
        return _ingest_bufr(path, digest, dry_run=dry_run)
    return IngestReport(
        path=str(path),
        detected_format=fmt,
        obs_class="unknown",
        instrument=None,
        n_cells=0,
        n_rejected_qc=0,
        messages=[f"Unsupported or unknown format: {fmt}"],
        dry_run=dry_run,
        source_file_hash=digest,
    )


def _ingest_netcdf(path: Path, digest: str, *, dry_run: bool) -> IngestReport:
    import xarray as xr

    ds = xr.open_dataset(path)
    try:
        instrument = str(ds.attrs.get("instrument") or ds.attrs.get("instrument_id") or "")
        if not instrument:
            raise ValueError("NetCDF missing global attr 'instrument'.")
        required = ("lat", "lon", "time", "wind_speed", "wind_dir")
        for name in required:
            if name not in ds:
                raise ValueError(f"NetCDF missing variable '{name}'.")
        lat = np.asarray(ds["lat"].values, dtype=float).reshape(-1)
        lon = np.asarray(ds["lon"].values, dtype=float).reshape(-1)
        time = pd.to_datetime(np.asarray(ds["time"].values).reshape(-1), utc=True)
        wspd = np.asarray(ds["wind_speed"].values, dtype=float).reshape(-1)
        wdir = np.asarray(ds["wind_dir"].values, dtype=float).reshape(-1)
        n = lat.size
        for arr in (lon, time, wspd, wdir):
            if arr.size != n:
                raise ValueError("Per-cell fields must share the same length.")

        qc = np.ones(n, dtype=bool)
        for flag_name in ("quality_flag", "land_flag", "ice_flag", "rain_flag"):
            if flag_name in ds:
                vals = np.asarray(ds[flag_name].values).reshape(-1)
                # nonzero / true means reject
                qc &= vals == 0

        u, v = wind_to_uv_ms(wspd, wdir, instrument=instrument)
        rejected = int((~qc).sum())
        keep = qc
        frame = pd.DataFrame(
            {
                "lat": lat[keep].astype(np.float32),
                "lon": lon[keep].astype(np.float32),
                "time": time[keep].tz_convert("UTC").tz_localize(None),
                "obs_u10": u[keep].astype(np.float32),
                "obs_v10": v[keep].astype(np.float32),
                "obs_speed_ms": wspd[keep].astype(np.float32),
                "instrument": instrument,
                "obs_class": "scatterometer",
            }
        )
        return IngestReport(
            path=str(path),
            detected_format="netcdf",
            obs_class="scatterometer",
            instrument=instrument,
            n_cells=int(len(frame)),
            n_rejected_qc=rejected,
            messages=[f"Ingested {len(frame)} cells from {instrument}"],
            dry_run=dry_run,
            source_file_hash=digest,
            cells=frame,
        )
    finally:
        ds.close()


def _ingest_json(path: Path, digest: str, *, dry_run: bool) -> IngestReport:
    raw = json.loads(path.read_text(encoding="utf-8"))
    obs_class = str(raw.get("obs_class", "scatterometer"))
    instrument = raw.get("instrument")
    cells = raw.get("cells", [])
    if obs_class == "scatterometer":
        if not instrument:
            raise ValueError("JSON scatterometer ingest requires 'instrument'.")
        rows = []
        rejected = 0
        for cell in cells:
            if any(cell.get(f) for f in ("land_flag", "ice_flag", "rain_flag", "quality_flag")):
                rejected += 1
                continue
            u, v = wind_to_uv_ms(
                cell["wind_speed"], cell["wind_dir"], instrument=str(instrument)
            )
            rows.append(
                {
                    "lat": float(cell["lat"]),
                    "lon": float(cell["lon"]),
                    "time": pd.Timestamp(cell["time"], tz="UTC").tz_localize(None),
                    "obs_u10": float(u),
                    "obs_v10": float(v),
                    "obs_speed_ms": float(cell["wind_speed"]),
                    "instrument": str(instrument),
                    "obs_class": "scatterometer",
                }
            )
        frame = pd.DataFrame(rows)
        return IngestReport(
            path=str(path),
            detected_format="json",
            obs_class=obs_class,
            instrument=str(instrument),
            n_cells=len(frame),
            n_rejected_qc=rejected,
            messages=[f"JSON scatterometer cells: {len(frame)}"],
            dry_run=dry_run,
            source_file_hash=digest,
            cells=frame,
        )
    if obs_class == "land_station":
        rows = []
        for cell in cells:
            rows.append(
                {
                    "station_id": str(cell["station_id"]),
                    "lat": float(cell["lat"]),
                    "lon": float(cell["lon"]),
                    "time": pd.Timestamp(cell["time"], tz="UTC").tz_localize(None),
                    "mslp_hpa": float(cell["mslp_hpa"]),
                    # Wind retained for timing diagnostics only — never calibration.
                    "obs_u10": float(cell.get("obs_u10", np.nan)),
                    "obs_v10": float(cell.get("obs_v10", np.nan)),
                    "instrument": "metar_synop",
                    "obs_class": "land_station",
                }
            )
        frame = pd.DataFrame(rows)
        return IngestReport(
            path=str(path),
            detected_format="json",
            obs_class=obs_class,
            instrument="metar_synop",
            n_cells=len(frame),
            n_rejected_qc=0,
            messages=[f"JSON land-station rows: {len(frame)}"],
            dry_run=dry_run,
            source_file_hash=digest,
            cells=frame,
        )
    raise ValueError(f"Unsupported JSON obs_class: {obs_class}")


def _ingest_csv(path: Path, digest: str, *, dry_run: bool) -> IngestReport:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "instrument" not in cols:
        raise ValueError("CSV must include an instrument column.")
    instrument = str(df[cols["instrument"]].iloc[0])
    lat = df[cols["lat"]].to_numpy(dtype=float)
    lon = df[cols["lon"]].to_numpy(dtype=float)
    time = pd.to_datetime(df[cols["time"]], utc=True)
    wspd = df[cols.get("wind_speed", cols.get("wind_speed_ms"))].to_numpy(dtype=float)
    wdir = df[cols["wind_dir"]].to_numpy(dtype=float)
    u, v = wind_to_uv_ms(wspd, wdir, instrument=instrument)
    frame = pd.DataFrame(
        {
            "lat": lat.astype(np.float32),
            "lon": lon.astype(np.float32),
            "time": time.tz_convert("UTC").tz_localize(None),
            "obs_u10": u.astype(np.float32),
            "obs_v10": v.astype(np.float32),
            "obs_speed_ms": wspd.astype(np.float32),
            "instrument": instrument,
            "obs_class": "scatterometer",
        }
    )
    return IngestReport(
        path=str(path),
        detected_format="csv",
        obs_class="scatterometer",
        instrument=instrument,
        n_cells=len(frame),
        n_rejected_qc=0,
        messages=[f"CSV cells: {len(frame)}"],
        dry_run=dry_run,
        source_file_hash=digest,
        cells=frame,
    )


def _ingest_metar(path: Path, digest: str, *, dry_run: bool) -> IngestReport:
    """Parse coded METAR lines (degrees true, knots). Spoken ATIS is ignored."""
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    rejected = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("METAR "):
            line = line[6:].strip()
        m = _METAR_RE.search(line)
        if not m:
            rejected += 1
            continue
        if m.group("dir") == "VRB":
            rejected += 1
            continue
        # Coded METAR: direction degrees true, speed whole knots.
        dir_true = float(m.group("dir"))
        spd_kt = float(m.group("spd"))
        hpa = float(m.group("hpa"))
        # Date: use file mtime day-of-month if not provided; METAR has day-of-month only.
        # For fixture use, require ISO comment header `# date: YYYY-MM-DD`.
        date_match = re.search(r"#\s*date:\s*(\d{4}-\d{2}-\d{2})", text)
        if not date_match:
            raise ValueError("METAR file requires '# date: YYYY-MM-DD' header for month/year.")
        day = int(m.group("dom"))
        hour = int(m.group("hour"))
        minute = int(m.group("minute"))
        base = pd.Timestamp(date_match.group(1), tz="UTC")
        # Replace day-of-month from METAR
        ts = base.replace(day=day, hour=hour, minute=minute, second=0)
        from contracts.schemas import tws_twd_to_uv

        u, v = tws_twd_to_uv(spd_kt, dir_true)
        rows.append(
            {
                "station_id": m.group("id").upper(),
                "lat": np.nan,
                "lon": np.nan,
                "time": ts.tz_localize(None),
                "mslp_hpa": hpa,
                "obs_u10": float(u),
                "obs_v10": float(v),
                "instrument": "metar_synop",
                "obs_class": "land_station",
            }
        )
    frame = pd.DataFrame(rows)
    return IngestReport(
        path=str(path),
        detected_format="metar",
        obs_class="land_station",
        instrument="metar_synop",
        n_cells=len(frame),
        n_rejected_qc=rejected,
        messages=[
            f"METAR rows: {len(frame)} (wind retained for timing only; not for calibration)",
            "METAR wind is 10-min mean quantised to 10° and whole knots.",
        ],
        dry_run=dry_run,
        source_file_hash=digest,
        cells=frame,
    )


def _ingest_grib(path: Path, digest: str, *, dry_run: bool) -> IngestReport:
    try:
        import cfgrib  # noqa: F401
        import xarray as xr
    except ImportError as exc:
        return IngestReport(
            path=str(path),
            detected_format="grib",
            obs_class="scatterometer",
            instrument=None,
            n_cells=0,
            n_rejected_qc=0,
            messages=[f"cfgrib/eccodes not installed: {exc}"],
            dry_run=dry_run,
            source_file_hash=digest,
        )
    ds = xr.open_dataset(path, engine="cfgrib")
    try:
        # Expect operator to set instrument attr when converting; otherwise refuse.
        instrument = str(ds.attrs.get("instrument", ""))
        if not instrument:
            return IngestReport(
                path=str(path),
                detected_format="grib",
                obs_class="scatterometer",
                instrument=None,
                n_cells=0,
                n_rejected_qc=0,
                messages=["GRIB missing instrument attr; refuse to guess direction convention."],
                dry_run=dry_run,
                source_file_hash=digest,
            )
        # Minimal path: treat as already-projected lat/lon wind speed/dir if present.
        raise ValueError("GRIB scatterometer ingest requires a documented mapping; use NetCDF.")
    finally:
        ds.close()


def _ingest_bufr(path: Path, digest: str, *, dry_run: bool) -> IngestReport:
    try:
        import eccodes  # noqa: F401
    except ImportError as exc:
        return IngestReport(
            path=str(path),
            detected_format="bufr",
            obs_class="scatterometer",
            instrument=None,
            n_cells=0,
            n_rejected_qc=0,
            messages=[f"eccodes not installed: {exc}"],
            dry_run=dry_run,
            source_file_hash=digest,
        )
    return IngestReport(
        path=str(path),
        detected_format="bufr",
        obs_class="scatterometer",
        instrument=None,
        n_cells=0,
        n_rejected_qc=0,
        messages=["BUFR ingest requires per-product descriptor maps; convert to NetCDF first."],
        dry_run=dry_run,
        source_file_hash=digest,
    )
