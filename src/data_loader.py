"""Dataset container and validated CSV import.

The loader makes **no assumptions** about the coordinate system, distance unit,
time unit or sampling rate of an uploaded file. The user declares them:

* column mapping (participant, timestamp, x/y/z, qx/qy/qz/qw, condition),
* time unit (s, ms, us, ns or a datetime column),
* position unit (converted to metres, or "arbitrary units" left unchanged),
* the vertical axis (defines the horizontal plane used for occupancy / top-down plots),
* the local forward axis (only used to draw gaze arrows, never for metrics).

Quaternions are mapped *per component*, so the component order of the source
file is unambiguous. The angle between two orientations does not depend on the
handedness / active-passive convention, so no further declaration is required
for the rotation metrics.

Uploaded and synthetic data live in separate :class:`Dataset` objects and are
never concatenated.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from .features import KEY_COLS, POS_COLS, QUAT_COLS

CANONICAL_COLS = KEY_COLS + ["timestamp"] + POS_COLS + QUAT_COLS

TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "datetime": None}
POSITION_UNITS = {"m": 1.0, "cm": 0.01, "mm": 1e-3, "ft": 0.3048, "in": 0.0254, "arbitrary units": 1.0}


def fingerprint_frame(df: pd.DataFrame) -> str:
    """Stable content hash of a table (used as cache key and for reproducibility notes)."""
    h = pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).to_numpy().tobytes()
    return hashlib.sha256(h + "|".join(map(str, df.columns)).encode()).hexdigest()


@dataclass
class Dataset:
    """A tidy 6-DoF table plus the metadata required to interpret it."""

    df: pd.DataFrame
    source: str                                   # "synthetic" | "uploaded"
    name: str
    has_rotation: bool = True
    has_condition: bool = True
    up_axis: str = "z"
    unit_label: str = "m"
    arena_bounds: tuple[float, float, float, float] | None = None
    forward_axis: str = "+x"
    meta: dict = field(default_factory=dict)

    @property
    def dataset_id(self) -> str:
        """Unique id used for caching. Includes the source so datasets can never be confused."""
        return f"{self.source}:{self.meta.get('fingerprint', 'nofp')[:16]}:{self.up_axis}:{self.unit_label}"

    @property
    def is_synthetic(self) -> bool:
        return self.source == "synthetic"


@dataclass
class ColumnMapping:
    participant: str | None            # None -> single participant "P01"
    timestamp: str
    x: str
    y: str
    z: str
    qx: str | None = None
    qy: str | None = None
    qz: str | None = None
    qw: str | None = None
    condition: str | None = None

    def quat_columns(self) -> list[str | None]:
        return [self.qx, self.qy, self.qz, self.qw]


@dataclass
class ImportSettings:
    time_unit: str = "s"
    position_unit: str = "m"
    up_axis: str = "z"
    forward_axis: str = "+x"


@dataclass
class ImportReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def read_table(source) -> pd.DataFrame:
    """Read a CSV from a path, bytes or file-like object; auto-detects the delimiter if needed."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    pos = source.tell() if hasattr(source, "tell") else None
    df = pd.read_csv(source)
    if df.shape[1] == 1:                           # probably ; or \t delimited
        if pos is not None:
            source.seek(pos)
        df = pd.read_csv(source, sep=None, engine="python")
    return df


def guess_mapping(columns: list[str]) -> dict[str, str | None]:
    """Heuristic column suggestions (user can always override)."""
    low = {c: c.lower().replace(" ", "_") for c in columns}

    def find(*pats, exclude=()):
        for c, l in low.items():
            if any(p == l or l.endswith(p) or l.startswith(p) for p in pats) and not any(e in l for e in exclude):
                return c
        for c, l in low.items():
            if any(p in l for p in pats) and not any(e in l for e in exclude):
                return c
        return None

    return {
        "participant": find("participant", "subject", "subj", "sub", "user_id", "user", "pid", "uid", "id"),
        "timestamp": find("timestamp", "time", "t"),
        "x": find("position_x", "pos_x", "px", "x", exclude=("quat", "rot")),
        "y": find("position_y", "pos_y", "py", "y", exclude=("quat", "rot")),
        "z": find("position_z", "pos_z", "pz", "z", exclude=("quat", "rot")),
        "qx": find("quaternion_x", "quat_x", "qx", "rot_x"),
        "qy": find("quaternion_y", "quat_y", "qy", "rot_y"),
        "qz": find("quaternion_z", "quat_z", "qz", "rot_z"),
        "qw": find("quaternion_w", "quat_w", "qw", "rot_w"),
        "condition": find("condition", "cond", "group", "grp", "task", "scenario", "arm", "treatment"),
    }


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_table(raw: pd.DataFrame, mapping: ColumnMapping, settings: ImportSettings,
                name: str = "uploaded.csv") -> tuple[Dataset | None, ImportReport]:
    """Validate and convert a raw table to a :class:`Dataset`.

    Returns ``(None, report)`` when ``report.errors`` is non-empty.
    """
    rep = ImportReport()
    cols = set(raw.columns)

    # ---- mapping validation ---------------------------------------------------------------
    required = {"timestamp": mapping.timestamp, "x": mapping.x, "y": mapping.y, "z": mapping.z}
    for k, c in required.items():
        if not c or c not in cols:
            rep.errors.append(f"Required column for '{k}' is not selected or not found in the file.")
    q_given = [c for c in mapping.quat_columns() if c]
    if 0 < len(q_given) < 4:
        rep.errors.append("Quaternion mapping is incomplete: map all four components (x, y, z, w) or none.")
    for c in q_given:
        if c not in cols:
            rep.errors.append(f"Quaternion column {c!r} not found.")
    used = [c for c in [mapping.participant, mapping.timestamp, mapping.x, mapping.y, mapping.z, *q_given,
                        mapping.condition] if c]
    dup = sorted({c for c in used if used.count(c) > 1})
    if dup:
        rep.errors.append(f"Each source column may be used once, but these are mapped twice: {', '.join(dup)}.")
    if settings.time_unit not in TIME_UNITS:
        rep.errors.append(f"Unknown time unit {settings.time_unit!r}.")
    if settings.position_unit not in POSITION_UNITS:
        rep.errors.append(f"Unknown position unit {settings.position_unit!r}.")
    if rep.errors:
        return None, rep
    has_rot = len(q_given) == 4

    # ---- build canonical table -----------------------------------------------------------------
    out = pd.DataFrame(index=raw.index)
    if mapping.participant:
        pid = raw[mapping.participant]
        n_missing = int(pid.isna().sum())
        if n_missing:
            rep.errors.append(f"{n_missing} rows have an empty participant ID; fix the file or map another column.")
        out["participant_id"] = pid.astype(str)
    else:
        out["participant_id"] = "P01"
        rep.warnings.append("No participant column mapped: all rows are treated as one participant 'P01'.")
    if mapping.condition:
        cond = raw[mapping.condition]
        n_missing = int(cond.isna().sum())
        if n_missing:
            rep.warnings.append(f"{n_missing} rows have no condition label and are labelled 'unlabelled'.")
        out["condition"] = cond.fillna("unlabelled").astype(str)
    else:
        out["condition"] = "All"
        rep.warnings.append("No condition column mapped: group comparisons are disabled (single group 'All').")
    if rep.errors:
        return None, rep

    # position ---------------------------------------------------------------------------------------
    scale = POSITION_UNITS[settings.position_unit]
    for target, src in zip(POS_COLS, (mapping.x, mapping.y, mapping.z)):
        num = _numeric(raw[src])
        n_bad = int(num.isna().sum() - raw[src].isna().sum())
        if n_bad:
            rep.warnings.append(f"Column {src!r}: {n_bad} non-numeric values were converted to missing.")
        if num.notna().sum() == 0:
            rep.errors.append(f"Column {src!r} contains no numeric values.")
        out[target] = num * scale
    # quaternion -------------------------------------------------------------------------------------
    for target, src in zip(QUAT_COLS, mapping.quat_columns()):
        out[target] = _numeric(raw[src]) if src else np.nan
    if has_rot:
        q = out[QUAT_COLS].to_numpy(float)
        fin = np.isfinite(q).all(axis=1)
        if fin.sum() == 0:
            rep.errors.append("Quaternion columns contain no numeric values.")
        else:
            nrm = np.linalg.norm(q[fin], axis=1)
            med = float(np.median(nrm))
            if abs(med - 1.0) > 0.05:
                rep.warnings.append(f"Median quaternion norm is {med:.3g}, not 1. Check that these are quaternions "
                                    "(not angles / Euler values). They will be renormalised during preprocessing.")
            if (nrm == 0).any():
                rep.warnings.append(f"{int((nrm == 0).sum())} zero-norm quaternions (orientation undefined).")
    else:
        rep.warnings.append("No quaternion columns mapped: rotation-based metrics are disabled "
                            "(angular velocity, cumulative rotation, gaze arrows).")
    # timestamp ----------------------------------------------------------------------------------------
    key = [out["participant_id"], out["condition"]]
    if settings.time_unit == "datetime":
        ts = pd.to_datetime(raw[mapping.timestamp], errors="coerce", utc=True)
        n_bad = int(ts.isna().sum() - raw[mapping.timestamp].isna().sum())
        gmin = ts.groupby(key).transform("min")
        t = (ts - gmin).dt.total_seconds()
    else:
        raw_t = _numeric(raw[mapping.timestamp])
        n_bad = int(raw_t.isna().sum() - raw[mapping.timestamp].isna().sum())
        gmin = raw_t.groupby(key).transform("min")
        t = (raw_t - gmin) * TIME_UNITS[settings.time_unit]
    if n_bad:
        rep.warnings.append(f"Timestamp column: {n_bad} unparseable values were converted to missing.")
    if t.notna().sum() == 0:
        rep.errors.append("Timestamp column contains no valid values for the chosen time unit.")
    out["timestamp"] = t.astype(float)
    if rep.errors:
        return None, rep

    out = out[CANONICAL_COLS].reset_index(drop=True)     # file row order is preserved on purpose

    # ---- plausibility checks -----------------------------------------------------------------------
    rates, durations, small = [], [], 0
    for _, g in out.groupby(KEY_COLS, sort=True):
        tt = np.sort(g["timestamp"].dropna().to_numpy())
        if tt.size < 3:
            small += 1
            continue
        d = np.diff(tt)
        d = d[d > 0]
        if d.size:
            rates.append(1.0 / np.median(d))
        durations.append(tt[-1] - tt[0])
    n_rec = out.groupby(KEY_COLS).ngroups
    if not rates:
        rep.errors.append("No recording has at least 3 samples with increasing timestamps - "
                          "check the participant / condition / timestamp mapping and the time unit.")
        return None, rep
    med_rate = float(np.median(rates))
    if med_rate < 1:
        rep.warnings.append(f"Estimated sampling rate is only {med_rate:.3g} Hz - is the time unit correct?")
    if med_rate > 2000:
        rep.warnings.append(f"Estimated sampling rate is {med_rate:.3g} Hz - is the time unit correct?")
    if small:
        rep.warnings.append(f"{small} recording(s) have fewer than 3 usable samples and cannot be analysed.")
    n_cond = out["condition"].nunique()
    if mapping.condition and n_cond > 2:
        rep.warnings.append(f"{n_cond} conditions found: the statistics page compares two selected conditions.")
    if out.duplicated(subset=KEY_COLS + ["timestamp"]).any():
        rep.warnings.append("Duplicate timestamps found within a recording. If a participant has several trials "
                            "under the same ID/condition, add a distinguishing value to the condition column.")
    rep.info = {
        "rows": int(len(out)), "participants": int(out["participant_id"].nunique()), "recordings": int(n_rec),
        "conditions": sorted(out["condition"].unique().tolist()), "median_rate_hz": med_rate,
        "median_duration_s": float(np.median(durations)) if durations else np.nan,
    }
    meta = {
        "fingerprint": fingerprint_frame(out) + hashlib.sha256(repr((asdict(mapping), asdict(settings))).encode()).hexdigest()[:8],
        "mapping": asdict(mapping), "settings": asdict(settings), "source_file": name,
    }
    ds = Dataset(
        df=out, source="uploaded", name=name, has_rotation=has_rot, has_condition=bool(mapping.condition),
        up_axis=settings.up_axis, unit_label="m" if settings.position_unit != "arbitrary units" else "units",
        arena_bounds=None, forward_axis=settings.forward_axis, meta=meta,
    )
    return ds, rep


def load_csv(source, mapping: ColumnMapping, settings: ImportSettings, name: str = "uploaded.csv"):
    """Read ``source`` and call :func:`parse_table`."""
    try:
        raw = read_table(source)
    except Exception as exc:                      # malformed file: report, do not crash the app
        rep = ImportReport(errors=[f"Could not read the file as CSV: {exc}"])
        return None, rep
    return parse_table(raw, mapping, settings, name)
