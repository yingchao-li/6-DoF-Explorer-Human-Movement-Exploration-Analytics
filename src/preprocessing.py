"""Data-quality inspection and (explicit, logged) preprocessing of 6-DoF recordings.

Design principles
-----------------
1. **Inspect first, never silently fix.** :func:`inspect_quality` only measures.
2. **Every modification is an explicit, logged operation.** :func:`preprocess`
   returns an *operations table* (rows before / after / affected + a plain
   language description of the effect on the analysis).
3. **Detection is separate from repair.** Discontinuities and outliers are
   *flagged*; they are only altered when ``repair_spikes`` is switched on and
   then only for isolated single-sample position spikes.

Operation order
---------------
sort by time -> drop rows with missing pose -> drop duplicate timestamps ->
quaternion renormalisation -> (flag discontinuities / outliers) ->
optional spike repair -> optional resampling -> optional smoothing.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

from .features import KEY_COLS, POS_COLS, QUAT_COLS, enforce_quaternion_continuity, normalize_quaternions


@dataclass(frozen=True)
class PreprocessConfig:
    """Switches and thresholds of the preprocessing pipeline."""

    drop_missing: bool = True
    drop_duplicate_timestamps: bool = True
    renormalize_quaternions: bool = True
    quat_norm_tol: float = 1e-3
    gap_factor: float = 10.0          # dt > gap_factor * median(dt) is a tracking gap
    max_speed: float = 6.0            # plausibility limit (data units / s) for discontinuities
    outlier_iqr_k: float = 3.0        # Tukey fence multiplier on step speed
    repair_spikes: bool = False       # replace isolated spike samples by linear interpolation
    resample_hz: float | None = None  # None = keep original timestamps
    smooth_sigma_s: float = 0.0       # Gaussian smoothing width (s); 0 = off

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityReport:
    """Result of :func:`inspect_quality` / :func:`preprocess`."""

    per_recording: pd.DataFrame
    totals: dict
    warnings: list[str] = field(default_factory=list)
    operations: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_recording_after: pd.DataFrame | None = None
    totals_after: dict | None = None


# --------------------------------------------------------------------------- #
# Step-wise plausibility analysis
# --------------------------------------------------------------------------- #
def _step_analysis(t: np.ndarray, pos: np.ndarray, cfg: PreprocessConfig) -> dict:
    """Flags for a *sorted, finite, strictly increasing* recording."""
    n = t.size
    res = {
        "flag_discontinuity": np.zeros(n, bool),
        "flag_outlier": np.zeros(n, bool),
        "flag_spike": np.zeros(n, bool),
        "gap_step": np.zeros(max(n - 1, 0), bool),
        "median_dt": np.nan,
        "dt": np.zeros(max(n - 1, 0)),
    }
    if n < 3:
        if n == 2:
            res["median_dt"] = float(t[1] - t[0])
            res["dt"] = np.diff(t)
        return res
    dt = np.diff(t)
    med = float(np.median(dt))
    gap = dt > cfg.gap_factor * med
    dist = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    speed = dist / dt
    disc = (speed > cfg.max_speed) & ~gap

    valid_speed = speed[~gap]
    if valid_speed.size >= 4:
        q1, q3 = np.percentile(valid_speed, [25, 75])
        fence = q3 + cfg.outlier_iqr_k * (q3 - q1)
        out = (speed > fence) & ~gap
    else:
        out = np.zeros_like(disc)

    spike = np.zeros(n, bool)
    both = disc[:-1] & disc[1:]
    idx = np.flatnonzero(both) + 1                      # sample between the two steps
    if idx.size:
        ret = np.linalg.norm(pos[idx + 1] - pos[idx - 1], axis=1)
        leg = np.minimum(dist[idx - 1], dist[idx])
        spike[idx[ret <= 0.5 * leg]] = True             # excursion that returns to its origin

    res.update(
        flag_discontinuity=np.concatenate([[False], disc]),
        flag_outlier=np.concatenate([[False], out]),
        flag_spike=spike,
        gap_step=gap,
        median_dt=med,
        dt=dt,
    )
    return res


def _clean_view(g: pd.DataFrame, has_rotation: bool) -> pd.DataFrame:
    """Sorted, finite, de-duplicated copy used for measuring (never returned to the user)."""
    cols = ["timestamp"] + POS_COLS + (QUAT_COLS if has_rotation else [])
    v = g[cols]
    ok = np.isfinite(v.to_numpy(dtype=float)).all(axis=1)
    if has_rotation:
        ok &= np.linalg.norm(v[QUAT_COLS].to_numpy(dtype=float), axis=1) > 0
    v = v[ok].sort_values("timestamp", kind="stable")
    return v[~v["timestamp"].duplicated(keep="first")]


def flag_samples(rec: pd.DataFrame, cfg: PreprocessConfig | None = None) -> pd.DataFrame:
    """Per-sample flags (``flag_discontinuity``, ``flag_outlier``, ``flag_spike``) of one recording.

    ``rec`` must contain ``timestamp`` and the position columns. The returned
    frame is sorted by time and de-duplicated (rows with non-finite values are
    ignored), so it can be plotted directly.
    """
    cfg = cfg or PreprocessConfig()
    v = _clean_view(rec, has_rotation=False)
    t = v["timestamp"].to_numpy(float)
    res = _step_analysis(t, v[POS_COLS].to_numpy(float), cfg) if t.size else None
    out = v[["timestamp"] + POS_COLS].reset_index(drop=True)
    for k in ("flag_discontinuity", "flag_outlier", "flag_spike"):
        out[k] = res[k] if res is not None else np.zeros(len(out), bool)
    return out


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
def _inspect_recording(g: pd.DataFrame, cfg: PreprocessConfig, has_rotation: bool) -> dict:
    t = g["timestamp"].to_numpy(float)
    pos = g[POS_COLS].to_numpy(float)
    t_ok = np.isfinite(t)
    pose_ok = np.isfinite(pos).all(axis=1)
    n_norm_bad = 0
    max_dev = np.nan
    if has_rotation:
        q = g[QUAT_COLS].to_numpy(float)
        q_fin = np.isfinite(q).all(axis=1)
        nrm = np.linalg.norm(q, axis=1)
        pose_ok &= q_fin & (nrm > 0)
        dev = np.abs(nrm - 1.0)
        n_norm_bad = int(np.sum(q_fin & (dev > cfg.quat_norm_tol)))
        max_dev = float(dev[q_fin].max()) if q_fin.any() else np.nan

    tt = t[t_ok]
    cols = ["timestamp"] + POS_COLS + (QUAT_COLS if has_rotation else [])
    row = {
        "n_rows": int(len(g)),
        "n_missing_rows": int(np.sum(~(t_ok & pose_ok))),
        "n_duplicate_timestamps": int(pd.Series(tt).duplicated().sum()),
        "n_duplicate_rows": int(g.duplicated(subset=cols).sum()),
        "n_nonmonotonic": int(np.sum(np.diff(tt) < 0)),
        "n_quat_norm_bad": n_norm_bad,
        "max_norm_dev": max_dev,
    }
    v = _clean_view(g, has_rotation)
    vt = v["timestamp"].to_numpy(float)
    if vt.size >= 3:
        res = _step_analysis(vt, v[POS_COLS].to_numpy(float), cfg)
        dt, gap = res["dt"], res["gap_step"]
        regular = dt[~gap]
        row.update(
            median_dt_s=res["median_dt"],
            rate_hz=1.0 / res["median_dt"],
            dt_cv=float(regular.std() / regular.mean()) if regular.size > 1 and regular.mean() > 0 else np.nan,
            n_gaps=int(gap.sum()),
            max_gap_s=float(dt[gap].max()) if gap.any() else 0.0,
            duration_s=float(vt[-1] - vt[0]),
            n_discontinuities=int(res["flag_discontinuity"].sum()),
            n_spikes=int(res["flag_spike"].sum()),
            n_outliers=int(res["flag_outlier"].sum()),
        )
    else:
        row.update(median_dt_s=np.nan, rate_hz=np.nan, dt_cv=np.nan, n_gaps=0, max_gap_s=0.0,
                   duration_s=float(vt[-1] - vt[0]) if vt.size == 2 else 0.0,
                   n_discontinuities=0, n_spikes=0, n_outliers=0)
    return row


_SUM_KEYS = ["n_rows", "n_missing_rows", "n_duplicate_timestamps", "n_duplicate_rows", "n_nonmonotonic",
             "n_quat_norm_bad", "n_gaps", "n_discontinuities", "n_spikes", "n_outliers"]


def _totals(per: pd.DataFrame) -> dict:
    if per.empty:
        return {k: 0 for k in _SUM_KEYS} | {"n_recordings": 0}
    tot = {k: int(per[k].sum()) for k in _SUM_KEYS}
    tot["n_recordings"] = int(len(per))
    tot["median_rate_hz"] = float(per["rate_hz"].median())
    tot["max_dt_cv"] = float(per["dt_cv"].max()) if per["dt_cv"].notna().any() else np.nan
    tot["max_gap_s"] = float(per["max_gap_s"].max())
    tot["max_norm_dev"] = float(per["max_norm_dev"].max()) if per["max_norm_dev"].notna().any() else np.nan
    tot["n_rows_with_issue"] = int(per[["n_missing_rows", "n_duplicate_timestamps", "n_nonmonotonic",
                                         "n_quat_norm_bad"]].sum().sum())
    return tot


def _warnings(tot: dict, cfg: PreprocessConfig) -> list[str]:
    w: list[str] = []
    if tot["n_missing_rows"]:
        w.append(f"{tot['n_missing_rows']} rows have missing/non-finite pose or timestamp.")
    if tot["n_duplicate_timestamps"]:
        w.append(f"{tot['n_duplicate_timestamps']} duplicate timestamps (dt = 0 would make speeds undefined).")
    if tot["n_nonmonotonic"]:
        w.append(f"{tot['n_nonmonotonic']} out-of-order timestamps; recordings will be sorted by time.")
    if tot["n_gaps"]:
        w.append(f"{tot['n_gaps']} tracking gaps (dt > {cfg.gap_factor:g}x median); steps across a gap are "
                 "excluded from speed / path / angular statistics.")
    if tot["n_discontinuities"]:
        w.append(f"{tot['n_discontinuities']} position discontinuities (step speed > {cfg.max_speed:g} units/s) are "
                 "flagged, not removed - path length and max speed are inflated unless spike repair is enabled.")
    if tot.get("n_quat_norm_bad"):
        w.append(f"{tot['n_quat_norm_bad']} quaternions deviate from unit norm by more than {cfg.quat_norm_tol:g}.")
    if tot.get("max_dt_cv") is not None and np.isfinite(tot.get("max_dt_cv", np.nan)) and tot["max_dt_cv"] > 0.1:
        w.append(f"Sampling is irregular (dt coefficient of variation up to {tot['max_dt_cv']:.2f}); all "
                 "derivatives use the actual timestamps.")
    return w


def inspect_quality(df: pd.DataFrame, cfg: PreprocessConfig | None = None, has_rotation: bool = True) -> QualityReport:
    """Measure data quality without modifying anything."""
    cfg = cfg or PreprocessConfig()
    rows = []
    for (pid, cond), g in df.groupby(KEY_COLS, sort=True):
        r = {"participant_id": pid, "condition": cond}
        r.update(_inspect_recording(g, cfg, has_rotation))
        rows.append(r)
    per = pd.DataFrame(rows)
    tot = _totals(per)
    return QualityReport(per_recording=per, totals=tot, warnings=_warnings(tot, cfg))


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #
def _segments(t: np.ndarray, gap_factor: float) -> list[np.ndarray]:
    if t.size < 2:
        return [np.arange(t.size)]
    dt = np.diff(t)
    brk = np.flatnonzero(dt > gap_factor * np.median(dt)) + 1
    return np.split(np.arange(t.size), brk)


def _repair_spikes(g: pd.DataFrame, cfg: PreprocessConfig) -> tuple[pd.DataFrame, int]:
    t = g["timestamp"].to_numpy(float)
    pos = g[POS_COLS].to_numpy(float)
    spike = _step_analysis(t, pos, cfg)["flag_spike"]
    n = int(spike.sum())
    if n:
        pos = pos.copy()
        for k in range(3):
            pos[spike, k] = np.interp(t[spike], t[~spike], pos[~spike, k])
        g = g.copy()
        g[POS_COLS] = pos
    return g, n


def _resample(g: pd.DataFrame, hz: float, cfg: PreprocessConfig, has_rotation: bool) -> tuple[pd.DataFrame, int]:
    """Linear interpolation of positions, SLERP of orientations, per contiguous segment.

    Grid points inside tracking gaps are *not* generated (no invented data).
    """
    t = g["timestamp"].to_numpy(float)
    pos = g[POS_COLS].to_numpy(float)
    quat = g[QUAT_COLS].to_numpy(float) if has_rotation else None
    new_t, new_pos, new_q = [], [], []
    dropped = 0
    for seg in _segments(t, cfg.gap_factor):
        if seg.size < 2:
            dropped += int(seg.size)
            continue
        ts = t[seg]
        grid = ts[0] + np.arange(0, int(np.floor((ts[-1] - ts[0]) * hz + 1e-9)) + 1) / hz
        grid = np.minimum(grid, ts[-1])          # the 1e-9 slack above must never push a grid point past the last sample
        new_t.append(grid)
        new_pos.append(np.column_stack([np.interp(grid, ts, pos[seg, k]) for k in range(3)]))
        if quat is not None:
            rot = Rotation.from_quat(enforce_quaternion_continuity(normalize_quaternions(quat[seg])))
            new_q.append(Slerp(ts, rot)(grid).as_quat())
    if not new_t:
        return g.iloc[0:0], dropped
    out = pd.DataFrame({"timestamp": np.concatenate(new_t)})
    p = np.vstack(new_pos)
    for k, c in enumerate(POS_COLS):
        out[c] = p[:, k]
    if quat is not None:
        q = np.vstack(new_q)
        for k, c in enumerate(QUAT_COLS):
            out[c] = q[:, k]
    else:
        for c in QUAT_COLS:
            out[c] = np.nan
    out.insert(0, "condition", g["condition"].iloc[0])
    out.insert(0, "participant_id", g["participant_id"].iloc[0])
    return out, dropped


def _smooth(g: pd.DataFrame, sigma_s: float, cfg: PreprocessConfig, has_rotation: bool) -> pd.DataFrame:
    """Gaussian smoothing per contiguous segment (assumes near-uniform sampling)."""
    t = g["timestamp"].to_numpy(float)
    if t.size < 3:
        return g
    sigma = sigma_s / float(np.median(np.diff(t)))
    pos = g[POS_COLS].to_numpy(float).copy()
    quat = g[QUAT_COLS].to_numpy(float).copy() if has_rotation else None
    for seg in _segments(t, cfg.gap_factor):
        if seg.size < 3:
            continue
        pos[seg] = gaussian_filter1d(pos[seg], sigma, axis=0, mode="nearest")
        if quat is not None:
            qs = gaussian_filter1d(enforce_quaternion_continuity(normalize_quaternions(quat[seg])), sigma,
                                   axis=0, mode="nearest")
            quat[seg] = normalize_quaternions(qs)       # nlerp: valid for small rotation within the window
    g = g.copy()
    g[POS_COLS] = pos
    if quat is not None:
        g[QUAT_COLS] = quat
    return g


def _path_stats(df: pd.DataFrame, gap_factor: float) -> tuple[float, float]:
    """(total path length, max step speed) over all recordings, excluding gaps."""
    total, vmax = 0.0, 0.0
    for _, g in df.groupby(KEY_COLS, sort=False):
        t = g["timestamp"].to_numpy(float)
        if t.size < 3:
            continue
        dt = np.diff(t)
        ok = dt <= gap_factor * np.median(dt)
        d = np.linalg.norm(np.diff(g[POS_COLS].to_numpy(float), axis=0), axis=1)
        total += float(d[ok].sum())
        vmax = max(vmax, float((d[ok] / dt[ok]).max())) if ok.any() else vmax
    return total, vmax


def _op(name: str, setting: str, before: int, after: int, affected: int, effect: str) -> dict:
    return {"operation": name, "setting": setting, "rows_before": before, "rows_after": after,
            "rows_affected": affected, "effect on analysis": effect}


def preprocess(df: pd.DataFrame, cfg: PreprocessConfig | None = None, has_rotation: bool = True):
    """Apply the configured operations; returns ``(clean_df, QualityReport)``."""
    cfg = cfg or PreprocessConfig()
    report = inspect_quality(df, cfg, has_rotation)
    cols = KEY_COLS + ["timestamp"] + POS_COLS + QUAT_COLS
    work = df[cols].copy()
    ops: list[dict] = []

    # 1 ── sort -----------------------------------------------------------------
    n0 = len(work)
    work = work.sort_values(KEY_COLS + ["timestamp"], kind="stable").reset_index(drop=True)
    ops.append(_op("Sort by timestamp", "per recording (always)", n0, len(work), report.totals["n_nonmonotonic"],
                   "Required for valid time differences. Out-of-order samples are re-ordered, not removed."))

    # 2 ── missing ---------------------------------------------------------------
    pose = work[POS_COLS + (QUAT_COLS if has_rotation else [])].to_numpy(float)
    bad = ~np.isfinite(work["timestamp"].to_numpy(float)) | ~np.isfinite(pose).all(axis=1)
    if has_rotation:
        bad |= np.linalg.norm(work[QUAT_COLS].to_numpy(float), axis=1) == 0
    n_bad = int(bad.sum())
    if cfg.drop_missing:
        work = work[~bad].reset_index(drop=True)
    ops.append(_op("Drop rows with missing / non-finite pose", "on" if cfg.drop_missing else "off", n0 - 0,
                   len(work), n_bad if cfg.drop_missing else 0,
                   "Removed rows create a longer time step; if it exceeds the gap threshold it is treated as a "
                   "tracking gap (excluded from speed statistics). Rows are never imputed."
                   if cfg.drop_missing else f"{n_bad} invalid rows kept -> affected recordings will be rejected."))

    # 3 ── duplicate timestamps --------------------------------------------------
    before = len(work)
    dup = work.duplicated(subset=KEY_COLS + ["timestamp"], keep="first")
    n_dup = int(dup.sum())
    if cfg.drop_duplicate_timestamps:
        work = work[~dup].reset_index(drop=True)
    ops.append(_op("Drop duplicate timestamps (keep first)", "on" if cfg.drop_duplicate_timestamps else "off", before,
                   len(work), n_dup if cfg.drop_duplicate_timestamps else 0,
                   "dt = 0 would make speed infinite; the first sample of each duplicated timestamp is kept."
                   if cfg.drop_duplicate_timestamps else f"{n_dup} duplicates kept -> affected recordings rejected."))

    # 4 ── quaternion renormalisation -------------------------------------------
    before = len(work)
    if has_rotation and cfg.renormalize_quaternions and len(work):
        q = work[QUAT_COLS].to_numpy(float)
        nrm = np.linalg.norm(q, axis=1, keepdims=True)
        n_fix = int(np.sum(np.abs(nrm[:, 0] - 1) > cfg.quat_norm_tol))
        work[QUAT_COLS] = q / nrm
        eff = ("All quaternions are rescaled to unit length. Directions are unchanged, so rotation angles are "
               "unaffected except where the stored norm was wrong.")
        ops.append(_op("Renormalise quaternions", f"tol = {cfg.quat_norm_tol:g}", before, len(work), n_fix, eff))
    elif has_rotation:
        ops.append(_op("Renormalise quaternions", "off", before, len(work), 0,
                       "Non-unit quaternions are normalised on the fly inside the angle computation."))
    else:
        ops.append(_op("Renormalise quaternions", "n/a (no orientation data)", before, len(work), 0, "-"))

    # 5 ── flag-only checks --------------------------------------------------
    n_disc = n_out = n_spk = 0
    groups = []
    for key, g in work.groupby(KEY_COLS, sort=True):
        t = g["timestamp"].to_numpy(float)
        if t.size >= 3:
            res = _step_analysis(t, g[POS_COLS].to_numpy(float), cfg)
            n_disc += int(res["flag_discontinuity"].sum())
            n_out += int(res["flag_outlier"].sum())
            n_spk += int(res["flag_spike"].sum())
        groups.append(g)
    ops.append(_op("Position discontinuity check", f"speed > {cfg.max_speed:g} units/s", len(work), len(work), 0,
                   f"{n_disc} samples flagged (flag only). Flagged jumps inflate path length and max speed."))
    ops.append(_op("Outlier detection (Tukey fence on step speed)", f"Q3 + {cfg.outlier_iqr_k:g} x IQR", len(work),
                   len(work), 0, f"{n_out} samples flagged (flag only). Statistical, so legitimate fast movement "
                   "can also be flagged - inspect before acting."))

    # 6 ── spike repair --------------------------------------------------------
    if cfg.repair_spikes:
        fixed, n_fixed = [], 0
        for g in groups:
            g2, k = _repair_spikes(g, cfg) if len(g) >= 3 else (g, 0)
            fixed.append(g2)
            n_fixed += k
        groups = fixed
        work = pd.concat(groups, ignore_index=True)
        ops.append(_op("Repair isolated position spikes", "linear interpolation", len(work), len(work), n_fixed,
                       "Positions of single-sample excursions that return to their origin are replaced by linear "
                       "interpolation in time. Multi-sample jumps are never altered."))
    else:
        ops.append(_op("Repair isolated position spikes", "off", len(work), len(work), 0,
                       f"{n_spk} isolated spikes detected but left untouched." if n_spk else "No isolated spikes."))

    # 7 ── resample --------------------------------------------------------------
    if cfg.resample_hz:
        p0, v0 = _path_stats(work, cfg.gap_factor)
        before = len(work)
        parts, dropped = [], 0
        for g in groups:
            if len(g) < 2:
                parts.append(g)
                continue
            r, d = _resample(g, cfg.resample_hz, cfg, has_rotation)
            parts.append(r)
            dropped += d
        groups = [p for p in parts if len(p)]
        work = pd.concat(groups, ignore_index=True) if groups else work.iloc[0:0]
        p1, v1 = _path_stats(work, cfg.gap_factor)
        ops.append(_op("Resample to uniform rate", f"{cfg.resample_hz:g} Hz (linear / SLERP)", before, len(work),
                       len(work), f"Path length {100 * (p1 - p0) / p0:+.2f}%, max speed {100 * (v1 - v0) / v0:+.2f}%. "
                       f"No samples are interpolated across tracking gaps; {dropped} isolated samples between gaps "
                       "could not be resampled." if p0 > 0 and v0 > 0 else "Resampled."))
    else:
        ops.append(_op("Resample to uniform rate", "off", len(work), len(work), 0,
                       "Original (possibly irregular) timestamps are kept; derivatives use actual dt."))

    # 8 ── smoothing -------------------------------------------------------------
    if cfg.smooth_sigma_s > 0:
        p0, v0 = _path_stats(work, cfg.gap_factor)
        groups = [_smooth(g, cfg.smooth_sigma_s, cfg, has_rotation) for g in groups]
        work = pd.concat(groups, ignore_index=True)
        p1, v1 = _path_stats(work, cfg.gap_factor)
        ops.append(_op("Gaussian smoothing", f"sigma = {cfg.smooth_sigma_s:g} s", len(work), len(work), len(work),
                       f"Path length {100 * (p1 - p0) / p0:+.2f}%, max speed {100 * (v1 - v0) / v0:+.2f}%. "
                       "Removes sensor jitter but attenuates genuine peaks; assumes near-uniform sampling; "
                       "quaternions are smoothed component-wise after sign alignment and renormalised."
                       if p0 > 0 and v0 > 0 else "Smoothed."))
    else:
        ops.append(_op("Gaussian smoothing", "off", len(work), len(work), 0,
                       "Raw jitter contributes to path length and speed (positive bias for stationary periods)."))

    work = work.reset_index(drop=True)
    after = inspect_quality(work, cfg, has_rotation) if len(work) else None
    report.operations = pd.DataFrame(ops)
    if after is not None:
        report.per_recording_after = after.per_recording
        report.totals_after = after.totals
    return work, report
