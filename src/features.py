"""Feature engineering for 6-DoF recordings.

This module contains

* quaternion helpers (scalar-last ``(x, y, z, w)``, Hamilton product),
* translational kinematics (path length, speed, acceleration, ...),
* rotational kinematics (angular displacement / velocity from *relative*
  quaternions - never from differenced Euler angles),
* movement direction-change statistics,
* orchestration helpers that turn a tidy long-format table into per-sample
  kinematics and per-recording summary metrics.

Conventions
-----------
* Quaternions are stored as ``(x, y, z, w)`` and are *unit* quaternions.
  ``q`` and ``-q`` encode the same rotation, so every rotation metric is
  invariant to the sign of the quaternion.
* The angle between two orientations is independent of whether a quaternion
  encodes a body->world or world->body rotation (the two relative rotations
  are conjugate and share the same angle). Only the component *order* matters.
* "Step" quantities (``step_dist``, ``speed``, ``ang_vel``) belong to the
  interval ``(t[i-1], t[i]]`` and are stored at row ``i`` (row 0 is NaN).
  Acceleration is a centred difference around sample ``i``.
* All time derivatives use the *actual* timestamp differences.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import entropy as ent

POS_COLS = ["position_x", "position_y", "position_z"]
QUAT_COLS = ["quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w"]
EULER_COLS = ["euler_yaw_deg", "euler_pitch_deg", "euler_roll_deg"]
KEY_COLS = ["participant_id", "condition"]


class TrajectoryError(ValueError):
    """Raised when a trajectory cannot be analysed (empty, NaN, unsorted, ...)."""


# --------------------------------------------------------------------------- #
# Quaternion mathematics
# --------------------------------------------------------------------------- #
def quaternion_norms(q: np.ndarray) -> np.ndarray:
    """Euclidean norm of each quaternion (last axis)."""
    return np.linalg.norm(np.asarray(q, dtype=float), axis=-1)


def normalize_quaternions(q: np.ndarray) -> np.ndarray:
    """Return unit quaternions. Raises :class:`TrajectoryError` for zero/NaN norm."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.all(np.isfinite(n)) or np.any(n == 0.0):
        raise TrajectoryError(
            "Quaternions with zero or non-finite norm cannot be normalised "
            "(orientation undefined). Run preprocessing first."
        )
    return q / n


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (= inverse for unit quaternions)."""
    q = np.asarray(q, dtype=float)
    out = q.copy()
    out[..., :3] *= -1.0
    return out


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a (x) b`` for ``(x, y, z, w)`` quaternions."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def quat_angle(q: np.ndarray) -> np.ndarray:
    """Rotation angle (rad, in [0, pi]) encoded by quaternion(s) ``q``.

    ``2 * atan2(|v|, |w|)`` is well conditioned for tiny angles (unlike
    ``2 * arccos(w)``) and, through ``|w|``, invariant to the sign of ``q``.
    """
    q = np.asarray(q, dtype=float)
    return 2.0 * np.arctan2(np.linalg.norm(q[..., :3], axis=-1), np.abs(q[..., 3]))


def quat_relative(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Relative rotation ``q1^-1 (x) q2`` (rotation taking orientation 1 to 2)."""
    return quat_multiply(quat_conjugate(q1), q2)


def angular_displacements(q: np.ndarray) -> np.ndarray:
    """Angle (rad) between consecutive orientations, length ``N-1``."""
    q = normalize_quaternions(q)
    if q.shape[0] < 2:
        return np.zeros(0)
    return quat_angle(quat_relative(q[:-1], q[1:]))


def enforce_quaternion_continuity(q: np.ndarray) -> np.ndarray:
    """Flip signs so that consecutive quaternions lie in the same hemisphere.

    Needed before component-wise operations (smoothing, interpolation). Not
    needed for angular displacement, which is sign-invariant by construction.
    """
    q = np.asarray(q, dtype=float)
    if q.shape[0] < 2:
        return q.copy()
    d = np.sign(np.einsum("ij,ij->i", q[1:], q[:-1]))
    d[d == 0] = 1.0
    s = np.concatenate([[1.0], np.cumprod(d)])
    return q * s[:, None]


def quat_to_euler_zyx(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Yaw (z), pitch (y), roll (x) in radians for ``q = qz(yaw) qy(pitch) qx(roll)``.

    Display only - Euler angles are never used for rotation metrics.
    """
    q = normalize_quaternions(q)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw, pitch, roll


def euler_zyx_to_quat(yaw: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quat_to_euler_zyx`; returns unit ``(x, y, z, w)``."""
    cy, sy = np.cos(np.asarray(yaw) / 2), np.sin(np.asarray(yaw) / 2)
    cp, sp = np.cos(np.asarray(pitch) / 2), np.sin(np.asarray(pitch) / 2)
    cr, sr = np.cos(np.asarray(roll) / 2), np.sin(np.asarray(roll) / 2)
    return np.stack(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        axis=-1,
    )


def rotate_vectors(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) ``v`` by unit quaternion(s) ``q`` (active rotation)."""
    q = normalize_quaternions(q)
    u = q[..., :3]
    w = q[..., 3:4]
    v = np.broadcast_to(np.asarray(v, dtype=float), u.shape)
    uv = np.cross(u, v)
    return v + 2.0 * (w * uv + np.cross(u, uv))


_AXES = {"x": 0, "y": 1, "z": 2}


def axis_vector(spec: str) -> np.ndarray:
    """``"+x"``, ``"-z"``, ... -> unit vector."""
    spec = spec.strip().lower()
    sign = -1.0 if spec.startswith("-") else 1.0
    axis = spec.lstrip("+-")
    if axis not in _AXES:
        raise ValueError(f"Unknown axis specification {spec!r}")
    v = np.zeros(3)
    v[_AXES[axis]] = sign
    return v


def add_euler_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append display-only Euler angles (deg) where a valid quaternion exists."""
    out = df.copy()
    q = out[QUAT_COLS].to_numpy(dtype=float)
    ok = np.isfinite(q).all(axis=1) & (np.linalg.norm(q, axis=1) > 0)
    for c in EULER_COLS:
        out[c] = np.nan
    if ok.any():
        yaw, pitch, roll = quat_to_euler_zyx(q[ok])
        out.loc[ok, EULER_COLS[0]] = np.degrees(yaw)
        out.loc[ok, EULER_COLS[1]] = np.degrees(pitch)
        out.loc[ok, EULER_COLS[2]] = np.degrees(roll)
    return out


def horizontal_axes(up_axis: str) -> tuple[str, str]:
    """Position columns that span the horizontal plane for a given up axis."""
    up_axis = up_axis.lower()
    if up_axis == "z":
        return "position_x", "position_y"
    if up_axis == "y":
        return "position_x", "position_z"
    if up_axis == "x":
        return "position_y", "position_z"
    raise ValueError(f"up_axis must be 'x', 'y' or 'z', got {up_axis!r}")


# --------------------------------------------------------------------------- #
# Kinematics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureConfig:
    """Parameters of the feature extraction.

    gap_factor
        A step whose duration exceeds ``gap_factor x median(dt)`` is treated as
        a *tracking gap*: it is excluded from speed / path-length / angular
        statistics instead of being interpreted as very slow movement.
    direction_stride_s
        Positions are sub-sampled at this temporal stride before turning angles
        are computed, so sensor jitter does not create spurious "turns".
    direction_min_step
        Coarse displacements shorter than this (data units) are ignored
        (stationary periods have no defined movement direction).
    turn_threshold_deg
        Turning angle above which a direction change is counted.
    """

    gap_factor: float = 10.0
    direction_stride_s: float = 0.5
    direction_min_step: float = 0.05
    turn_threshold_deg: float = 45.0


def validate_trajectory(t, pos, quat=None, *, name: str = "recording"):
    """Validate and coerce one recording. Raises :class:`TrajectoryError`."""
    t = np.asarray(t, dtype=float)
    pos = np.asarray(pos, dtype=float)
    if t.ndim != 1:
        raise TrajectoryError(f"{name}: timestamps must be 1-D.")
    if t.size == 0:
        raise TrajectoryError(f"{name}: empty trajectory (0 samples).")
    if t.size < 2:
        raise TrajectoryError(f"{name}: at least 2 samples are required, got {t.size}.")
    if pos.shape != (t.size, 3):
        raise TrajectoryError(f"{name}: positions must have shape (N, 3), got {pos.shape}.")
    if not (np.isfinite(t).all() and np.isfinite(pos).all()):
        raise TrajectoryError(
            f"{name}: timestamps/positions contain NaN or inf - run preprocessing first."
        )
    dt = np.diff(t)
    n_bad = int(np.sum(dt <= 0))
    if n_bad:
        raise TrajectoryError(
            f"{name}: timestamps must be strictly increasing ({n_bad} non-positive "
            "intervals) - sort and de-duplicate in preprocessing."
        )
    if quat is not None:
        quat = np.asarray(quat, dtype=float)
        if quat.shape != (t.size, 4):
            raise TrajectoryError(f"{name}: quaternions must have shape (N, 4), got {quat.shape}.")
        if not np.isfinite(quat).all():
            raise TrajectoryError(f"{name}: quaternions contain NaN or inf.")
        quat = normalize_quaternions(quat)
    return t, pos, quat


def _pad_front(a: np.ndarray, fill: float = np.nan) -> np.ndarray:
    return np.concatenate([[fill], a])


def compute_kinematics(t, pos, quat=None, cfg: FeatureConfig | None = None, *, name="recording") -> pd.DataFrame:
    """Per-sample kinematics of one recording.

    Returns a DataFrame with ``N`` rows: ``t`` (s from first sample), the three
    position columns, ``dt``, ``step_valid``, ``step_dist``, ``speed``,
    ``accel`` (|centred velocity difference|), ``cum_dist`` and, when
    quaternions are given, the normalised quaternion columns plus ``ang_disp``
    (rad), ``ang_vel`` (rad/s) and ``cum_rot`` (rad).
    """
    cfg = cfg or FeatureConfig()
    t, pos, quat = validate_trajectory(t, pos, quat, name=name)
    n = t.size
    dt = np.diff(t)
    med_dt = float(np.median(dt))
    valid = dt <= cfg.gap_factor * med_dt

    dp = np.diff(pos, axis=0)
    dist = np.linalg.norm(dp, axis=1)
    vel = dp / dt[:, None]
    speed = np.where(valid, dist / dt, np.nan)

    # Acceleration at sample i (1..N-2): change of the velocity vector between
    # the two adjacent steps divided by the time between the step mid-points.
    accel = np.full(n, np.nan)
    if n >= 3:
        mid_gap = 0.5 * (dt[1:] + dt[:-1])
        a_vec = (vel[1:] - vel[:-1]) / mid_gap[:, None]
        a_mag = np.linalg.norm(a_vec, axis=1)
        ok = valid[1:] & valid[:-1]
        accel[1:-1] = np.where(ok, a_mag, np.nan)

    out = pd.DataFrame(
        {
            "t": t - t[0],
            POS_COLS[0]: pos[:, 0],
            POS_COLS[1]: pos[:, 1],
            POS_COLS[2]: pos[:, 2],
            "dt": _pad_front(dt),
            "step_valid": np.concatenate([[False], valid]),
            "step_dist": _pad_front(dist),
            "speed": _pad_front(speed),
            "accel": accel,
        }
    )
    out["cum_dist"] = np.concatenate([[0.0], np.cumsum(np.where(valid, dist, 0.0))])

    if quat is not None:
        for k, c in enumerate(QUAT_COLS):
            out[c] = quat[:, k]
        ang = angular_displacements(quat)
        ang_vel = np.where(valid, ang / dt, np.nan)
        out["ang_disp"] = _pad_front(ang)
        out["ang_vel"] = _pad_front(ang_vel)
        out["cum_rot"] = np.concatenate([[0.0], np.cumsum(np.where(valid, ang, 0.0))])
    return out


def direction_change_stats(kin: pd.DataFrame, cfg: FeatureConfig | None = None) -> dict:
    """Movement direction changes from coarse-sampled displacement vectors.

    1. Positions are sub-sampled at ``direction_stride_s``.
    2. Displacement vectors shorter than ``direction_min_step`` or spanning a
       tracking gap are discarded.
    3. The turning angle between successive *remaining* vectors is
       ``atan2(|a x b|, a . b)`` in [0, 180 deg]. Turns across a stationary
       period therefore count (direction before vs. after the pause).
    """
    cfg = cfg or FeatureConfig()
    t = kin["t"].to_numpy()
    pos = kin[POS_COLS].to_numpy()
    n = t.size
    nan_result = {"turn_rate_per_min": np.nan, "mean_turn_angle_deg": np.nan, "n_turns": 0, "n_turn_samples": 0}
    if n < 3:
        return nan_result
    dt = np.diff(t)
    med_dt = float(np.median(dt))
    k = max(1, int(round(cfg.direction_stride_s / med_dt)))
    idx = np.arange(0, n, k)
    if idx.size < 3:
        return nan_result
    d = np.diff(pos[idx], axis=0)
    length = np.linalg.norm(d, axis=1)
    dtc = np.diff(t[idx])
    ok = (length >= cfg.direction_min_step) & (dtc <= cfg.gap_factor * k * med_dt)
    dv = d[ok]
    if dv.shape[0] < 2:
        return nan_result
    a, b = dv[:-1], dv[1:]
    ang = np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum("ij,ij->i", a, b)))
    n_turns = int(np.sum(ang > cfg.turn_threshold_deg))
    observed = float(np.sum(dt[dt <= cfg.gap_factor * med_dt]))
    rate = n_turns / (observed / 60.0) if observed > 0 else np.nan
    return {
        "turn_rate_per_min": rate,
        "mean_turn_angle_deg": float(np.mean(ang)),
        "n_turns": n_turns,
        "n_turn_samples": int(ang.size),
    }


def _weighted_mean_std(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not m.any():
        return np.nan, np.nan
    x, w = x[m], w[m]
    mean = float(np.sum(w * x) / np.sum(w))
    var = float(np.sum(w * (x - mean) ** 2) / np.sum(w))
    return mean, float(np.sqrt(var))


def summarize_kinematics(kin: pd.DataFrame, cfg: FeatureConfig | None = None) -> dict:
    """Scalar translational / rotational metrics of one recording (or a time slice).

    Row 0 of ``kin`` is excluded from step statistics because its step belongs
    to the interval *before* the slice.
    """
    cfg = cfg or FeatureConfig()
    steps = kin.iloc[1:]
    valid = steps["step_valid"].to_numpy(dtype=bool)
    dt = steps["dt"].to_numpy()[valid]
    dist = steps["step_dist"].to_numpy()[valid]
    speed = steps["speed"].to_numpy()[valid]
    observed = float(dt.sum())
    path = float(dist.sum())
    mean_speed = path / observed if observed > 0 else np.nan
    _, speed_std = _weighted_mean_std(speed, dt)
    pos = kin[POS_COLS].to_numpy()
    net = float(np.linalg.norm(pos[-1] - pos[0]))
    accel = kin["accel"].to_numpy()
    accel = accel[np.isfinite(accel)]

    res = {
        "duration_s": float(kin["t"].iloc[-1] - kin["t"].iloc[0]),
        "observed_time_s": observed,
        "n_samples": int(len(kin)),
        "n_gap_steps": int((~valid).sum()),
        "path_length": path,
        "mean_speed": mean_speed,
        "speed_std": speed_std,
        "speed_cv": speed_std / mean_speed if mean_speed and mean_speed > 0 else np.nan,
        "max_speed": float(np.nanmax(speed)) if speed.size else np.nan,
        "speed_p95": float(np.percentile(speed, 95)) if speed.size else np.nan,
        "mean_accel": float(accel.mean()) if accel.size else np.nan,
        "max_accel": float(accel.max()) if accel.size else np.nan,
        "net_displacement": net,
        "straightness": net / path if path > 0 else np.nan,
    }
    if "ang_disp" in kin.columns:
        ang = steps["ang_disp"].to_numpy()[valid]
        av = steps["ang_vel"].to_numpy()[valid]
        cum = float(np.degrees(ang.sum()))
        res.update(
            {
                "cum_rotation_deg": cum,
                "mean_ang_vel_dps": cum / observed if observed > 0 else np.nan,
                "max_ang_vel_dps": float(np.degrees(np.nanmax(av))) if av.size else np.nan,
                "ang_vel_p95_dps": float(np.degrees(np.percentile(av, 95))) if av.size else np.nan,
            }
        )
    res.update(direction_change_stats(kin, cfg))
    return res


# --------------------------------------------------------------------------- #
# Table-level orchestration
# --------------------------------------------------------------------------- #
def kinematics_table(df: pd.DataFrame, cfg: FeatureConfig | None = None, has_rotation: bool = True):
    """Kinematics of every recording in a tidy table.

    Returns ``(kin, skipped)`` where ``kin`` is the concatenated per-sample
    table (with ``participant_id`` / ``condition``) and ``skipped`` maps
    ``(participant, condition)`` to the reason a recording could not be
    analysed. Nothing is skipped silently - callers must display ``skipped``.
    """
    cfg = cfg or FeatureConfig()
    parts: list[pd.DataFrame] = []
    skipped: dict[tuple[str, str], str] = {}
    for (pid, cond), g in df.groupby(KEY_COLS, sort=True):
        g = g.sort_values("timestamp", kind="stable")
        try:
            kin = compute_kinematics(
                g["timestamp"].to_numpy(),
                g[POS_COLS].to_numpy(),
                g[QUAT_COLS].to_numpy() if has_rotation else None,
                cfg,
                name=f"{pid}/{cond}",
            )
        except TrajectoryError as exc:
            skipped[(str(pid), str(cond))] = str(exc)
            continue
        kin.insert(0, "condition", str(cond))
        kin.insert(0, "participant_id", str(pid))
        parts.append(kin)
    if not parts:
        return pd.DataFrame(), skipped
    return pd.concat(parts, ignore_index=True), skipped


def movement_metrics_table(kin: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """One row per recording with translational/rotational/direction metrics."""
    cfg = cfg or FeatureConfig()
    rows = []
    for (pid, cond), g in kin.groupby(KEY_COLS, sort=True):
        row = {"participant_id": pid, "condition": cond}
        row.update(summarize_kinematics(g.reset_index(drop=True), cfg))
        rows.append(row)
    return pd.DataFrame(rows)


def exploration_metrics_table(df: pd.DataFrame, grid: ent.SpatialGrid, u_col: str, v_col: str,
                              cfg: FeatureConfig | None = None, min_visit_s: float = 0.0) -> pd.DataFrame:
    """One row per recording with spatial entropy / coverage / revisitation."""
    cfg = cfg or FeatureConfig()
    return ent.spatial_metrics_table(df, grid, u_col, v_col, gap_factor=cfg.gap_factor, min_visit_s=min_visit_s)


# --------------------------------------------------------------------------- #
# Metric catalogue (labels / units for the UI and the statistics page)
# --------------------------------------------------------------------------- #
def metric_catalogue(length_unit: str = "m") -> dict[str, dict]:
    u = length_unit
    return {
        "duration_s": dict(label="Recording duration", unit="s", group="Recording", rotation=False),
        "path_length": dict(label="Path length", unit=u, group="Translation", rotation=False),
        "mean_speed": dict(label="Mean speed", unit=f"{u}/s", group="Translation", rotation=False),
        "max_speed": dict(label="Max speed", unit=f"{u}/s", group="Translation", rotation=False),
        "speed_p95": dict(label="Speed (95th percentile)", unit=f"{u}/s", group="Translation", rotation=False),
        "speed_cv": dict(label="Speed variability (CV)", unit="", group="Translation", rotation=False),
        "mean_accel": dict(label="Mean acceleration", unit=f"{u}/s²", group="Translation", rotation=False),
        "net_displacement": dict(label="Net displacement", unit=u, group="Translation", rotation=False),
        "straightness": dict(label="Straightness (net / path)", unit="", group="Translation", rotation=False),
        "cum_rotation_deg": dict(label="Cumulative rotation", unit="deg", group="Rotation", rotation=True),
        "mean_ang_vel_dps": dict(label="Mean angular velocity", unit="deg/s", group="Rotation", rotation=True),
        "ang_vel_p95_dps": dict(label="Angular velocity (95th pct.)", unit="deg/s", group="Rotation", rotation=True),
        "max_ang_vel_dps": dict(label="Max angular velocity", unit="deg/s", group="Rotation", rotation=True),
        "turn_rate_per_min": dict(label="Direction changes / min", unit="1/min", group="Direction", rotation=False),
        "mean_turn_angle_deg": dict(label="Mean turning angle", unit="deg", group="Direction", rotation=False),
        "entropy_bits": dict(label="Spatial entropy", unit="bits", group="Exploration", rotation=False),
        "entropy_norm": dict(label="Normalised spatial entropy", unit="", group="Exploration", rotation=False),
        "perplexity": dict(label="Effective number of cells (2^H)", unit="cells", group="Exploration", rotation=False),
        "coverage": dict(label="Spatial coverage", unit="", group="Exploration", rotation=False),
        "revisit_ratio": dict(label="Revisit ratio", unit="", group="Exploration", rotation=False),
        "revisits_per_min": dict(label="Region revisits / min", unit="1/min", group="Exploration", rotation=False),
    }
