"""Reproducible *synthetic* 6-DoF exploration data.

**All data produced here is simulated. It is NOT an observation of real VR
experiments** and says nothing about how real people explore virtual spaces.

Simulation model
----------------
* Arena: 10 m x 10 m room, ``z`` is up, eye height ~1.65 m. A 5 x 5 lattice of
  25 "points of interest" (POIs, 2 m spacing) serves as navigation targets.
* Locomotion: a smooth kinematic model (first-order speed and turn-rate
  dynamics, wall repulsion, correlated heading noise) so that trajectories are
  continuous with bounded acceleration.
* Head orientation: body heading + a critically damped gaze process for yaw and
  pitch + slowly varying roll, converted to quaternions (ZYX Euler ->
  unit quaternion, scalar last). Quaternions are stored in the canonical
  hemisphere ``w >= 0`` (as many tracking systems do), which produces genuine
  sign flips that analysis code must handle.
* Conditions
    - **Structured**: boustrophedon sweep through all POIs (second lap in
      reverse), low heading noise, moderate head rotation.
    - **Free**: random POI hops with a strong preference to *return* to
      already visited POIs, irregular mid-route re-targeting, high heading
      noise and larger / faster gaze shifts.
  Cruise speed has the same distribution in both conditions by construction;
  any difference in mean speed arises from dwell / turning dynamics.
* Between-subject design: each participant contributes one recording.
* Optional tracking artifacts (spikes, gaps, duplicates, NaNs, non-unit
  quaternions) can be injected to demonstrate quality control. The exact
  injected counts are returned in ``meta['injected']``.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .data_loader import Dataset, fingerprint_frame
from .features import EULER_COLS, POS_COLS, QUAT_COLS, euler_zyx_to_quat, quat_multiply, quat_to_euler_zyx

ARENA = (0.0, 10.0, 0.0, 10.0)          # u_min, u_max, v_min, v_max (m)
POI_COORDS = np.array([[x, y] for y in (1.0, 3.0, 5.0, 7.0, 9.0) for x in (1.0, 3.0, 5.0, 7.0, 9.0)])
SPAWN = (5.0, 5.0)
GENERATOR_VERSION = "1.0"

CONDITIONS = ("Structured", "Free")

# Population-level parameters (mean, sd across participants unless stated)
_COMMON = dict(v_cruise=(1.15, 0.10), eye_height=(1.65, 0.06))
_COND = {
    "Structured": dict(
        heading_noise=(0.06, 0.02), dwell_median=1.2, dwell_sigma=0.25, p_revisit=0.0, p_retarget=0.0, p_skip=0.08,
        gaze_yaw_sd=(np.radians(22), np.radians(5)), gaze_pitch_sd=(np.radians(5), np.radians(1.5)),
        gaze_interval=2.2, gaze_wn=6.0, roll_sd=np.radians(1.5),
    ),
    "Free": dict(
        heading_noise=(0.35, 0.10), dwell_median=1.2, dwell_sigma=0.9, p_revisit=(0.30, 0.10), p_retarget=0.04,
        p_skip=0.0, gaze_yaw_sd=(np.radians(32), np.radians(7)), gaze_pitch_sd=(np.radians(9), np.radians(2)),
        gaze_interval=1.4, gaze_wn=8.0, roll_sd=np.radians(2.5),
    ),
}


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _structured_route(rng: np.random.Generator) -> list[int]:
    """Boustrophedon through the 5x5 lattice, random orientation, then reversed."""
    grid = np.arange(25).reshape(5, 5)                    # row = y index, col = x index
    if rng.random() < 0.5:
        grid = grid.T
    if rng.random() < 0.5:
        grid = grid[::-1]
    if rng.random() < 0.5:
        grid = grid[:, ::-1]
    route = []
    for r in range(5):
        row = grid[r] if r % 2 == 0 else grid[r][::-1]
        route.extend(int(i) for i in row)
    return route + route[::-1][1:]


def _simulate(rng: np.random.Generator, condition: str, duration: float, fs: float) -> dict:
    p = _COND[condition]
    dt = 1.0 / fs
    n = int(round(duration * fs)) + 1

    v_cruise = max(0.7, rng.normal(*_COMMON["v_cruise"]))
    eye_h = rng.normal(*_COMMON["eye_height"])
    sig_delta = max(0.01, rng.normal(*p["heading_noise"]))
    yaw_sd = max(np.radians(8), rng.normal(*p["gaze_yaw_sd"]))
    pitch_sd = max(np.radians(2), rng.normal(*p["gaze_pitch_sd"]))
    p_rev = float(np.clip(rng.normal(*p["p_revisit"]), 0.2, 0.7)) if condition == "Free" else 0.0
    wn = p["gaze_wn"]
    activity = float(rng.lognormal(0.0, 0.25))             # head-movement vigour, same distribution in both conditions
    yaw_sd *= activity
    pitch_sd *= activity
    gaze_interval = p["gaze_interval"] / activity

    # pre-drawn randomness (per-step) --------------------------------------------------
    eta_delta = rng.standard_normal(n)
    u_retarget = rng.random(n)
    u_gaze = rng.random(n)
    eta_gaze = rng.standard_normal((n, 2))
    eta_roll = rng.standard_normal(n)

    route = _structured_route(rng) if condition == "Structured" else []
    route_i = 0
    visits = np.zeros(len(POI_COORDS))
    visits[12] = 1                                        # spawn is the central POI
    cur_poi = 12

    def pick_free(cur: int) -> int:
        d = np.linalg.norm(POI_COORDS - POI_COORDS[cur], axis=1)
        visited = visits > 0
        if visited.sum() > 1 and rng.random() < p_rev:
            w = visits * np.exp(-d / 4.0)
            w[cur] = 0
            w = w if w.sum() > 0 else np.ones_like(w)
        else:
            w = np.exp(-d / 3.0) * (~visited)
            w[cur] = 0
            if w.sum() <= 0:
                w = np.exp(-d / 3.0)
                w[cur] = 0
        w = w / w.sum()
        return int(rng.choice(len(w), p=w))

    def next_target(cur: int) -> int:
        nonlocal route_i
        if condition == "Structured":
            if route_i >= len(route):
                route_i = 0
            if rng.random() < p["p_skip"]:
                route_i += 1                                # occasionally skips a POI
                if route_i >= len(route):
                    route_i = 0
            tgt = route[route_i]
            route_i += 1
            if tgt == cur and route_i < len(route):
                tgt = route[route_i]
                route_i += 1
            return tgt
        return pick_free(cur)

    def dwell_time() -> float:
        d = rng.lognormal(math.log(p["dwell_median"]), p["dwell_sigma"])
        return float(np.clip(d, 0.5 if condition == "Structured" else 0.3, 2.4 if condition == "Structured" else 8.0))

    # state ---------------------------------------------------------------------------
    x, y = SPAWN
    th = rng.uniform(-math.pi, math.pi)
    v = 0.0
    om = 0.0
    delta = 0.0
    tgt = next_target(cur_poi)
    phase = "dwell"
    dwell_left = dwell_time()
    gy_t = gp_t = 0.0
    gy = gp = gvy = gvp = 0.0
    roll = 0.0
    next_gaze = rng.exponential(gaze_interval)

    X = np.empty(n); Y = np.empty(n); V = np.empty(n); TH = np.empty(n)
    GY = np.empty(n); GP = np.empty(n); R = np.empty(n)
    margin, tau_om, tau_v, k_h, om_max = 0.9, 0.25, 0.5, 3.0, 3.5
    ar = 0.35                                              # arrival radius
    tau_d = 1.0                                            # heading-noise correlation time
    a_d = math.exp(-dt / tau_d)
    s_d = sig_delta * math.sqrt(1 - a_d * a_d)

    for i in range(n):
        X[i], Y[i], V[i], TH[i], GY[i], GP[i], R[i] = x, y, v, th, gy, gp, roll
        # --- decision logic -------------------------------------------------------
        if phase == "dwell":
            dwell_left -= dt
            if dwell_left <= 0:
                tgt = next_target(cur_poi)
                phase = "move"
        else:
            if p["p_retarget"] > 0 and u_retarget[i] < p["p_retarget"] * dt:
                tgt = pick_free(cur_poi)
        tx, ty = POI_COORDS[tgt]
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        if phase == "move" and dist < ar:
            phase = "dwell"
            cur_poi = tgt
            visits[tgt] += 1
            dwell_left = dwell_time()

        # --- heading control ----------------------------------------------------------
        if dist > 1e-6:
            ux, uy = dx / dist, dy / dist
        else:
            ux, uy = math.cos(th), math.sin(th)
        for d_w, nx_, ny_ in ((x, 1, 0), (ARENA[1] - x, -1, 0), (y, 0, 1), (ARENA[3] - y, 0, -1)):
            if d_w < margin:
                wgt = 2.0 * ((margin - d_w) / margin) ** 2
                ux += wgt * nx_
                uy += wgt * ny_
        delta = a_d * delta + s_d * eta_delta[i]
        phi = math.atan2(uy, ux) + delta
        err = _wrap(phi - th)
        om_des = max(-om_max, min(om_max, k_h * err))
        om += (om_des - om) * dt / tau_om
        th += om * dt

        # --- speed control ------------------------------------------------------------
        if phase == "dwell":
            v_des = 0.0
        else:
            v_des = v_cruise * min(1.0, dist / 1.2) * (0.25 + 0.75 * max(0.0, math.cos(err)))
        v += (v_des - v) * dt / tau_v
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        x = min(max(x, 0.15), ARENA[1] - 0.15)
        y = min(max(y, 0.15), ARENA[3] - 0.15)

        # --- gaze (critically damped 2nd order tracking of random targets) ----------------
        if i * dt >= next_gaze:
            scale = 1.5 if phase == "dwell" else 1.0
            gy_t = float(np.clip(yaw_sd * scale * eta_gaze[i, 0], -math.radians(110), math.radians(110)))
            gp_t = float(np.clip(-0.08 + pitch_sd * eta_gaze[i, 1], -math.radians(40), math.radians(40)))
            next_gaze = i * dt + rng.exponential(gaze_interval) * (0.7 if u_gaze[i] < 0.3 else 1.0)
        gvy += (wn * wn * (gy_t - gy) - 2 * wn * gvy) * dt
        gy += gvy * dt
        gvp += (wn * wn * (gp_t - gp) - 2 * wn * gvp) * dt
        gp += gvp * dt
        roll += (-roll / 1.5) * dt + p["roll_sd"] * math.sqrt(2 * dt / 1.5) * eta_roll[i]

    return dict(t=np.arange(n) * dt, x=X, y=Y, v=V, theta=TH, gaze_yaw=GY, gaze_pitch=GP, roll=R, eye_h=eye_h,
                v_cruise=v_cruise, p_revisit=p_rev, heading_noise=sig_delta, gaze_yaw_sd_deg=math.degrees(yaw_sd))


def _recording_frame(rng: np.random.Generator, sim: dict, pid: str, condition: str, fs: float,
                     pos_noise: float = 0.0002, rot_noise: float = 1.5e-4) -> pd.DataFrame:
    n = sim["t"].size
    s = np.cumsum(sim["v"]) / fs                           # distance travelled
    bob = 0.012 * np.minimum(1.0, sim["v"] / 1.2) * np.sin(2 * np.pi * s / 0.7)
    sway = gaussian_filter1d(rng.standard_normal(n), 1.5 * fs)
    sway = 0.004 * sway / (sway.std() + 1e-12)
    z = sim["eye_h"] + bob + sway
    pos = np.column_stack([sim["x"], sim["y"], z]) + rng.normal(0, pos_noise, (n, 3))

    yaw = sim["theta"] + sim["gaze_yaw"]
    q = euler_zyx_to_quat(yaw, sim["gaze_pitch"], sim["roll"])
    eps = rng.normal(0, rot_noise, (n, 3))                 # small-angle jitter
    dq = np.concatenate([eps / 2, np.ones((n, 1))], axis=1)
    dq /= np.linalg.norm(dq, axis=1, keepdims=True)
    q = quat_multiply(q, dq)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 3] < 0] *= -1.0                                  # canonical hemisphere w >= 0

    df = pd.DataFrame(
        {"participant_id": pid, "condition": condition, "timestamp": sim["t"]}
    )
    for k, c in enumerate(POS_COLS):
        df[c] = pos[:, k]
    for k, c in enumerate(QUAT_COLS):
        df[c] = q[:, k]
    return df


def _inject_artifacts(rng: np.random.Generator, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Insert documented tracking artifacts into one recording (non-overlapping segments)."""
    n = len(df)
    df = df.copy().reset_index(drop=True)
    inj = dict(spikes=0, missing_rows=0, quat_non_unit_rows=0, duplicate_rows=0, gap_rows_removed=0, gap_seconds=0.0)

    def seg(lo: float, hi: float) -> tuple[int, int]:
        return int(lo * n), int(hi * n)

    a, b = seg(0.10, 0.30)                                  # isolated position spikes
    k = int(rng.integers(1, 4))
    idx = np.sort(rng.choice(np.arange(a + 2, b - 2, 12), size=k, replace=False))
    for i in idx:
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(1.5, 4.0)
        df.loc[i, "position_x"] += r * np.cos(ang)
        df.loc[i, "position_y"] += r * np.sin(ang)
    inj["spikes"] = int(k)

    a, b = seg(0.30, 0.45)                                  # missing positions
    k = int(rng.integers(2, 6))
    idx = rng.choice(np.arange(a, b), size=k, replace=False)
    df.loc[idx, "position_x"] = np.nan
    inj["missing_rows"] = int(k)

    a, b = seg(0.45, 0.60)                                  # non-unit quaternions
    k = int(rng.integers(5, 21))
    idx = np.arange(a, a + k)
    scale = rng.uniform(1.02, 1.2)
    df.loc[idx, QUAT_COLS] = df.loc[idx, QUAT_COLS].to_numpy() * scale
    inj["quat_non_unit_rows"] = int(k)

    a, b = seg(0.62, 0.72)                                  # tracking gap (rows removed)
    L = int(rng.integers(20, 61))
    t_before = df["timestamp"].iloc[a - 1]
    t_after = df["timestamp"].iloc[a + L]
    df = df.drop(index=np.arange(a, a + L)).reset_index(drop=True)
    inj["gap_rows_removed"] = L
    inj["gap_seconds"] = float(t_after - t_before)

    a, b = seg(0.80, 0.95)                                  # duplicated rows
    k = int(rng.integers(1, 4))
    idx = np.sort(rng.choice(np.arange(a, b), size=k, replace=False))
    dup = df.loc[idx]
    df = pd.concat([df, dup]).sort_index(kind="stable").reset_index(drop=True)
    inj["duplicate_rows"] = int(k)
    return df, inj


def generate_dataframe(n_participants: int = 20, duration_range: tuple[int, int] = (60, 120), fs: float = 30.0,
                       seed: int = 42, inject_artifacts: bool = False) -> tuple[pd.DataFrame, dict]:
    """Generate the tidy table and its metadata. Deterministic for a given ``seed``."""
    if n_participants < 2:
        raise ValueError("n_participants must be >= 2.")
    ss = np.random.SeedSequence(seed)
    child = ss.spawn(n_participants + 2)
    rng_assign = np.random.default_rng(child[0])
    rng_art = np.random.default_rng(child[1])
    half = n_participants // 2
    labels = np.array(["Structured"] * half + ["Free"] * (n_participants - half))
    labels = labels[rng_assign.permutation(n_participants)]

    frames, part_rows, inj_rows = [], [], []
    for i in range(n_participants):
        rng = np.random.default_rng(child[i + 2])
        cond = str(labels[i])
        pid = f"P{i + 1:02d}"
        duration = int(rng.integers(duration_range[0], duration_range[1] + 1))
        sim = _simulate(rng, cond, duration, fs)
        frame = _recording_frame(rng, sim, pid, cond, fs)
        if inject_artifacts and rng_art.random() < 0.4:
            frame, inj = _inject_artifacts(rng_art, frame)
            inj_rows.append({"participant_id": pid, **inj})
        frames.append(frame)
        part_rows.append({"participant_id": pid, "condition": cond, "duration_s": duration,
                          "true_cruise_speed": sim["v_cruise"], "true_heading_noise": sim["heading_noise"],
                          "true_gaze_yaw_sd_deg": sim["gaze_yaw_sd_deg"], "true_p_revisit": sim["p_revisit"]})
    df = pd.concat(frames, ignore_index=True)
    ypr = quat_to_euler_zyx(df[QUAT_COLS].to_numpy())      # display-only Euler angles (no NaN quaternions injected)
    for c, a in zip(EULER_COLS, ypr):
        df[c] = np.degrees(a)
    inj_cols = ["participant_id", "spikes", "missing_rows", "quat_non_unit_rows", "duplicate_rows",
                "gap_rows_removed", "gap_seconds"]
    meta = {
        "seed": seed, "n_participants": n_participants, "fs_hz": fs, "duration_range_s": duration_range,
        "inject_artifacts": inject_artifacts, "generator_version": GENERATOR_VERSION,
        "arena": ARENA, "pois": POI_COORDS.copy(), "spawn": SPAWN,
        "participants": pd.DataFrame(part_rows),
        "injected": pd.DataFrame(inj_rows, columns=inj_cols) if inj_rows else pd.DataFrame(columns=inj_cols),
        "parameters": {"common": _COMMON, "conditions": _COND},
    }
    return df, meta


def generate_dataset(n_participants: int = 20, duration_range: tuple[int, int] = (60, 120), fs: float = 30.0,
                     seed: int = 42, inject_artifacts: bool = False) -> Dataset:
    """Synthetic data wrapped in a :class:`~src.data_loader.Dataset` (source = ``"synthetic"``)."""
    df, meta = generate_dataframe(n_participants, duration_range, fs, seed, inject_artifacts)
    meta["fingerprint"] = fingerprint_frame(df)
    return Dataset(
        df=df, source="synthetic",
        name=f"Synthetic demo (seed {seed}, n={n_participants}{', with artifacts' if inject_artifacts else ''})",
        has_rotation=True, has_condition=True, up_axis="z", unit_label="m", arena_bounds=ARENA,
        forward_axis="+x", meta=meta,
    )
