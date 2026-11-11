"""Scientific-correctness tests for translational / rotational features."""
import numpy as np
import pandas as pd
import pytest

from src.features import (
    FeatureConfig, TrajectoryError, angular_displacements, compute_kinematics, direction_change_stats,
    enforce_quaternion_continuity, euler_zyx_to_quat, kinematics_table, movement_metrics_table, quat_angle,
    quat_multiply, quat_relative, quat_to_euler_zyx, rotate_vectors, summarize_kinematics,
)


def line(n=301, fs=30.0, v=1.2):
    t = np.arange(n) / fs
    pos = np.column_stack([v * t, np.zeros(n), np.full(n, 1.6)])
    return t, pos


def yaw_quats(yaw):
    return euler_zyx_to_quat(np.asarray(yaw, float), np.zeros(len(yaw)), np.zeros(len(yaw)))


# ── translation ──────────────────────────────────────────────────────────────────────────
def test_stationary_trajectory_has_zero_path_length():
    t = np.arange(100) / 30
    pos = np.tile([1.0, 2.0, 1.6], (100, 1))
    kin = compute_kinematics(t, pos)
    s = summarize_kinematics(kin)
    assert s["path_length"] == 0.0
    assert s["mean_speed"] == 0.0 and s["max_speed"] == 0.0
    assert s["net_displacement"] == 0.0
    assert np.isnan(s["straightness"])           # undefined, not 0/0 -> inf


def test_constant_speed_is_recovered():
    t, pos = line(v=1.2)
    kin = compute_kinematics(t, pos)
    np.testing.assert_allclose(kin["speed"].iloc[1:], 1.2, rtol=1e-9)
    np.testing.assert_allclose(kin["accel"].iloc[1:-1], 0.0, atol=1e-8)
    s = summarize_kinematics(kin)
    assert s["path_length"] == pytest.approx(1.2 * 10.0)
    assert s["mean_speed"] == pytest.approx(1.2)
    assert s["speed_cv"] == pytest.approx(0.0, abs=1e-9)
    assert s["straightness"] == pytest.approx(1.0)


def test_irregular_sampling_uses_actual_timestamps():
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.uniform(0.01, 0.06, 400))
    pos = np.column_stack([2.0 * t, np.zeros_like(t), np.zeros_like(t)])
    kin = compute_kinematics(t, pos, cfg=FeatureConfig(gap_factor=100))
    np.testing.assert_allclose(kin["speed"].iloc[1:], 2.0, rtol=1e-9)
    np.testing.assert_allclose(kin["accel"].iloc[1:-1], 0.0, atol=1e-8)
    # a naive "per-frame" speed would NOT be constant:
    assert np.ptp(np.linalg.norm(np.diff(pos, axis=0), axis=1)) > 0.05


def test_circular_motion_acceleration():
    fs, r, w = 200.0, 2.0, 1.0                    # v = 2 m/s, a = v^2 / r = 2 m/s^2
    t = np.arange(0, 6.3, 1 / fs)
    pos = np.column_stack([r * np.cos(w * t), r * np.sin(w * t), np.zeros_like(t)])
    kin = compute_kinematics(t, pos)
    np.testing.assert_allclose(kin["speed"].iloc[1:], r * w, rtol=1e-3)
    np.testing.assert_allclose(kin["accel"].iloc[1:-1], r * w * w, rtol=1e-2)


def test_tracking_gap_is_not_interpreted_as_slow_movement():
    t = np.concatenate([np.arange(0, 2, 1 / 30), np.arange(5, 7, 1 / 30)])
    pos = np.column_stack([t * 1.0, np.zeros_like(t), np.zeros_like(t)])
    kin = compute_kinematics(t, pos)
    s = summarize_kinematics(kin)
    assert s["n_gap_steps"] == 1
    assert s["mean_speed"] == pytest.approx(1.0)   # gap excluded from both path and time
    assert s["observed_time_s"] < 4.1


def test_time_slice_summary():
    t, pos = line(n=601, v=1.0)
    kin = compute_kinematics(t, pos)
    s = summarize_kinematics(kin.iloc[100:301].reset_index(drop=True))
    assert s["path_length"] == pytest.approx(200 / 30)


# ── input validation ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("t,pos,msg", [
    (np.array([]), np.zeros((0, 3)), "empty"),
    (np.array([0.0]), np.zeros((1, 3)), "at least 2"),
    (np.array([0.0, 1.0, 1.0]), np.zeros((3, 3)), "strictly increasing"),
    (np.array([0.0, 2.0, 1.0]), np.zeros((3, 3)), "strictly increasing"),
    (np.array([0.0, np.nan, 2.0]), np.zeros((3, 3)), "NaN"),
    (np.array([0.0, 1.0, 2.0]), np.zeros((3, 2)), "shape"),
])
def test_invalid_trajectories_raise_clear_errors(t, pos, msg):
    with pytest.raises(TrajectoryError, match=msg):
        compute_kinematics(t, pos)


def test_invalid_quaternions_raise():
    t, pos = line(n=5)
    with pytest.raises(TrajectoryError):
        compute_kinematics(t, pos, np.zeros((5, 4)))
    q = np.tile([0, 0, 0, 1.0], (5, 1))
    q[2, 0] = np.nan
    with pytest.raises(TrajectoryError):
        compute_kinematics(t, pos, q)
    with pytest.raises(TrajectoryError, match="shape"):
        compute_kinematics(t, pos, np.ones((4, 4)))


def test_kinematics_table_reports_rejected_recordings_instead_of_hiding_them():
    good = pd.DataFrame({"participant_id": "A", "condition": "c", "timestamp": np.arange(10) / 30.0})
    for k, c in enumerate(["position_x", "position_y", "position_z"]):
        good[c] = np.arange(10) * (k == 0) * 0.01
    bad = good.copy()
    bad["participant_id"] = "B"
    bad["timestamp"] = 0.0                       # all identical timestamps
    kin, skipped = kinematics_table(pd.concat([good, bad]), has_rotation=False)
    assert set(kin["participant_id"]) == {"A"}
    assert ("B", "c") in skipped and "strictly increasing" in skipped[("B", "c")]


# ── rotation ─────────────────────────────────────────────────────────────────────────────
def test_stationary_orientation_has_zero_angular_displacement():
    q = np.tile([0.1, 0.2, 0.3, 0.9], (50, 1))
    np.testing.assert_allclose(angular_displacements(q), 0.0, atol=1e-12)
    t = np.arange(50) / 30
    kin = compute_kinematics(t, np.zeros((50, 3)), q)
    s = summarize_kinematics(kin)
    assert s["cum_rotation_deg"] == pytest.approx(0.0, abs=1e-9)
    assert s["max_ang_vel_dps"] == pytest.approx(0.0, abs=1e-9)


def test_quaternion_sign_flips_do_not_create_rotation_jumps():
    rng = np.random.default_rng(1)
    yaw = np.linspace(0, 2 * np.pi, 300)         # one full turn
    q = yaw_quats(yaw)
    flipped = q * np.where(rng.random(300) < 0.5, -1.0, 1.0)[:, None]
    a, b = angular_displacements(q), angular_displacements(flipped)
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert a.max() < 0.05                         # no artificial ~pi/2*... jump
    assert a.sum() == pytest.approx(2 * np.pi, rel=1e-6)


def test_constant_angular_velocity_is_recovered_across_yaw_wrap():
    fs, rate = 30.0, np.radians(90)
    t = np.arange(0, 8, 1 / fs)                   # 720 deg total -> crosses +-180 deg wrap and w = 0
    q = yaw_quats(rate * t)
    q[q[:, 3] < 0] *= -1                          # canonical hemisphere: real sign flips occur
    assert (np.einsum("ij,ij->i", q[1:], q[:-1]) < 0).any()
    kin = compute_kinematics(t, np.zeros((t.size, 3)), q)
    np.testing.assert_allclose(kin["ang_vel"].iloc[1:], rate, rtol=1e-9)
    assert summarize_kinematics(kin)["cum_rotation_deg"] == pytest.approx(np.degrees(rate * t[-1]), rel=1e-9)


def test_angle_wrapping_euler_subtraction_would_fail_but_quaternions_do_not():
    q = yaw_quats([np.radians(170), np.radians(190)])         # a 20 deg turn across the +-180 deg seam
    assert angular_displacements(q)[0] == pytest.approx(np.radians(20))
    yaw, _, _ = quat_to_euler_zyx(q)                           # wrapped Euler output: 170 deg -> -170 deg
    assert abs(np.degrees(yaw[1] - yaw[0])) > 300              # naive Euler subtraction gives ~340 deg


def test_small_angles_are_numerically_accurate():
    eps = 1e-7
    q = np.array([[0, 0, 0, 1.0], [0, 0, np.sin(eps / 2), np.cos(eps / 2)]])
    assert angular_displacements(q)[0] == pytest.approx(eps, rel=1e-6)


def test_quaternion_algebra():
    a = euler_zyx_to_quat(0.3, -0.2, 0.5)
    b = euler_zyx_to_quat(-0.4, 0.1, 0.2)
    ab = quat_multiply(a, b)
    assert np.linalg.norm(ab) == pytest.approx(1.0)
    np.testing.assert_allclose(quat_relative(a, ab), b, atol=1e-12)           # a^-1 (a b) = b
    assert quat_angle(np.array([0, 0, np.sin(0.4), np.cos(0.4)])) == pytest.approx(0.8)
    assert quat_angle(np.array([0, 0, -np.sin(0.4), -np.cos(0.4)])) == pytest.approx(0.8)


def test_euler_roundtrip_and_vector_rotation():
    yaw, pitch, roll = 0.7, -0.3, 0.2
    q = euler_zyx_to_quat(yaw, pitch, roll)
    y2, p2, r2 = quat_to_euler_zyx(q)
    assert (y2, p2, r2) == pytest.approx((yaw, pitch, roll))
    np.testing.assert_allclose(rotate_vectors(yaw_quats([np.pi / 2])[0], [1, 0, 0]), [0, 1, 0], atol=1e-12)


def test_enforce_continuity():
    q = yaw_quats(np.linspace(0, 4 * np.pi, 200))
    q[q[:, 3] < 0] *= -1
    c = enforce_quaternion_continuity(q)
    assert (np.einsum("ij,ij->i", c[1:], c[:-1]) > 0).all()


# ── direction changes ─────────────────────────────────────────────────────────────────────
def test_straight_walk_has_no_direction_changes():
    t, pos = line(n=600)
    kin = compute_kinematics(t, pos)
    st = direction_change_stats(kin)
    assert st["n_turns"] == 0
    assert st["mean_turn_angle_deg"] == pytest.approx(0.0, abs=1e-6)


def test_square_path_has_four_right_angle_turns():
    fs, v, side = 30.0, 1.0, 4.0
    pts = [(0, 0), (side, 0), (side, side), (0, side), (0, 0), (side, 0)]
    pos, t0 = [], 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        n = int(side / v * fs)
        s = np.linspace(0, 1, n, endpoint=False)
        pos.extend([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, 0.0) for a in s])
    pos = np.array(pos)
    t = np.arange(len(pos)) / fs
    kin = compute_kinematics(t, pos)
    st = direction_change_stats(kin, FeatureConfig(direction_stride_s=0.5, turn_threshold_deg=45))
    assert st["n_turns"] == 4
    assert max(1, st["n_turn_samples"]) > 4


def test_metrics_table_end_to_end_columns():
    t, pos = line()
    df = pd.DataFrame({"participant_id": "A", "condition": "c", "timestamp": t,
                       "position_x": pos[:, 0], "position_y": pos[:, 1], "position_z": pos[:, 2]})
    q = yaw_quats(np.linspace(0, 1, len(t)))
    for k, c in enumerate(["quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w"]):
        df[c] = q[:, k]
    kin, skipped = kinematics_table(df)
    m = movement_metrics_table(kin)
    assert not skipped and len(m) == 1
    assert {"path_length", "mean_speed", "cum_rotation_deg", "turn_rate_per_min"} <= set(m.columns)
    assert m["cum_rotation_deg"].iloc[0] == pytest.approx(np.degrees(1.0), rel=1e-6)
