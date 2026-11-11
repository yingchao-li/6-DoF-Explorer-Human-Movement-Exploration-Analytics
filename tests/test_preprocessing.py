"""Tests for data-quality inspection and logged preprocessing."""
import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_dataset
from src.features import QUAT_COLS, POS_COLS, euler_zyx_to_quat, quat_angle, quat_relative
from src.preprocessing import PreprocessConfig, flag_samples, inspect_quality, preprocess


def make_df(n=300, fs=30.0, pid="A", cond="c", speed=1.0, rotate=True):
    t = np.arange(n) / fs
    df = pd.DataFrame({"participant_id": pid, "condition": cond, "timestamp": t,
                       "position_x": speed * t, "position_y": 0.0, "position_z": 1.6})
    yaw = np.radians(30) * t if rotate else np.zeros(n)
    q = euler_zyx_to_quat(yaw, np.zeros(n), np.zeros(n))
    for k, c in enumerate(QUAT_COLS):
        df[c] = q[:, k]
    return df


def test_clean_data_has_no_findings():
    rep = inspect_quality(make_df())
    r = rep.per_recording.iloc[0]
    assert r["n_missing_rows"] == r["n_duplicate_timestamps"] == r["n_nonmonotonic"] == r["n_gaps"] == 0
    assert r["n_discontinuities"] == r["n_spikes"] == r["n_quat_norm_bad"] == 0
    assert r["rate_hz"] == pytest.approx(30.0)
    assert r["median_dt_s"] == pytest.approx(1 / 30)
    assert rep.warnings == []


def test_inspection_detects_each_issue_exactly_and_does_not_modify_input():
    df = make_df(400)
    df.loc[10, "position_x"] = np.nan                     # missing
    df.loc[11, QUAT_COLS] = np.nan                         # missing quaternion
    df = pd.concat([df, df.iloc[[20, 21]]], ignore_index=True)   # 2 duplicate rows (appended -> also out of order)
    df.loc[100:104, QUAT_COLS] = df.loc[100:104, QUAT_COLS].to_numpy() * 1.1   # 5 non-unit quats
    df.loc[150, "position_y"] += 5.0                       # isolated spike
    df = df.drop(index=range(200, 240)).reset_index(drop=True)  # tracking gap (40 frames)
    snapshot = df.copy(deep=True)
    rep = inspect_quality(df)
    r = rep.per_recording.iloc[0]
    assert r["n_missing_rows"] == 2
    assert r["n_duplicate_timestamps"] == 2 and r["n_duplicate_rows"] == 2
    assert r["n_nonmonotonic"] >= 1
    assert r["n_quat_norm_bad"] == 5
    assert r["max_norm_dev"] == pytest.approx(0.1, abs=1e-6)
    assert r["n_gaps"] == 1 and r["max_gap_s"] == pytest.approx(41 / 30)
    assert r["n_spikes"] == 1 and r["n_discontinuities"] == 2
    pd.testing.assert_frame_equal(df, snapshot)           # inspection never changes the data
    assert any("gap" in w for w in rep.warnings)


def test_preprocess_removals_are_fully_accounted_for():
    df = make_df(400)
    df.loc[[10, 11, 12], "position_x"] = np.nan
    df = pd.concat([df, df.iloc[[50, 60]]], ignore_index=True)
    clean, rep = preprocess(df)
    ops = rep.operations.set_index("operation")
    removed = len(df) - len(clean)
    logged = (ops.loc["Drop rows with missing / non-finite pose", "rows_affected"]
              + ops.loc["Drop duplicate timestamps (keep first)", "rows_affected"])
    assert removed == logged == 5                         # nothing disappears silently
    assert (np.diff(clean["timestamp"]) > 0).all()        # sorted & strictly increasing
    assert "effect on analysis" in rep.operations.columns
    assert rep.operations["effect on analysis"].str.len().min() > 0


def test_disabling_drops_keeps_rows():
    df = make_df(100)
    df.loc[5, "position_x"] = np.nan
    clean, rep = preprocess(df, PreprocessConfig(drop_missing=False))
    assert len(clean) == 100


def test_quaternion_renormalisation():
    df = make_df(100)
    df.loc[:, QUAT_COLS] = df[QUAT_COLS].to_numpy() * 1.3
    clean, rep = preprocess(df)
    np.testing.assert_allclose(np.linalg.norm(clean[QUAT_COLS].to_numpy(), axis=1), 1.0, atol=1e-12)
    assert rep.operations.set_index("operation").loc["Renormalise quaternions", "rows_affected"] == 100
    off, _ = preprocess(df, PreprocessConfig(renormalize_quaternions=False))
    assert np.linalg.norm(off[QUAT_COLS].to_numpy(), axis=1)[0] == pytest.approx(1.3)


def test_spike_flagging_and_optional_repair():
    df = make_df(200)
    df.loc[80, "position_y"] += 4.0
    flags = flag_samples(df)
    assert flags["flag_spike"].sum() == 1 and flags.loc[flags["flag_spike"], "timestamp"].iloc[0] == df.timestamp[80]
    kept, rep = preprocess(df)                             # default: flag only
    assert kept.loc[80, "position_y"] == pytest.approx(4.0)
    fixed, rep2 = preprocess(df, PreprocessConfig(repair_spikes=True))
    assert fixed.loc[80, "position_y"] == pytest.approx(0.0, abs=1e-9)
    assert rep2.operations.set_index("operation").loc["Repair isolated position spikes", "rows_affected"] == 1


def test_multi_sample_jump_is_never_repaired():
    df = make_df(200)
    df.loc[100:, "position_y"] += 4.0                      # permanent teleport (single discontinuity)
    fixed, _ = preprocess(df, PreprocessConfig(repair_spikes=True))
    np.testing.assert_allclose(fixed["position_y"].to_numpy(), df["position_y"].to_numpy())


def test_resampling_slerp_unit_norm_and_no_interpolation_across_gaps():
    df = make_df(450)
    df = df[(df.timestamp < 4.0) | (df.timestamp > 8.0)].reset_index(drop=True)   # 4 s gap
    clean, rep = preprocess(df, PreprocessConfig(resample_hz=10.0))
    dt = np.diff(clean["timestamp"].to_numpy())
    assert np.allclose(dt[dt < 1], 0.1)
    inside = clean[(clean.timestamp > 4.0 + 0.05) & (clean.timestamp < 8.0 - 0.05)]
    assert inside.empty                                    # no invented data inside the gap
    np.testing.assert_allclose(np.linalg.norm(clean[QUAT_COLS].to_numpy(), axis=1), 1.0, atol=1e-9)
    # SLERP keeps the constant 30 deg/s rotation rate
    q = clean[QUAT_COLS].to_numpy()
    seg = clean.timestamp.to_numpy() < 3.9
    ang = quat_angle(quat_relative(q[:-1][seg[:-1]], q[1:][seg[:-1]]))
    np.testing.assert_allclose(ang, np.radians(30) * 0.1, rtol=1e-6)


def test_resampling_grid_never_exceeds_last_sample_for_irregular_timestamps():
    """Regression: floating-point slack in the grid length must not push a grid point past the segment end."""
    rng = np.random.default_rng(1)
    for hz in (15.0, 37.0, 60.0, 97.0):
        df = make_df(300)
        t = np.cumsum(rng.uniform(0.02, 0.05, 300))       # irregular sampling, arbitrary end time
        df["timestamp"] = t
        clean, _ = preprocess(df, PreprocessConfig(resample_hz=hz))
        assert clean["timestamp"].max() <= t[-1] + 1e-12
        np.testing.assert_allclose(np.linalg.norm(clean[QUAT_COLS].to_numpy(), axis=1), 1.0, atol=1e-9)


def test_resampling_with_injected_artifacts_does_not_raise():
    ds = generate_dataset(n_participants=6, seed=42, inject_artifacts=True)
    has_rot = ds.has_rotation
    for hz in (15.0, 60.0):
        clean, rep = preprocess(ds.df, PreprocessConfig(resample_hz=hz, repair_spikes=True, smooth_sigma_s=0.1), has_rot)
        assert len(clean) > 0
        assert clean["timestamp"].is_monotonic_increasing or clean.groupby("participant_id")["timestamp"].apply(lambda s: s.is_monotonic_increasing).all()


def test_smoothing_reduces_jitter_but_keeps_trend():
    rng = np.random.default_rng(0)
    df = make_df(600)
    noisy = df.copy()
    noisy[POS_COLS] = noisy[POS_COLS].to_numpy() + rng.normal(0, 0.003, (600, 3))
    raw, _ = preprocess(noisy)
    sm, rep = preprocess(noisy, PreprocessConfig(smooth_sigma_s=0.1))
    path = lambda d: np.linalg.norm(np.diff(d[POS_COLS].to_numpy(), axis=0), axis=1).sum()
    assert path(sm) < path(raw)
    assert path(sm) == pytest.approx(path(df), rel=0.05)
    assert np.abs(sm.position_x.to_numpy()[50:-50] - df.position_x.to_numpy()[50:-50]).max() < 0.005
    np.testing.assert_allclose(np.linalg.norm(sm[QUAT_COLS].to_numpy(), axis=1), 1.0, atol=1e-12)
    assert "Path length" in rep.operations.set_index("operation").loc["Gaussian smoothing", "effect on analysis"]


def test_works_without_orientation_data():
    df = make_df(100)
    df[QUAT_COLS] = np.nan
    clean, rep = preprocess(df, has_rotation=False)
    assert len(clean) == 100 and rep.totals["n_missing_rows"] == 0


def test_injected_synthetic_artifacts_are_detected_with_exact_counts():
    ds = generate_dataset(n_participants=20, seed=7, inject_artifacts=True)
    inj = ds.meta["injected"].set_index("participant_id")
    assert len(inj) > 0
    rep = inspect_quality(ds.df)
    per = rep.per_recording.set_index("participant_id")
    for pid, row in inj.iterrows():
        r = per.loc[pid]
        assert r["n_spikes"] == row["spikes"]
        assert r["n_missing_rows"] == row["missing_rows"]
        assert r["n_duplicate_timestamps"] == row["duplicate_rows"]
        assert r["n_quat_norm_bad"] == row["quat_non_unit_rows"]
        assert r["n_gaps"] == 1 and r["max_gap_s"] == pytest.approx(row["gap_seconds"], abs=1e-6)
    clean_pids = set(per.index) - set(inj.index)
    assert per.loc[list(clean_pids), ["n_spikes", "n_missing_rows", "n_gaps"]].to_numpy().sum() == 0
