"""Tests for the synthetic data generator and pipeline integration."""
import numpy as np
import pandas as pd
import pytest

from src.data_generator import ARENA, generate_dataset
from src.data_loader import fingerprint_frame
from src.features import EULER_COLS, QUAT_COLS, POS_COLS
from src.pipeline import AnalysisConfig, run_analysis


@pytest.fixture(scope="module")
def ds():
    return generate_dataset(seed=42)


def test_schema_and_design(ds):
    df = ds.df
    assert {"participant_id", "condition", "timestamp", *POS_COLS, *QUAT_COLS, *EULER_COLS} <= set(df.columns)
    assert df.participant_id.nunique() == 20
    assert df.groupby("condition").participant_id.nunique().to_dict() == {"Free": 10, "Structured": 10}
    assert ds.is_synthetic and ds.source == "synthetic" and "Synthetic" in ds.name


def test_sampling_rate_and_duration(ds):
    for pid, g in ds.df.groupby("participant_id"):
        dt = np.diff(g.timestamp.to_numpy())
        np.testing.assert_allclose(dt, 1 / 30, atol=1e-9)
        assert 60 <= g.timestamp.max() <= 120 + 1e-9


def test_reproducible_with_fixed_seed_and_sensitive_to_seed():
    a = generate_dataset(n_participants=6, seed=5).df
    b = generate_dataset(n_participants=6, seed=5).df
    c = generate_dataset(n_participants=6, seed=6).df
    pd.testing.assert_frame_equal(a, b)
    assert fingerprint_frame(a) == fingerprint_frame(b) != fingerprint_frame(c)


def test_quaternions_are_unit_canonical_and_contain_sign_flips(ds):
    q = ds.df[QUAT_COLS].to_numpy()
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-12)
    assert (q[:, 3] >= 0).all()
    flips = 0
    for _, g in ds.df.groupby("participant_id"):
        qq = g[QUAT_COLS].to_numpy()
        flips += int((np.einsum("ij,ij->i", qq[1:], qq[:-1]) < 0).sum())
    assert flips > 0                                       # analysis code must cope with them


def test_trajectories_are_continuous_and_inside_the_arena(ds):
    u0, u1, v0, v1 = ARENA
    assert ds.df.position_x.between(u0, u1).all() and ds.df.position_y.between(v0, v1).all()
    assert ds.df.position_z.between(1.3, 2.0).all()
    for _, g in ds.df.groupby("participant_id"):
        step = np.linalg.norm(np.diff(g[POS_COLS].to_numpy(), axis=0), axis=1)
        assert step.max() * 30 < 2.0                       # < 2 m/s: no teleports, human walking speeds


def test_pipeline_reproduces_designed_differences(ds):
    res = run_analysis(ds, AnalysisConfig())
    m = res.metrics.groupby("condition").mean(numeric_only=True)
    assert m.loc["Free", "mean_ang_vel_dps"] > 1.5 * m.loc["Structured", "mean_ang_vel_dps"]
    assert m.loc["Structured", "coverage"] > m.loc["Free", "coverage"]
    assert m.loc["Structured", "entropy_norm"] > m.loc["Free", "entropy_norm"]
    assert m.loc["Free", "revisit_ratio"] > m.loc["Structured", "revisit_ratio"]
    assert len(res.metrics) == 20 and not res.skipped
    assert res.qc.totals["n_missing_rows"] == 0 and res.qc.totals["n_gaps"] == 0


def test_stationary_dwell_periods_exist_and_speeds_are_physical(ds):
    res = run_analysis(ds)
    assert res.metrics.max_speed.max() < 2.0
    assert (res.kin.speed.dropna() < 0.05).mean() > 0.02    # people stop at points of interest


def test_artifact_injection_is_optional_and_off_by_default(ds):
    assert ds.meta["injected"].empty
    d2 = generate_dataset(seed=42, inject_artifacts=True)
    assert not d2.meta["injected"].empty
    assert len(d2.df) != len(ds.df) or not d2.df.equals(ds.df)
