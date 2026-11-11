"""End-to-end checks of the analysis pipeline (synthetic data only; tests the code, not human behaviour)."""
import numpy as np
import pandas as pd
import pytest

from src import statistics as stt
from src.data_generator import generate_dataset
from src.data_loader import ColumnMapping, ImportSettings, parse_table
from src.pipeline import AnalysisConfig, default_stat_metrics, run_analysis


@pytest.fixture(scope="module")
def result():
    ds = generate_dataset(n_participants=12, seed=5)
    return ds, run_analysis(ds, AnalysisConfig())


def test_one_row_per_participant_not_per_frame(result):
    ds, res = result
    assert len(res.metrics) == ds.df["participant_id"].nunique() == 12
    assert len(res.kin) > 100 * len(res.metrics)                 # many frames, few statistical units
    cmp = stt.compare_metrics(res.metrics, ["mean_ang_vel_dps"], "Structured", "Free")
    assert int(cmp["n_a"].iloc[0]) + int(cmp["n_b"].iloc[0]) == 12   # never the number of frames


def test_designed_rotation_difference_is_recovered_with_correct_sign(result):
    _, res = result
    cmp = stt.compare_metrics(res.metrics, ["mean_ang_vel_dps"], "Structured", "Free").iloc[0]
    assert cmp["diff"] > 0                                       # differences are Free - Structured
    assert cmp["p_param"] < 0.05


def test_pipeline_is_deterministic():
    a = run_analysis(generate_dataset(n_participants=6, seed=3)).metrics
    b = run_analysis(generate_dataset(n_participants=6, seed=3)).metrics
    pd.testing.assert_frame_equal(a, b)


def test_all_default_stat_metrics_exist_and_are_finite(result):
    _, res = result
    for m in default_stat_metrics(True):
        assert m in res.metrics.columns
        assert np.isfinite(res.metrics[m]).all(), m
    assert (res.metrics["entropy_norm"].between(0, 1)).all()
    assert (res.metrics["coverage"].between(0, 1)).all()


def test_without_quaternions_rotation_metrics_are_absent_not_zero():
    ds = generate_dataset(n_participants=4, seed=2)
    raw = ds.df.drop(columns=["quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w"] +
                     [c for c in ds.df.columns if c.startswith("euler")])
    mp = ColumnMapping(participant="participant_id", timestamp="timestamp", x="position_x", y="position_y", z="position_z",
                       condition="condition")
    up, rep = parse_table(raw, mp, ImportSettings(time_unit="s", position_unit="m", up_axis="z", forward_axis="+x"), "noq.csv")
    assert rep.ok and not up.has_rotation
    res = run_analysis(up)
    assert "mean_ang_vel_dps" not in res.metrics.columns and "cum_rotation_deg" not in res.metrics.columns
    assert "mean_ang_vel_dps" not in default_stat_metrics(False)
    assert res.metrics["path_length"].gt(0).all()


def test_common_window_equalises_observation_time():
    ds = generate_dataset(n_participants=6, seed=9)
    res = run_analysis(ds, AnalysisConfig(window_s=45.0))
    assert res.metrics["duration_s"].between(44.0, 45.0 + 1e-6).all()
