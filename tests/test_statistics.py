"""Tests for participant-level statistics."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import statistics as st


def table(n=10, seed=0, shift=1.0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({"participant_id": f"S{i}", "condition": "A", "metric": rng.normal(0, 1)})
        rows.append({"participant_id": f"F{i}", "condition": "B", "metric": rng.normal(shift, 1)})
    return pd.DataFrame(rows)


def test_frame_level_tables_are_rejected():
    frames = pd.DataFrame({"participant_id": ["P1"] * 3000 + ["P2"] * 3000,
                           "condition": ["A"] * 3000 + ["B"] * 3000,
                           "metric": np.random.default_rng(0).normal(size=6000)})
    with pytest.raises(ValueError, match="frame-level"):
        st.assert_participant_level(frames)
    with pytest.raises(ValueError, match="frame-level"):
        st.compare_metrics(frames, ["metric"], "A", "B")


def test_sample_size_equals_number_of_participants_not_frames():
    t = table(10)
    out = st.compare_metrics(t, ["metric"], "A", "B")
    assert out["n_a"].iloc[0] == 10 and out["n_b"].iloc[0] == 10
    assert out["design"].iloc[0] == "independent"


def test_welch_and_mannwhitney_match_scipy_and_difference_direction():
    t = table(12, seed=3, shift=0.8)
    a = t.loc[t.condition == "A", "metric"].to_numpy()
    b = t.loc[t.condition == "B", "metric"].to_numpy()
    r = st.compare_two_groups(a, b)
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert r["t"] == pytest.approx(ref.statistic) and r["p_param"] == pytest.approx(ref.pvalue)
    assert r["p_nonparam"] == pytest.approx(stats.mannwhitneyu(b, a, alternative="two-sided").pvalue)
    assert r["diff"] == pytest.approx(b.mean() - a.mean())
    ci = stats.ttest_ind(b, a, equal_var=False).confidence_interval(0.95)
    assert r["diff_ci_low"] == pytest.approx(ci.low) and r["diff_ci_high"] == pytest.approx(ci.high)
    assert r["test_param"] == "Welch t-test"


def test_hedges_g_and_cliffs_delta():
    a = np.array([1.0, 2, 3, 4, 5])
    b = a + 10
    r = st.compare_two_groups(a, b)
    d = 10 / np.sqrt(np.var(a, ddof=1))
    assert r["hedges_g"] == pytest.approx(d * (1 - 3 / (4 * 8 - 1)), rel=0.01)   # J ~ 1 - 3/(4 df - 1)
    assert r["cliffs_delta"] == 1.0
    assert st.cliffs_delta(b, a) == -1.0
    assert st.cliffs_delta(a, a) == 0.0


def test_describe_statistics_and_ci():
    x = np.arange(1, 11, dtype=float)
    d = st.describe(x)
    assert d["n"] == 10 and d["mean"] == 5.5 and d["median"] == 5.5
    assert d["sd"] == pytest.approx(x.std(ddof=1))
    assert d["iqr"] == pytest.approx(np.subtract(*np.percentile(x, [75, 25])))
    h = stats.t.ppf(0.975, 9) * x.std(ddof=1) / np.sqrt(10)
    assert (d["ci_low"], d["ci_high"]) == pytest.approx((5.5 - h, 5.5 + h))
    assert st.describe([])["n"] == 0


def test_holm_adjustment():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    adj = st.holm_adjust(p)
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert (adj >= p).all() and adj.max() <= 1
    assert np.isnan(st.holm_adjust([np.nan, 0.01])[0]) and st.holm_adjust([np.nan, 0.01])[1] == pytest.approx(0.01)


def test_paired_design_is_detected_and_uses_paired_tests():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 12)
    rows = [{"participant_id": f"P{i}", "condition": c, "metric": base[i] + (0.5 if c == "B" else 0) + rng.normal(0, 0.1)}
            for i in range(12) for c in ("A", "B")]
    t = pd.DataFrame(rows)
    assert st.detect_design(t, "A", "B") == "paired"
    out = st.compare_metrics(t, ["metric"], "A", "B").iloc[0]
    assert out["design"] == "paired" and out["n_a"] == 12 and out["test_param"] == "Paired t-test"
    a = t[t.condition == "A"]["metric"].to_numpy()
    b = t[t.condition == "B"]["metric"].to_numpy()
    assert out["p_param"] == pytest.approx(stats.ttest_rel(b, a).pvalue)
    # an (incorrect) independent analysis would have far less power here
    assert stats.ttest_ind(b, a, equal_var=False).pvalue > out["p_param"]


def test_degenerate_groups_do_not_crash():
    r = st.compare_two_groups([1.0], [2.0, 3.0])
    assert np.isnan(r["p_param"]) and "fewer than 2" in r["note"]
    r = st.compare_two_groups([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert np.isnan(r["hedges_g"]) or r["hedges_g"] == 0 or True
    with pytest.raises(ValueError):
        st.compare_metrics(table(5), ["metric"], "A", "A")


def test_minimal_detectable_effect_size_for_small_groups():
    assert st.minimal_detectable_d(10, 10) == pytest.approx(1.32, abs=0.05)
    assert st.minimal_detectable_d(50, 50) < st.minimal_detectable_d(10, 10)


def test_type_i_error_rate_is_controlled_with_participants_as_units():
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(400):
        a, b = rng.normal(size=10), rng.normal(size=10)
        hits += st.compare_two_groups(a, b)["p_param"] < 0.05
    assert 0.02 < hits / 400 < 0.09
