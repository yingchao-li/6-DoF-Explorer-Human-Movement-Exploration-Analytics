"""Tests for spatial occupancy, entropy, coverage and revisitation."""
import numpy as np
import pandas as pd
import pytest

from src import entropy as ent


def grid10(cell=1.0):
    return ent.SpatialGrid.from_bounds(0, 10, 0, 10, cell)


def test_entropy_is_zero_for_a_single_occupied_bin():
    t = np.arange(200) / 30
    u = np.full(200, 3.5)
    v = np.full(200, 4.5)
    occ = ent.occupancy_time(u, v, t, grid10())
    assert np.count_nonzero(occ) == 1
    assert ent.shannon_entropy(occ) == 0.0
    assert ent.normalized_entropy(occ, 100) == 0.0
    assert ent.spatial_coverage(occ) == pytest.approx(0.01)


def test_uniform_occupancy_gives_log2_k_and_normalised_one():
    occ = np.ones((10, 10))
    assert ent.shannon_entropy(occ) == pytest.approx(np.log2(100))
    assert ent.normalized_entropy(occ) == pytest.approx(1.0)
    assert ent.spatial_coverage(occ) == 1.0
    assert ent.max_entropy(100) == pytest.approx(np.log2(100))


def test_entropy_never_exceeds_log2_k():
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = rng.random(37) * (rng.random(37) < 0.7)
        if w.sum() == 0:
            continue
        assert ent.shannon_entropy(w) <= np.log2(37) + 1e-12
        assert 0 <= ent.normalized_entropy(w) <= 1 + 1e-12


def test_known_two_cell_entropy_and_base_conversion():
    w = np.array([3.0, 1.0])
    p = np.array([0.75, 0.25])
    h = -(p * np.log2(p)).sum()
    assert ent.shannon_entropy(w) == pytest.approx(h)
    assert ent.shannon_entropy(w, base=np.e) == pytest.approx(h * np.log(2))


def test_normalisation_undefined_for_one_cell():
    assert np.isnan(ent.normalized_entropy(np.array([5.0])))


def test_invalid_occupancy_raises():
    with pytest.raises(ValueError):
        ent.shannon_entropy(np.zeros(5))
    with pytest.raises(ValueError):
        ent.shannon_entropy(np.array([1.0, -1.0]))
    with pytest.raises(ValueError):
        ent.shannon_entropy(np.array([1.0, np.nan]))


def test_occupancy_is_time_weighted_for_irregular_sampling():
    # trapezoid sample durations: 1.5, 1.75, 0.5, 0.25 s  -> cell A (first two samples) 3.25 s, cell B 0.75 s
    t = np.array([0.0, 3.0, 3.5, 4.0])
    u = np.array([0.5, 0.5, 1.5, 1.5])
    v = np.zeros(4) + 0.5
    occ = ent.occupancy_time(u, v, t, grid10())
    assert occ.sum() == pytest.approx(4.0)               # total observed time
    assert occ[0, 0] == pytest.approx(3.25)
    assert occ[0, 1] == pytest.approx(0.75)
    # the naive sample-count estimate would give 2 vs 2 samples = exactly 1 bit
    assert ent.shannon_entropy(occ) == pytest.approx(-(0.8125 * np.log2(0.8125) + 0.1875 * np.log2(0.1875)))
    assert ent.shannon_entropy(occ) < 1.0


def test_tracking_gap_time_is_not_counted_as_occupancy():
    t = np.array([0.0, 0.1, 0.2, 10.2, 10.3, 10.4])
    u = np.array([0.5, 0.5, 0.5, 5.5, 5.5, 5.5])
    v = np.full(6, 0.5)
    occ = ent.occupancy_time(u, v, t, grid10(), max_dt=1.0)
    assert occ.sum() == pytest.approx(0.4)
    assert occ.sum() < 1.0


def test_grid_geometry_and_clipping():
    g = ent.SpatialGrid.from_bounds(0, 10, 0, 5, 2.0)
    assert (g.nx, g.ny, g.n_cells) == (5, 3, 15)
    ids = g.cell_ids([0.0, 9.99, 10.0, -3.0, 3.0], [0.0, 4.99, 5.0, 100.0, 2.0])
    assert ids.tolist() == [0, 2 * 5 + 4, 2 * 5 + 4, 2 * 5 + 0, 1 * 5 + 1]
    with pytest.raises(ValueError):
        ent.SpatialGrid.from_bounds(0, 1, 0, 1, 0)


def test_refining_a_nested_grid_cannot_decrease_entropy():
    rng = np.random.default_rng(3)
    n = 4000
    xy = np.clip(np.cumsum(rng.normal(0, 0.1, (n, 2)), axis=0) + 5, 0, 10)
    t = np.arange(n) / 30
    out = ent.entropy_vs_resolution(xy[:, 0], xy[:, 1], t, (0, 10, 0, 10), [5, 2.5, 1.25, 0.625])
    assert out["n_cells"].tolist() == [4, 16, 64, 256]
    assert (np.diff(out["entropy_bits"]) >= -1e-12).all()   # refinement of a partition never lowers entropy
    assert (out["entropy_norm"] <= 1 + 1e-12).all()


def test_entropy_and_coverage_grow_with_observation_window():
    rng = np.random.default_rng(4)
    n = 3000
    xy = np.clip(np.cumsum(rng.normal(0, 0.12, (n, 2)), axis=0) + 5, 0, 10)
    t = np.arange(n) / 30
    out = ent.entropy_vs_duration(xy[:, 0], xy[:, 1], t, grid10(), [5, 20, 50, 100])
    assert (np.diff(out["coverage"]) >= 0).all()          # visited cells can only accumulate


def test_revisit_statistics():
    ids = np.array([1, 1, 2, 2, 3, 1, 1, 2, 4])
    t = np.arange(len(ids), dtype=float)
    r = ent.revisit_stats(ids, t)
    assert r["n_visits"] == 6                              # 1 2 3 1 2 4
    assert r["n_unique_cells"] == 4
    assert r["n_revisits"] == 2
    assert r["revisit_ratio"] == pytest.approx(2 / 6)
    assert ent.revisit_stats(np.array([5, 5, 5]), np.arange(3.0))["n_revisits"] == 0


def test_min_visit_duration_merges_flicker():
    ids = np.array([1] * 10 + [2] + [1] * 10)
    t = np.arange(len(ids), dtype=float)
    assert ent.revisit_stats(ids, t)["n_revisits"] == 1
    assert ent.revisit_stats(ids, t, min_visit_s=3.0)["n_revisits"] == 0


def test_hysteresis_suppresses_boundary_flicker():
    g = grid10()
    n = 300
    u = 1.0 + 0.02 * np.sin(np.linspace(0, 40 * np.pi, n))   # wobbling across the x = 1 cell edge
    v = np.full(n, 0.5)
    t = np.arange(n) / 30
    plain = ent.revisit_stats(g.cell_ids(u, v), t)
    hyst = ent.revisit_stats(ent.hysteresis_cell_ids(u, v, g, 0.25), t)
    assert plain["n_revisits"] > 10
    assert hyst["n_revisits"] == 0
    np.testing.assert_array_equal(ent.hysteresis_cell_ids(u, v, g, 0.0), g.cell_ids(u, v))


def test_spatial_metrics_and_table():
    n = 600
    t = np.arange(n) / 30
    u = (np.arange(n) + 0.5) / n * 10.0                 # evenly spread over 10 cells
    v = np.full(n, 5.5)
    m = ent.spatial_metrics(u, v, t, grid10())
    assert m["coverage"] == pytest.approx(0.10)
    assert m["entropy_bits"] == pytest.approx(np.log2(10), abs=0.02)
    assert m["perplexity"] == pytest.approx(10, rel=0.02)
    assert m["n_revisits"] == 0
    df = pd.DataFrame({"participant_id": "A", "condition": "c", "timestamp": t, "position_x": u, "position_y": v})
    tab = ent.spatial_metrics_table(df, grid10(), "position_x", "position_y")
    assert len(tab) == 1 and tab["entropy_bits"].iloc[0] == pytest.approx(m["entropy_bits"])


def test_spatial_metrics_occupied_cells_match_coverage_and_visit_count_is_separate():
    rng = np.random.default_rng(11)
    n = 1500
    xy = np.clip(np.cumsum(rng.normal(0, 0.15, (n, 2)), axis=0) + 5, 0, 10)
    t = np.arange(n) / 30
    g = grid10()
    r = ent.spatial_metrics(xy[:, 0], xy[:, 1], t, g)
    assert r["n_unique_cells"] == round(r["coverage"] * r["n_cells"])      # same quantity as coverage, by construction
    assert r["n_cells_in_visits"] <= r["n_unique_cells"]                    # hysteresis can only merge, never add, cells
