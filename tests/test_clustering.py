"""Tests for DBSCAN wrapper and participant-level clustering."""
import numpy as np
import pandas as pd
import pytest

from src import clustering as cl


def two_blobs(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal([2, 2, 1.6], 0.1, (200, 3))
    b = rng.normal([7, 7, 1.6], 0.1, (200, 3))
    noise = np.array([[5.0, 0.2, 1.6], [0.2, 9.5, 1.6], [9.5, 0.3, 1.6], [4.5, 9.0, 1.6], [9.7, 9.7, 1.6]])
    return np.vstack([a, b, noise])


def test_dbscan_finds_clusters_and_labels_isolated_points_as_noise():
    X = two_blobs()
    r = cl.run_dbscan(X, eps=0.3, min_samples=10)
    assert r.n_clusters == 2
    assert r.n_noise == 5
    assert (r.labels[-5:] == -1).all()                       # the isolated points are noise
    assert sorted(r.sizes.values()) == [200, 200]
    assert r.noise_fraction == pytest.approx(5 / 405)
    assert r.silhouette is not None and r.silhouette > 0.8


def test_everything_is_noise_when_min_samples_is_too_large():
    r = cl.run_dbscan(two_blobs(), eps=0.3, min_samples=500)
    assert r.n_clusters == 0 and r.noise_fraction == 1.0 and r.silhouette is None


def test_eps_controls_merging():
    X = two_blobs()
    assert cl.run_dbscan(X, eps=0.3, min_samples=10).n_clusters == 2
    assert cl.run_dbscan(X, eps=8.0, min_samples=10).n_clusters == 1


def test_scaling_changes_the_geometry():
    X = two_blobs()
    # z is nearly constant; standardising blows its tiny spread up to unit variance
    raw = cl.run_dbscan(X, eps=0.3, min_samples=10, scaling="none")
    std = cl.run_dbscan(X, eps=0.3, min_samples=10, scaling="standard")
    assert raw.scaling == "none" and std.scaling == "standard"
    assert std.n_clusters != raw.n_clusters or std.n_noise != raw.n_noise
    assert cl.scale_features(X, "minmax").max() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cl.scale_features(X, "bogus")


@pytest.mark.parametrize("X,eps,ms", [
    (np.zeros((0, 3)), 0.3, 5), (np.array([[0, 0, np.nan]]), 0.3, 5), (np.zeros((10, 3)), 0.0, 5), (np.zeros((10, 3)), 0.3, 0),
])
def test_invalid_inputs_raise(X, eps, ms):
    with pytest.raises(ValueError):
        cl.run_dbscan(X, eps, ms)


def test_too_many_neighbour_pairs_is_refused_not_crashed():
    X = np.zeros((50_000, 3)) + np.random.default_rng(0).normal(0, 0.01, (50_000, 3))
    with pytest.raises(ValueError, match="neighbour pairs"):
        cl.run_dbscan(X, eps=1.0, min_samples=5)


def test_cluster_summary_and_k_distance_and_sensitivity():
    X = two_blobs()
    r = cl.run_dbscan(X, 0.3, 10)
    pts = pd.DataFrame(X, columns=["position_x", "position_y", "position_z"])
    pts["participant_id"] = "A"
    s = cl.cluster_summary(pts, r.labels, ["position_x", "position_y"], median_dt=1 / 30)
    assert s["n_points"].sum() == len(X) and s.iloc[-1]["cluster"] == "noise"
    assert s["share_%"].sum() == pytest.approx(100.0)
    kd = cl.k_distance(X, 10)
    assert len(kd) == len(X) and (np.diff(kd) >= 0).all()
    sens = cl.eps_sensitivity(X, [0.1, 0.3, 8.0], 10)
    assert sens["n_clusters"].tolist()[1] == 2 and sens["n_clusters"].tolist()[2] == 1


def test_participant_level_clustering_is_separate_and_recovers_groups():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.3, (10, 4)) + [3, 3, 0, 0]
    b = rng.normal(0, 0.3, (10, 4)) - [3, 3, 0, 0]
    tab = pd.DataFrame(np.vstack([a, b]), columns=["f1", "f2", "f3", "f4"])
    tab["condition"] = ["S"] * 10 + ["F"] * 10
    out = cl.participant_level_clustering(tab, ["f1", "f2", "f3", "f4"], k_range=(2, 4))
    assert out["best_k"] == 2
    assert out["ari_vs_condition"] == pytest.approx(1.0)
    assert out["pca"].shape == (20, 2) and len(out["labels"]) == 20
    with pytest.raises(ValueError):
        cl.participant_level_clustering(tab.iloc[:3], ["f1", "f2"])
