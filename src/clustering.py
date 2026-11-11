"""Unsupervised analyses: point-level DBSCAN and participant-level clustering.

Two *different* questions
-------------------------
**Point-level (DBSCAN on positions).** Every 6-DoF sample is a point in space.
DBSCAN groups points that are densely packed. Because samples are equally
spaced in *time*, density is high wherever a person lingers or moves slowly, so
clusters are best read as **dwell regions** - not as "behavioural states" or
"strategies". Samples of one participant are strongly autocorrelated, so the
number of points says nothing about the number of independent observations.

**Participant-level (k-means on behavioural feature vectors).** Each
participant is one point described by summary metrics (speed, rotation,
entropy, ...). This asks whether participants group into behavioural *types*
and is the only level at which cluster-vs-condition agreement is meaningful.

Neither result is validated ground truth. Cluster labels depend on the
parameters (eps, min_samples, scaling) - see :func:`eps_sensitivity`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SCALERS = {"none": None, "standard": StandardScaler, "minmax": MinMaxScaler}
MAX_POINTS = 60_000
MAX_NEIGHBOUR_PAIRS = 40_000_000


@dataclass
class DBSCANResult:
    labels: np.ndarray                 # -1 = noise
    n_clusters: int
    n_noise: int
    noise_fraction: float
    sizes: dict[int, int]              # cluster id -> n points
    n_core: int
    silhouette: float | None           # on non-noise points (sampled), None if undefined
    eps: float
    min_samples: int
    scaling: str


def scale_features(X: np.ndarray, scaling: str = "none") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if scaling not in SCALERS:
        raise ValueError(f"scaling must be one of {list(SCALERS)}, got {scaling!r}")
    if scaling == "none":
        return X
    return SCALERS[scaling]().fit_transform(X)


def _check_input(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("DBSCAN needs a non-empty 2-D array of points.")
    if not np.isfinite(X).all():
        raise ValueError("DBSCAN input contains NaN or inf.")
    if not eps > 0:
        raise ValueError("eps must be > 0.")
    if int(min_samples) < 1:
        raise ValueError("min_samples must be >= 1.")
    if X.shape[0] > MAX_POINTS:
        raise ValueError(f"{X.shape[0]} points exceed the limit of {MAX_POINTS}. Select fewer participants "
                         "or increase the sample stride (this is an explicit, visible analysis choice).")
    return X


def _estimate_pairs(X: np.ndarray, eps: float, seed: int = 0) -> float:
    """Rough number of neighbour pairs DBSCAN would materialise (guards against memory blow-ups)."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n, 300), replace=False)
    nn = NearestNeighbors(radius=eps).fit(X)
    counts = [len(a) for a in nn.radius_neighbors(X[idx], return_distance=False)]
    return float(np.mean(counts)) * n


def run_dbscan(X, eps: float, min_samples: int, scaling: str = "none", silhouette_sample: int = 5000,
               seed: int = 0) -> DBSCANResult:
    """DBSCAN with optional feature scaling and basic validation measures."""
    X = _check_input(X, eps, min_samples)
    Xs = scale_features(X, scaling)
    pairs = _estimate_pairs(Xs, eps, seed)
    if pairs > MAX_NEIGHBOUR_PAIRS:
        raise ValueError(f"eps = {eps:g} would create ~{pairs:,.0f} neighbour pairs (too many for a laptop). "
                         "Reduce eps, select fewer participants, or increase the sample stride.")
    labels = DBSCAN(eps=eps, min_samples=int(min_samples), algorithm="kd_tree").fit(Xs)
    lab = labels.labels_
    uniq, counts = np.unique(lab[lab >= 0], return_counts=True)
    sizes = {int(u): int(c) for u, c in sorted(zip(uniq, counts), key=lambda kv: -kv[1])}
    sil = None
    mask = lab >= 0
    if len(uniq) >= 2 and mask.sum() > len(uniq):
        sil = float(silhouette_score(Xs[mask], lab[mask], sample_size=min(int(mask.sum()), silhouette_sample),
                                     random_state=seed))
    return DBSCANResult(
        labels=lab, n_clusters=int(len(uniq)), n_noise=int((lab < 0).sum()),
        noise_fraction=float((lab < 0).mean()), sizes=sizes, n_core=int(len(labels.core_sample_indices_)),
        silhouette=sil, eps=float(eps), min_samples=int(min_samples), scaling=scaling,
    )


def cluster_summary(points: pd.DataFrame, labels: np.ndarray, coord_cols, median_dt: float | None = None) -> pd.DataFrame:
    """Per-cluster size, centroid, spatial spread and (optionally) dwell time."""
    p = points.reset_index(drop=True).copy()
    p["_label"] = labels
    rows = []
    for lab, g in p.groupby("_label"):
        c = g[list(coord_cols)].to_numpy(float)
        row = {
            "cluster": "noise" if lab < 0 else int(lab),
            "n_points": int(len(g)),
            "share_%": 100.0 * len(g) / len(p),
            "rms_radius": float(np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean())),
        }
        for k, col in enumerate(coord_cols):
            row[f"centroid_{col.replace('position_', '')}"] = float(c[:, k].mean())
        if "participant_id" in g:
            row["n_participants"] = int(g["participant_id"].nunique())
        if median_dt:
            row["dwell_time_s"] = float(len(g) * median_dt)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["_o"] = out["cluster"].map(lambda c: 1e9 if c == "noise" else 0)
    return out.sort_values(["_o", "n_points"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)


def k_distance(X, k: int, scaling: str = "none") -> np.ndarray:
    """Sorted distance of each point to its k-th nearest neighbour (self excluded) - the usual eps heuristic."""
    X = scale_features(np.asarray(X, float), scaling)
    k = int(min(max(k, 1), X.shape[0] - 1))
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return np.sort(d[:, -1])


def eps_sensitivity(X, eps_values, min_samples: int, scaling: str = "none") -> pd.DataFrame:
    """How strongly do the results depend on eps? (a basic stability check)"""
    rows = []
    for e in eps_values:
        try:
            r = run_dbscan(X, float(e), min_samples, scaling, silhouette_sample=2000)
            rows.append({"eps": float(e), "n_clusters": r.n_clusters, "noise_%": 100 * r.noise_fraction,
                         "largest_cluster_%": 100 * (max(r.sizes.values()) / len(r.labels)) if r.sizes else 0.0})
        except ValueError as exc:
            rows.append({"eps": float(e), "n_clusters": np.nan, "noise_%": np.nan, "largest_cluster_%": np.nan,
                         "note": str(exc)[:60]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Participant level
# --------------------------------------------------------------------------- #
def participant_level_clustering(table: pd.DataFrame, feature_cols: list[str], condition_col: str = "condition",
                                 k_range=(2, 5), seed: int = 0) -> dict:
    """Standardise participant feature vectors, pick k by silhouette, project with PCA.

    Returns a dict with ``labels``, ``best_k``, ``silhouettes`` (k -> score),
    ``pca`` (N x 2), ``explained`` and ``ari_vs_condition`` (agreement with the
    experimental label - only a sanity check, not validation).
    """
    if len(feature_cols) < 2:
        raise ValueError("Select at least two features.")
    X = table[feature_cols].to_numpy(float)
    ok = np.isfinite(X).all(axis=1)
    if ok.sum() < 4:
        raise ValueError("Need at least 4 participants with complete feature vectors.")
    Xs = StandardScaler().fit_transform(X[ok])
    ks = [k for k in range(k_range[0], k_range[1] + 1) if k < ok.sum()]
    sil, fits = {}, {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
        fits[k] = km.labels_
        sil[k] = float(silhouette_score(Xs, km.labels_)) if len(set(km.labels_)) > 1 else float("nan")
    best_k = max(sil, key=lambda k: (np.nan_to_num(sil[k], nan=-1.0), -k))
    pca = PCA(n_components=2, random_state=seed).fit(Xs)
    labels = np.full(len(table), -1)
    labels[ok] = fits[best_k]
    ari = float(adjusted_rand_score(table.loc[ok, condition_col].astype(str), fits[best_k])) \
        if table[condition_col].nunique() > 1 else float("nan")
    emb = np.full((len(table), 2), np.nan)
    emb[ok] = pca.transform(Xs)
    return {"labels": labels, "best_k": int(best_k), "silhouettes": sil, "pca": emb,
            "explained": pca.explained_variance_ratio_.tolist(), "ari_vs_condition": ari,
            "loadings": pd.DataFrame(pca.components_.T, index=feature_cols, columns=["PC1", "PC2"])}
