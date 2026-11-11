"""Spatial occupancy, Shannon entropy, coverage and region revisitation.

Spatial discretisation
----------------------
Positions are projected onto the horizontal plane ``(u, v)`` and binned on a
regular square grid of side ``cell_size`` that spans a *common* bounding box
shared by all recordings (so cells are comparable across participants).
``K = nx * ny`` is the number of cells of that grid.

Occupancy
---------
The occupancy probability of cell ``k`` is the fraction of *observed time*
spent there, ``p_k = T_k / sum_j T_j``. Each sample is weighted by its
trapezoidal duration, so irregular sampling does not bias the estimate and
tracking gaps (steps longer than ``max_dt``) do not count as observed time.

Entropy
-------
``H = -sum_k p_k log2 p_k`` (plug-in / maximum-likelihood estimator, bits).

* ``H`` depends on the grid resolution (finer cells -> larger H) and on the
  observation duration (longer recordings reach more cells). It must only be
  compared between recordings analysed with the same grid and window.
* The normalisation ``H / log2 K`` is mathematically valid because
  ``H <= log2 K`` with equality for a uniform distribution over all K cells.
  It is only *interpretable* if all K cells are reachable; cells that cannot
  be visited (walls, outside the arena) make a value of 1 unattainable.
* Entropy measures the *evenness* of time allocation, not exploration quality.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpatialGrid:
    u_min: float
    v_min: float
    cell_size: float
    nx: int
    ny: int

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    @property
    def u_edges(self) -> np.ndarray:
        return self.u_min + self.cell_size * np.arange(self.nx + 1)

    @property
    def v_edges(self) -> np.ndarray:
        return self.v_min + self.cell_size * np.arange(self.ny + 1)

    @classmethod
    def from_bounds(cls, u_min: float, u_max: float, v_min: float, v_max: float, cell_size: float) -> "SpatialGrid":
        if not cell_size > 0:
            raise ValueError("cell_size must be positive.")
        if not (np.isfinite([u_min, u_max, v_min, v_max]).all()):
            raise ValueError("Grid bounds must be finite.")
        nx = max(1, int(np.ceil((u_max - u_min) / cell_size - 1e-9)))
        ny = max(1, int(np.ceil((v_max - v_min) / cell_size - 1e-9)))
        return cls(float(u_min), float(v_min), float(cell_size), nx, ny)

    def cell_ids(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Flat cell index ``iy * nx + ix``. Points outside are clipped to the border cell."""
        ix = np.clip(np.floor((np.asarray(u, float) - self.u_min) / self.cell_size).astype(int), 0, self.nx - 1)
        iy = np.clip(np.floor((np.asarray(v, float) - self.v_min) / self.cell_size).astype(int), 0, self.ny - 1)
        return iy * self.nx + ix


def bounds_from_data(df: pd.DataFrame, u_col: str, v_col: str) -> tuple[float, float, float, float]:
    """Bounding box of all finite samples (used when no arena size is known)."""
    u = df[u_col].to_numpy(dtype=float)
    v = df[v_col].to_numpy(dtype=float)
    ok = np.isfinite(u) & np.isfinite(v)
    if not ok.any():
        raise ValueError("No finite positions to derive bounds from.")
    return float(u[ok].min()), float(u[ok].max()), float(v[ok].min()), float(v[ok].max())


# --------------------------------------------------------------------------- #
# Occupancy
# --------------------------------------------------------------------------- #
def sample_durations(t: np.ndarray, max_dt: float | None = None) -> np.ndarray:
    """Trapezoidal duration represented by each sample (sums to the observed time)."""
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        raise ValueError("At least 2 samples are required to define sample durations.")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Timestamps must be strictly increasing.")
    if max_dt is not None:
        dt = np.where(dt <= max_dt, dt, 0.0)  # tracking gap: not observed
    w = np.zeros(t.size)
    w[:-1] += 0.5 * dt
    w[1:] += 0.5 * dt
    return w


def occupancy_time(u, v, t, grid: SpatialGrid, max_dt: float | None = None) -> np.ndarray:
    """Seconds spent per cell, shape ``(ny, nx)``."""
    w = sample_durations(t, max_dt)
    flat = grid.cell_ids(u, v)
    out = np.bincount(flat, weights=w, minlength=grid.n_cells)
    return out.reshape(grid.ny, grid.nx)


def _probabilities(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0 or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Occupancy weights must be finite and non-negative.")
    total = w.sum()
    if total <= 0:
        raise ValueError("Occupancy is empty (total observed time is zero).")
    return w / total


def shannon_entropy(weights, base: float = 2.0) -> float:
    """Shannon entropy ``-sum p log p`` of an (unnormalised) occupancy array."""
    p = _probabilities(weights)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(base))


def max_entropy(n_cells: int, base: float = 2.0) -> float:
    """Upper bound ``log_b K`` reached by a uniform distribution over K cells."""
    return float(np.log(n_cells) / np.log(base)) if n_cells > 0 else np.nan


def normalized_entropy(weights, n_cells: int | None = None, base: float = 2.0) -> float:
    """``H / log_b(K)``; NaN when ``K < 2`` (normaliser would be zero)."""
    w = np.asarray(weights, dtype=float).ravel()
    k = int(w.size if n_cells is None else n_cells)
    if k < 2:
        return float("nan")
    return shannon_entropy(w, base) / max_entropy(k, base)


def spatial_coverage(weights, n_cells: int | None = None) -> float:
    """Fraction of the K grid cells that were visited at all."""
    w = np.asarray(weights, dtype=float).ravel()
    k = int(w.size if n_cells is None else n_cells)
    return float(np.count_nonzero(w > 0) / k)


def hysteresis_cell_ids(u, v, grid: SpatialGrid, margin_frac: float = 0.25) -> np.ndarray:
    """Cell sequence that only changes cell once the position is clearly outside the current one.

    The current cell is kept while the point lies within its box enlarged by
    ``margin_frac * cell_size`` on every side. This suppresses boundary
    flicker (a path wobbling across a cell edge) that would otherwise be
    counted as many "revisits". ``margin_frac = 0`` reproduces plain binning.
    """
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    plain = grid.cell_ids(u, v)
    if margin_frac <= 0 or u.size == 0:
        return plain
    m = margin_frac * grid.cell_size
    out = np.empty(u.size, dtype=plain.dtype)
    cur = plain[0]
    cs = grid.cell_size
    for i in range(u.size):
        ix, iy = cur % grid.nx, cur // grid.nx
        u0, v0 = grid.u_min + ix * cs, grid.v_min + iy * cs
        if not (u0 - m <= u[i] < u0 + cs + m and v0 - m <= v[i] < v0 + cs + m):
            cur = plain[i]
        out[i] = cur
    return out


def revisit_stats(cell_ids: np.ndarray, t: np.ndarray, min_visit_s: float = 0.0) -> dict:
    """Visit sequence statistics.

    A *visit* is a maximal run of consecutive samples in the same cell. A
    *revisit* is a visit to a cell that was already visited earlier (and left
    in between). Visits shorter than ``min_visit_s`` can be discarded to reduce
    boundary flicker; neighbouring visits of the same cell are then merged.
    """
    cell_ids = np.asarray(cell_ids)
    t = np.asarray(t, dtype=float)
    if cell_ids.size == 0:
        raise ValueError("Empty cell sequence.")
    change = np.flatnonzero(np.diff(cell_ids) != 0) + 1
    starts = np.concatenate([[0], change])
    ids = cell_ids[starts]
    t_ends = np.concatenate([t[starts[1:]], [t[-1]]])
    dur = t_ends - t[starts]
    if min_visit_s > 0:
        keep = dur >= min_visit_s
        ids = ids[keep]
        if ids.size:
            ids = ids[np.concatenate([[True], ids[1:] != ids[:-1]])]
    n_visits = int(ids.size)
    n_unique = int(np.unique(ids).size)
    n_rev = n_visits - n_unique
    minutes = max(float(t[-1] - t[0]), 1e-12) / 60.0
    return {
        "n_visits": n_visits,
        "n_unique_cells": n_unique,
        "n_revisits": n_rev,
        "revisit_ratio": n_rev / n_visits if n_visits else 0.0,
        "revisits_per_min": n_rev / minutes,
    }


def spatial_metrics(u, v, t, grid: SpatialGrid, max_dt: float | None = None, min_visit_s: float = 0.0,
                    hysteresis: float = 0.25) -> dict:
    """All exploration metrics of one recording on a given grid.

    Occupancy / entropy / coverage use plain binning; revisit statistics use the
    hysteresis cell sequence (see :func:`hysteresis_cell_ids`).
    """
    occ = occupancy_time(u, v, t, grid, max_dt)
    h = shannon_entropy(occ)
    res = {
        "entropy_bits": h,
        "entropy_norm": normalized_entropy(occ, grid.n_cells),
        "perplexity": float(2.0 ** h),
        "coverage": spatial_coverage(occ, grid.n_cells),
        "n_cells": int(grid.n_cells),
        "max_entropy_bits": max_entropy(grid.n_cells),
    }
    rv = revisit_stats(hysteresis_cell_ids(u, v, grid, hysteresis), np.asarray(t, float), min_visit_s)
    # `n_unique_cells` is the number of occupied cells (the numerator of coverage). The visit sequence is hysteresis-
    # filtered, so its distinct-cell count can be smaller; it is reported separately to avoid mixing the two.
    rv["n_cells_in_visits"] = rv.pop("n_unique_cells")
    res.update(rv)
    res["n_unique_cells"] = int(np.count_nonzero(occ > 0))
    return res


def spatial_metrics_table(df: pd.DataFrame, grid: SpatialGrid, u_col: str, v_col: str,
                          gap_factor: float = 10.0, min_visit_s: float = 0.0,
                          key_cols=("participant_id", "condition")) -> pd.DataFrame:
    """One row of exploration metrics per recording (rows must be pre-validated)."""
    rows = []
    for key, g in df.groupby(list(key_cols), sort=True):
        g = g.sort_values("timestamp", kind="stable")
        t = g["timestamp"].to_numpy(dtype=float)
        if t.size < 2:
            continue
        max_dt = gap_factor * float(np.median(np.diff(t)))
        row = dict(zip(key_cols, key))
        row.update(spatial_metrics(g[u_col].to_numpy(), g[v_col].to_numpy(), t, grid, max_dt, min_visit_s))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sensitivity analyses
# --------------------------------------------------------------------------- #
def entropy_vs_resolution(u, v, t, bounds, cell_sizes, max_dt: float | None = None) -> pd.DataFrame:
    """Entropy / coverage of one recording as a function of the cell size."""
    rows = []
    for cs in cell_sizes:
        grid = SpatialGrid.from_bounds(*bounds, cell_size=float(cs))
        occ = occupancy_time(u, v, t, grid, max_dt)
        rows.append(
            {
                "cell_size": float(cs),
                "n_cells": grid.n_cells,
                "entropy_bits": shannon_entropy(occ),
                "entropy_norm": normalized_entropy(occ, grid.n_cells),
                "coverage": spatial_coverage(occ, grid.n_cells),
            }
        )
    return pd.DataFrame(rows)


def entropy_vs_duration(u, v, t, grid: SpatialGrid, durations, max_dt: float | None = None) -> pd.DataFrame:
    """Entropy / coverage using only the first ``T`` seconds, for each ``T``."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    t = np.asarray(t, float)
    t_rel = t - t[0]
    rows = []
    for T in durations:
        m = t_rel <= T
        if m.sum() < 2:
            continue
        occ = occupancy_time(u[m], v[m], t[m], grid, max_dt)
        if occ.sum() <= 0:
            continue
        rows.append(
            {
                "window_s": float(T),
                "entropy_bits": shannon_entropy(occ),
                "entropy_norm": normalized_entropy(occ, grid.n_cells),
                "coverage": spatial_coverage(occ, grid.n_cells),
            }
        )
    return pd.DataFrame(rows)
