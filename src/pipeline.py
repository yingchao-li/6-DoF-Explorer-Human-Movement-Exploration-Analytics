"""End-to-end analysis pipeline (no UI code).

``Dataset`` -> preprocessing (+ QC report) -> optional common analysis window ->
per-sample kinematics -> per-recording metrics (movement + exploration) ->
participant-level statistics.

The Streamlit app calls the three *stages* separately so that each can be
cached independently (changing the grid size does not recompute kinematics).
Run ``python -m src.pipeline`` for a command-line report on the synthetic data.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import entropy as ent
from .data_generator import generate_dataset
from .data_loader import Dataset
from .features import (KEY_COLS, FeatureConfig, exploration_metrics_table, horizontal_axes, kinematics_table,
                        metric_catalogue, movement_metrics_table)
from .preprocessing import PreprocessConfig, QualityReport, preprocess
from .statistics import compare_metrics


@dataclass(frozen=True)
class AnalysisConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    cell_size: float = 1.0
    min_visit_s: float = 0.0
    window_s: float | None = None        # analyse only the first T seconds of every recording


@dataclass
class AnalysisResult:
    clean: pd.DataFrame
    qc: QualityReport
    kin: pd.DataFrame
    skipped: dict
    movement: pd.DataFrame
    grid: ent.SpatialGrid
    exploration: pd.DataFrame
    metrics: pd.DataFrame


# ── stage 1 ───────────────────────────────────────────────────────────────────────────────────
def apply_window(df: pd.DataFrame, window_s: float | None) -> tuple[pd.DataFrame, int]:
    """Keep only samples within ``window_s`` of each recording's start; returns (df, n_shorter_recordings)."""
    if not window_s:
        return df, 0
    t0 = df.groupby(KEY_COLS)["timestamp"].transform("min")
    keep = (df["timestamp"] - t0) <= window_s
    dur = (df.groupby(KEY_COLS)["timestamp"].agg(lambda s: s.max() - s.min()))
    return df[keep].reset_index(drop=True), int((dur < window_s).sum())


def stage_prepare(ds: Dataset, pre_cfg: PreprocessConfig, window_s: float | None = None) -> tuple[pd.DataFrame, QualityReport]:
    clean, qc = preprocess(ds.df, pre_cfg, ds.has_rotation)
    if window_s:
        before = len(clean)
        clean, n_short = apply_window(clean, window_s)
        qc.operations = pd.concat([qc.operations, pd.DataFrame([{
            "operation": "Common analysis window", "setting": f"first {window_s:g} s of each recording",
            "rows_before": before, "rows_after": len(clean), "rows_affected": before - len(clean),
            "effect on analysis": f"All metrics use the same observation time, removing the dependence of entropy / "
                                  f"coverage on recording length. {n_short} recording(s) are shorter than the window "
                                  "and are used in full."}])], ignore_index=True)
    return clean, qc


# ── stage 2 ───────────────────────────────────────────────────────────────────────────────────
def stage_kinematics(clean: pd.DataFrame, has_rotation: bool, feat_cfg: FeatureConfig):
    kin, skipped = kinematics_table(clean, feat_cfg, has_rotation)
    movement = movement_metrics_table(kin, feat_cfg) if len(kin) else pd.DataFrame()
    return kin, skipped, movement


# ── stage 3 ───────────────────────────────────────────────────────────────────────────────────
def make_grid(ds: Dataset, clean: pd.DataFrame, cell_size: float) -> ent.SpatialGrid:
    """Common grid for all recordings: known arena if available, else the data bounding box."""
    u_col, v_col = horizontal_axes(ds.up_axis)
    if ds.arena_bounds is not None:
        b = ds.arena_bounds
    else:
        b = ent.bounds_from_data(clean, u_col, v_col)
    return ent.SpatialGrid.from_bounds(*b, cell_size=cell_size)


def stage_exploration(ds: Dataset, clean: pd.DataFrame, feat_cfg: FeatureConfig, cell_size: float,
                      min_visit_s: float = 0.0):
    grid = make_grid(ds, clean, cell_size)
    u_col, v_col = horizontal_axes(ds.up_axis)
    expl = exploration_metrics_table(clean, grid, u_col, v_col, feat_cfg, min_visit_s)
    return grid, expl


def merge_metrics(movement: pd.DataFrame, exploration: pd.DataFrame) -> pd.DataFrame:
    if movement.empty or exploration.empty:
        return movement
    keep = [c for c in exploration.columns if c not in ("n_visits", "n_revisits")]
    return movement.merge(exploration[keep], on=KEY_COLS, how="inner")


def run_analysis(ds: Dataset, cfg: AnalysisConfig | None = None) -> AnalysisResult:
    cfg = cfg or AnalysisConfig()
    clean, qc = stage_prepare(ds, cfg.preprocess, cfg.window_s)
    kin, skipped, movement = stage_kinematics(clean, ds.has_rotation, cfg.features)
    grid, expl = stage_exploration(ds, clean, cfg.features, cfg.cell_size, cfg.min_visit_s)
    return AnalysisResult(clean, qc, kin, skipped, movement, grid, expl, merge_metrics(movement, expl))


def default_stat_metrics(has_rotation: bool) -> list[str]:
    cols = ["path_length", "mean_speed", "speed_cv", "mean_accel", "straightness", "turn_rate_per_min",
            "entropy_norm", "coverage", "revisit_ratio"]
    if has_rotation:
        cols[5:5] = ["mean_ang_vel_dps", "cum_rotation_deg"]
    return cols


# ── command line ──────────────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the 6-DoF analysis on the synthetic demonstration data.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--participants", type=int, default=20)
    ap.add_argument("--cell-size", type=float, default=1.0)
    ap.add_argument("--artifacts", action="store_true", help="inject synthetic tracking artifacts")
    ap.add_argument("--csv", help="optionally write the synthetic dataset to this CSV path")
    args = ap.parse_args(argv)

    ds = generate_dataset(args.participants, seed=args.seed, inject_artifacts=args.artifacts)
    if args.csv:
        ds.df.to_csv(args.csv, index=False)
    res = run_analysis(ds, AnalysisConfig(cell_size=args.cell_size))
    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("*** SYNTHETIC DEMONSTRATION DATA - not observations from real VR experiments ***\n")
    print(f"dataset: {ds.name}   rows={len(ds.df):,}   recordings={res.qc.totals['n_recordings']}")
    print("data-quality totals:", {k: v for k, v in res.qc.totals.items() if k.startswith("n_") and v})
    print("\noperations:\n", res.qc.operations[["operation", "setting", "rows_affected"]].to_string(index=False))
    cat = metric_catalogue()
    cols = [c for c in default_stat_metrics(True)]
    print("\nmean by condition:\n", res.metrics.groupby("condition")[cols].mean().round(3).T)
    cmp = compare_metrics(res.metrics, cols, "Structured", "Free")
    show = cmp[["metric", "mean_a", "mean_b", "diff", "hedges_g", "p_param", "p_param_holm", "p_nonparam"]].copy()
    show["metric"] = show["metric"].map(lambda m: cat[m]["label"])
    print("\nStructured (a) vs Free (b), participant-level, Holm-adjusted:\n", show.round(4).to_string(index=False))
    print("\nNOTE: significance in synthetic data reflects the simulation parameters, not human behaviour.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
