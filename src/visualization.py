"""Plotly figure builders with one consistent dark scientific theme.

Colour system (validated with the dataviz palette checker on the chart surface
``#111a30``; lightness band L 0.48-0.67, chroma >= 0.10, contrast >= 3:1):

* condition identity: cyan ``#17a2c4`` (slot 1) and violet ``#8b7bea`` (slot 2);
  the cyan/violet pair has a CVD separation in the 6-8 dE band, so every chart
  also carries a **secondary encoding** (legend, marker symbol circle vs.
  diamond, direct axis labels, hover text);
* further categorical slots (clusters): amber, magenta, blue, aqua - never
  cycled; clusters beyond six are folded into one neutral "other" colour;
* magnitude (time / speed / angular velocity / occupancy): a single-hue cyan
  ramp, dark -> light;
* text always uses ink colours, never series colours.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .features import POS_COLS, QUAT_COLS, axis_vector, rotate_vectors

# ── tokens ──────────────────────────────────────────────────────────────────────────────────
PAGE_BG = "#0b1020"
SURFACE = "#111a30"
INK = "#e6eaf5"
INK2 = "#b4bdd6"
MUTED = "#8792b0"
GRID = "#1d2945"
AXIS = "#2c3b5e"
CYAN = "#17a2c4"
VIOLET = "#8b7bea"
CATEGORICAL = ["#17a2c4", "#8b7bea", "#c98500", "#d55181", "#3987e5", "#199e70"]
OTHER = "#9aa3bc"
NOISE = "#5d6a8a"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"

SEQ = [[0.0, "#0e7490"], [0.2, "#0891b2"], [0.4, "#06b6d4"], [0.6, "#22d3ee"], [0.8, "#67e8f9"], [1.0, "#cffafe"]]
SYMBOLS = {"Structured": "circle", "Free": "diamond"}
_FALLBACK_SYMBOLS = ["circle", "diamond", "square", "cross", "x"]


def condition_colors(conditions) -> dict[str, str]:
    """Stable condition -> colour map (fixed slots, independent of filtering)."""
    fixed = {"Structured": CYAN, "Free": VIOLET}
    out, nxt = {}, 0
    for c in sorted(set(map(str, conditions))):
        if c in fixed:
            out[c] = fixed[c]
        else:
            while CATEGORICAL[nxt % len(CATEGORICAL)] in fixed.values() and nxt < len(CATEGORICAL):
                nxt += 1
            out[c] = CATEGORICAL[min(nxt, len(CATEGORICAL) - 1)]
            nxt += 1
    return out


def condition_symbols(conditions) -> dict[str, str]:
    out = {}
    for i, c in enumerate(sorted(set(map(str, conditions)))):
        out[c] = SYMBOLS.get(c, _FALLBACK_SYMBOLS[i % len(_FALLBACK_SYMBOLS)])
    return out


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ── base styling ──────────────────────────────────────────────────────────────────────────────
def style(fig: go.Figure, height: int = 380, legend: bool = True, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK2, size=12),
        margin=dict(l=56, r=24, t=48 if (title or legend) else 24, b=48),
        hoverlabel=dict(bgcolor="#0b1020", bordercolor=AXIS, font=dict(family=FONT, color=INK, size=12)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=INK2, size=12), itemsizing="constant"),
        colorway=CATEGORICAL,
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.0, xanchor="left", font=dict(size=14, color=INK)))
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED), title_font=dict(color=INK2),
                     automargin=True)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED), title_font=dict(color=INK2),
                     automargin=True)
    return fig


def _ax(label: str, unit: str = "") -> str:
    return f"{label} ({unit})" if unit else label


def decimate_indices(n: int, max_points: int) -> np.ndarray:
    """Uniform index decimation that always keeps the first and last sample (visualisation only)."""
    if n <= max_points:
        return np.arange(n)
    idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    return idx


COLOR_LABELS = {"t": ("Time", "s"), "speed": ("Speed", "m/s"), "ang_vel_dps": ("Angular velocity", "deg/s")}


def _color_values(g: pd.DataFrame, color_by: str) -> np.ndarray:
    if color_by == "ang_vel_dps":
        return np.degrees(g["ang_vel"].to_numpy(float))
    return g[color_by].to_numpy(float)


def _robust_range(kin: pd.DataFrame, color_by: str) -> tuple[float, float]:
    v = _color_values(kin, color_by)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [1, 99]) if color_by != "t" else (v.min(), v.max())
    return (float(lo), float(hi)) if hi > lo else (float(lo), float(lo) + 1.0)


# ── 3-D trajectories ───────────────────────────────────────────────────────────────────────────
def trajectory_3d(kin: pd.DataFrame, color_by: str = "t", up_axis: str = "z", unit: str = "m",
                  max_points: int = 1500, exaggerate: bool = False, gaze_every_s: float | None = None,
                  forward_axis: str = "+x", arena: tuple | None = None, pois: np.ndarray | None = None,
                  height: int = 660, color_range: tuple[float, float] | None = None) -> tuple[go.Figure, dict]:
    """Interactive 3-D trajectory plot.

    ``kin`` must contain ``participant_id``, ``condition``, ``t``, the position
    columns, and for the corresponding colour modes ``speed`` / ``ang_vel``
    (and the quaternion columns for head-direction arrows).
    Returns the figure and a dict with display notes (decimation, exaggeration).
    """
    conds = condition_colors(kin["condition"].unique())
    syms = condition_symbols(kin["condition"].unique())
    fig = go.Figure()
    notes = {"decimated": False, "shown_points": 0, "total_points": 0, "exaggeration": 1.0}
    by_cond = color_by == "condition"
    cmin, cmax = color_range or (_robust_range(kin, color_by) if not by_cond else (0, 1))
    clabel, cunit = COLOR_LABELS.get(color_by, ("", ""))
    first_bar = True
    seen_cond: set[str] = set()
    for pid, g in kin.groupby("participant_id", sort=True):
        g = g.reset_index(drop=True)
        idx = decimate_indices(len(g), max_points)
        gd = g.iloc[idx]
        notes["total_points"] += len(g)
        notes["shown_points"] += len(gd)
        notes["decimated"] |= len(gd) < len(g)
        cond = str(g["condition"].iloc[0])
        hover = [f"<b>{pid}</b> · {cond}<br>t = {t:.1f} s<br>speed = {s:.2f} m/s" if np.isfinite(s) else
                 f"<b>{pid}</b> · {cond}<br>t = {t:.1f} s" for t, s in zip(gd["t"], gd["speed"])]
        if by_cond:
            line = dict(color=conds[cond], width=4)
            show_leg = cond not in seen_cond
            seen_cond.add(cond)
        else:
            vals = _color_values(gd, color_by)
            line = dict(color=vals, colorscale=SEQ, cmin=cmin, cmax=cmax, width=5,
                        colorbar=dict(title=dict(text=_ax(clabel, cunit), side="right"), thickness=12, len=0.55,
                                      x=1.0, tickfont=dict(color=MUTED), outlinewidth=0) if first_bar else None,
                        showscale=first_bar)
            first_bar = False
            show_leg = False
        fig.add_trace(go.Scatter3d(
            x=gd[POS_COLS[0]], y=gd[POS_COLS[1]], z=gd[POS_COLS[2]], mode="lines", line=line,
            name=cond if by_cond else str(pid), legendgroup=cond if by_cond else str(pid),
            showlegend=bool(by_cond and show_leg), text=hover, hoverinfo="text"))
        if gaze_every_s and all(c in g.columns for c in QUAT_COLS):
            step = max(1, int(round(gaze_every_s / max(float(g["t"].diff().median()), 1e-6))))
            gi = g.iloc[::step]
            p0 = gi[POS_COLS].to_numpy()
            dirs = rotate_vectors(gi[QUAT_COLS].to_numpy(), axis_vector(forward_axis))
            span = float(np.ptp(kin[POS_COLS[0]]) + np.ptp(kin[POS_COLS[1]])) / 2 or 1.0
            p1 = p0 + dirs * 0.04 * span
            xs, ys, zs = (np.column_stack([p0[:, k], p1[:, k], np.full(len(p0), np.nan)]).ravel() for k in range(3))
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=INK2, width=2),
                                       name="Head direction", legendgroup="gaze", showlegend=pid == kin["participant_id"].iloc[0],
                                       hoverinfo="skip", opacity=0.85))
    # start / end markers (one legend entry each)
    starts = kin.groupby("participant_id").head(1)
    ends = kin.groupby("participant_id").tail(1)
    fig.add_trace(go.Scatter3d(x=starts[POS_COLS[0]], y=starts[POS_COLS[1]], z=starts[POS_COLS[2]], mode="markers",
                               marker=dict(size=6, color=INK, symbol="circle", line=dict(color=SURFACE, width=2)),
                               name="Start", text=[f"Start · {p}" for p in starts["participant_id"]], hoverinfo="text"))
    fig.add_trace(go.Scatter3d(x=ends[POS_COLS[0]], y=ends[POS_COLS[1]], z=ends[POS_COLS[2]], mode="markers",
                               marker=dict(size=6, color=SURFACE, symbol="diamond", line=dict(color=INK, width=3)),
                               name="End", text=[f"End · {p}" for p in ends["participant_id"]], hoverinfo="text"))

    # scene geometry -----------------------------------------------------------------------------------
    up = {"x": 0, "y": 1, "z": 2}[up_axis]
    lo = kin[POS_COLS].min().to_numpy(float).copy()
    hi = kin[POS_COLS].max().to_numpy(float).copy()
    hor = [i for i in range(3) if i != up]
    if arena is not None and up_axis == "z":
        lo[0], hi[0], lo[1], hi[1] = arena[0], arena[1], arena[2], arena[3]
    rng = np.maximum(hi - lo, 1e-9)
    pad = max(0.1 * rng[up], 0.05)
    up_lo, up_hi = lo[up] - pad, hi[up] + pad
    true_floor = arena is not None and up_axis == "z" and not exaggerate
    if true_floor:                                         # true scale: floor at z = 0, head height above it
        up_lo, up_hi = min(0.0, lo[up] - 0.05), max(hi[up] + 0.35, 2.0)
    if arena is not None and up_axis == "z":
        u0, u1, v0, v1 = arena
        ring = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1], [u0, v0]])
        fig.add_trace(go.Scatter3d(x=ring[:, 0], y=ring[:, 1], z=np.full(5, up_lo), mode="lines",
                                   line=dict(color=AXIS, width=3), name="Arena floor plan" if true_floor else "Arena outline (on axis floor)",
                                   hoverinfo="skip"))
    if pois is not None and up_axis == "z":
        fig.add_trace(go.Scatter3d(x=pois[:, 0], y=pois[:, 1], z=np.full(len(pois), up_lo), mode="markers",
                                   marker=dict(size=2.5, color=MUTED), name="Navigation targets (simulation)",
                                   hoverinfo="skip", visible="legendonly"))
    names = ["x", "y", "z"]
    ranges = {names[i]: [lo[i] - 0.02 * rng[i], hi[i] + 0.02 * rng[i]] for i in hor}
    ranges[names[up]] = [up_lo, up_hi]
    axes = {}
    for i, n in enumerate(names):
        axes[f"{n}axis"] = dict(title=dict(text=_ax(f"{n}" + (" up" if i == up else ""), unit), font=dict(color=INK2)),
                                range=ranges[n], backgroundcolor=SURFACE, gridcolor=GRID, zerolinecolor=AXIS,
                                showbackground=True, linecolor=AXIS, tickfont=dict(color=MUTED))
    if exaggerate:
        hmax = float(max(rng[i] for i in hor))
        ratio_up = float(np.clip(rng[up] / hmax * 1.0, 0.0, 1.0))
        shown = 0.28
        notes["exaggeration"] = shown / max((up_hi - up_lo) / hmax, 1e-9)
        aspect = {names[i]: float(rng[i] / hmax) for i in hor}
        aspect[names[up]] = shown
        scene = dict(aspectmode="manual", aspectratio=aspect, **axes)
    else:
        scene = dict(aspectmode="data", **axes)
    cam_iso = dict(eye=dict(x=1.2, y=-1.3, z=0.85))
    cam_top = dict(eye=dict(x=0.001, y=0.001, z=2.1), up=dict(x=0, y=1, z=0))
    cam_side = dict(eye=dict(x=0.0, y=-2.1, z=0.25))
    if up_axis != "z":                                     # rotate the preset views to the dataset's up axis
        cam_iso = cam_top = cam_side = dict(eye=dict(x=1.6, y=1.6, z=1.6))
    fig.update_layout(
        height=height, paper_bgcolor=SURFACE, font=dict(family=FONT, color=INK2, size=12), scene=scene,
        scene_camera=cam_iso, margin=dict(l=0, r=0, t=36, b=0), uirevision="traj3d",
        legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="left", x=0.0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=INK2, size=12), itemsizing="constant"),
        hoverlabel=dict(bgcolor="#0b1020", bordercolor=AXIS, font=dict(family=FONT, color=INK)),
        updatemenus=[dict(type="buttons", direction="right", x=1.0, xanchor="right", y=1.0, yanchor="top", active=-1,
                          bgcolor="#16223f", bordercolor=AXIS, font=dict(color=INK2, size=11),
                          buttons=[dict(label="Isometric", method="relayout", args=[{"scene.camera": cam_iso}]),
                                   dict(label="Top", method="relayout", args=[{"scene.camera": cam_top}]),
                                   dict(label="Side", method="relayout", args=[{"scene.camera": cam_side}])])],
    )
    return fig, notes


# ── 2-D top-down ───────────────────────────────────────────────────────────────────────────────
def topdown_2d(kin: pd.DataFrame, u_col: str, v_col: str, color_by: str = "t", unit: str = "m",
               max_points: int = 3000, arena: tuple | None = None, pois: np.ndarray | None = None,
               height: int = 520, color_range: tuple[float, float] | None = None) -> go.Figure:
    conds = condition_colors(kin["condition"].unique())
    fig = go.Figure()
    by_cond = color_by == "condition"
    cmin, cmax = color_range or (_robust_range(kin, color_by) if not by_cond else (0, 1))
    clabel, cunit = COLOR_LABELS.get(color_by, ("", ""))
    first = True
    seen_cond: set[str] = set()
    for pid, g in kin.groupby("participant_id", sort=True):
        gd = g.iloc[decimate_indices(len(g), max_points)]
        cond = str(g["condition"].iloc[0])
        fig.add_trace(go.Scattergl(x=gd[u_col], y=gd[v_col], mode="lines", line=dict(color=rgba(conds[cond], 0.35), width=1),
                                   hoverinfo="skip", showlegend=False))
        if by_cond:
            fig.add_trace(go.Scattergl(x=gd[u_col], y=gd[v_col], mode="markers", marker=dict(size=3, color=conds[cond]),
                                       name=cond, legendgroup=cond, showlegend=cond not in seen_cond,
                                       text=[f"{pid} · t = {t:.1f} s" for t in gd["t"]], hoverinfo="text"))
            seen_cond.add(cond)
        else:
            v = _color_values(gd, color_by)
            fig.add_trace(go.Scattergl(x=gd[u_col], y=gd[v_col], mode="markers",
                                       marker=dict(size=3.5, color=v, colorscale=SEQ, cmin=cmin, cmax=cmax,
                                                   showscale=first, colorbar=dict(title=dict(text=_ax(clabel, cunit), side="right"),
                                                                                  thickness=12, len=0.7, outlinewidth=0,
                                                                                  tickfont=dict(color=MUTED))),
                                       text=[f"{pid} · t = {t:.1f} s" for t in gd["t"]], hoverinfo="text", showlegend=False))
        first = False
    s = kin.groupby("participant_id").head(1)
    e = kin.groupby("participant_id").tail(1)
    fig.add_trace(go.Scatter(x=s[u_col], y=s[v_col], mode="markers", name="Start",
                             marker=dict(size=10, color=INK, symbol="circle", line=dict(color=SURFACE, width=2)),
                             text=s["participant_id"], hoverinfo="text"))
    fig.add_trace(go.Scatter(x=e[u_col], y=e[v_col], mode="markers", name="End",
                             marker=dict(size=10, color=SURFACE, symbol="diamond", line=dict(color=INK, width=2.5)),
                             text=e["participant_id"], hoverinfo="text"))
    if arena is not None:
        u0, u1, v0, v1 = arena
        fig.add_shape(type="rect", x0=u0, x1=u1, y0=v0, y1=v1, line=dict(color=AXIS, width=1.5))
    if pois is not None:
        fig.add_trace(go.Scatter(x=pois[:, 0], y=pois[:, 1], mode="markers", name="Navigation targets (simulation)",
                                 marker=dict(size=5, color=MUTED, symbol="cross-thin", line=dict(color=MUTED, width=1)),
                                 hoverinfo="skip", visible="legendonly"))
    style(fig, height=height)
    fig.update_xaxes(title=_ax(u_col.replace("position_", ""), unit))
    fig.update_yaxes(title=_ax(v_col.replace("position_", ""), unit), scaleanchor="x", scaleratio=1)
    return fig


# ── time series ────────────────────────────────────────────────────────────────────────────────
def timeseries(kin: pd.DataFrame, y: str, ylabel: str, color: str, smooth_s: float = 0.5, height: int = 280,
               scale: float = 1.0, ref_lines: dict | None = None) -> go.Figure:
    """Raw (faint) + moving-average (solid) line of one per-sample quantity of one recording."""
    t = kin["t"].to_numpy()
    v = kin[y].to_numpy(float) * scale
    med_dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    w = max(1, int(round(smooth_s / med_dt)))
    sm = pd.Series(v).rolling(w, center=True, min_periods=max(1, w // 3)).mean().to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=v, mode="lines", name="raw", line=dict(color=rgba(color, 0.35), width=1),
                             hovertemplate="t = %{x:.2f} s<br>%{y:.3f}<extra>raw</extra>"))
    fig.add_trace(go.Scatter(x=t, y=sm, mode="lines", name=f"{smooth_s:g} s moving average", line=dict(color=color, width=2),
                             hovertemplate="t = %{x:.2f} s<br>%{y:.3f}<extra>mean</extra>"))
    bad = kin.loc[~kin["step_valid"] & (kin.index > kin.index[0])]
    for _, r in bad.iterrows():
        fig.add_vrect(x0=r["t"] - r["dt"], x1=r["t"], fillcolor=STATUS["warning"], opacity=0.25, line_width=0)
    style(fig, height=height)
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title=ylabel, rangemode="tozero")
    return fig


def cumulative_distance(kin: pd.DataFrame, unit: str = "m", height: int = 360) -> go.Figure:
    conds = condition_colors(kin["condition"].unique())
    fig = go.Figure()
    shown: set[str] = set()
    for pid, g in kin.groupby("participant_id", sort=True):
        cond = str(g["condition"].iloc[0])
        gd = g.iloc[decimate_indices(len(g), 600)]
        fig.add_trace(go.Scatter(x=gd["t"], y=gd["cum_dist"], mode="lines", line=dict(color=rgba(conds[cond], 0.35), width=1),
                                 name=f"{cond} (individual)", legendgroup=cond + "i", showlegend=cond not in shown,
                                 text=[f"{pid}"] * len(gd), hovertemplate="%{text}<br>t = %{x:.1f} s<br>%{y:.1f}<extra></extra>"))
        shown.add(cond)
    tmax = float(kin.groupby("participant_id")["t"].max().min())
    grid = np.linspace(0, tmax, 300)
    for cond, gc in kin.groupby("condition"):
        curves = [np.interp(grid, g["t"], g["cum_dist"]) for _, g in gc.groupby("participant_id")]
        fig.add_trace(go.Scatter(x=grid, y=np.mean(curves, axis=0), mode="lines", line=dict(color=conds[str(cond)], width=3),
                                 name=f"{cond} mean (n = {len(curves)})", hovertemplate="t = %{x:.1f} s<br>%{y:.1f}<extra>mean</extra>"))
    style(fig, height=height)
    fig.update_xaxes(title="Time since start (s)")
    fig.update_yaxes(title=_ax("Cumulative path length", unit))
    return fig


def histogram_by_condition(values: dict[str, np.ndarray], xlabel: str, nbins: int = 50, height: int = 340) -> go.Figure:
    conds = condition_colors(values.keys())
    fig = go.Figure()
    lo = min(np.nanmin(v) for v in values.values() if len(v))
    hi = max(np.nanpercentile(v, 99.5) for v in values.values() if len(v))
    for cond, v in values.items():
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        fig.add_trace(go.Histogram(x=v, histnorm="probability density", xbins=dict(start=lo, end=hi, size=(hi - lo) / nbins),
                                   name=f"{cond}", marker=dict(color=conds[cond], line=dict(width=0)), opacity=0.6))
    fig.update_layout(barmode="overlay")
    style(fig, height=height)
    fig.update_xaxes(title=xlabel, range=[lo, hi])
    fig.update_yaxes(title="Density")
    return fig


def participant_dotplot(table: pd.DataFrame, metric: str, ylabel: str, height: int = 340) -> go.Figure:
    """One marker per participant (ordered by condition), plus condition mean line."""
    conds = condition_colors(table["condition"].unique())
    syms = condition_symbols(table["condition"].unique())
    t = table.sort_values(["condition", "participant_id"])
    fig = go.Figure()
    for cond, g in t.groupby("condition", sort=True):
        fig.add_trace(go.Scatter(x=g["participant_id"], y=g[metric], mode="markers", name=str(cond),
                                 marker=dict(size=10, color=conds[str(cond)], symbol=syms[str(cond)],
                                             line=dict(color=SURFACE, width=2)),
                                 hovertemplate="%{x}<br>%{y:.3f}<extra>" + str(cond) + "</extra>"))
        fig.add_trace(go.Scatter(x=[g["participant_id"].iloc[0], g["participant_id"].iloc[-1]], y=[g[metric].mean()] * 2,
                                 mode="lines", line=dict(color=conds[str(cond)], width=2), showlegend=False,
                                 hovertemplate=f"{cond} mean = %{{y:.3f}}<extra></extra>"))
    style(fig, height=height)
    fig.update_xaxes(title="Participant (grouped by condition)", type="category", categoryorder="array",
                     categoryarray=t["participant_id"].tolist())
    fig.update_yaxes(title=ylabel)
    return fig


def metric_scatter(table: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, height: int = 360) -> go.Figure:
    conds = condition_colors(table["condition"].unique())
    syms = condition_symbols(table["condition"].unique())
    fig = go.Figure()
    for cond, g in table.groupby("condition", sort=True):
        fig.add_trace(go.Scatter(x=g[x], y=g[y], mode="markers", name=str(cond), text=g["participant_id"],
                                 marker=dict(size=11, color=conds[str(cond)], symbol=syms[str(cond)], line=dict(color=SURFACE, width=2)),
                                 hovertemplate="%{text}<br>" + xlabel + " = %{x:.3f}<br>" + ylabel + " = %{y:.3f}<extra></extra>"))
    style(fig, height=height)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title=ylabel)
    return fig


# ── entropy / occupancy ──────────────────────────────────────────────────────────────────────────────
def occupancy_heatmap(occ: np.ndarray, grid, probability: bool = False, unit: str = "m", title: str | None = None,
                      zmax: float | None = None, height: int = 460, cbar: bool = True, u_name: str = "horizontal axis 1",
                      v_name: str = "horizontal axis 2") -> go.Figure:
    """Occupancy map. Never-visited cells are left empty (plot background) so they are distinct from rarely visited ones."""
    z = occ / occ.sum() if probability else occ.copy()
    z = np.where(occ > 0, z, np.nan)
    xc = 0.5 * (grid.u_edges[:-1] + grid.u_edges[1:])
    yc = 0.5 * (grid.v_edges[:-1] + grid.v_edges[1:])
    ztxt = "p" if probability else "time (s)"
    fig = go.Figure(go.Heatmap(
        x=xc, y=yc, z=z, colorscale=SEQ, zmin=0, zmax=zmax, xgap=1, ygap=1, hoverongaps=False,
        colorbar=dict(title=dict(text="Occupancy probability" if probability else "Time in cell (s)", side="right"),
                      thickness=12, outlinewidth=0, tickfont=dict(color=MUTED)) if cbar else None, showscale=cbar,
        hovertemplate="cell centre (%{x:.2f}, %{y:.2f})<br>" + ztxt + " = %{z:.4g}<extra></extra>"))
    style(fig, height=height, legend=False, title=title)
    fig.update_xaxes(title=_ax(u_name, unit), range=[grid.u_edges[0], grid.u_edges[-1]], showgrid=False,
                     constrain="domain")
    fig.update_yaxes(title=_ax(v_name, unit), range=[grid.v_edges[0], grid.v_edges[-1]], showgrid=False,
                     scaleanchor="x", scaleratio=1, constrain="domain")
    return fig


def rank_occupancy(occ: np.ndarray, top: int = 40, height: int = 340) -> go.Figure:
    p = np.sort((occ / occ.sum()).ravel())[::-1]
    p = p[p > 0]
    shown = p[:top]
    fig = go.Figure(go.Bar(x=np.arange(1, len(shown) + 1), y=shown, marker=dict(color=CYAN, cornerradius=3),
                           hovertemplate="rank %{x}<br>p = %{y:.4f}<extra></extra>"))
    style(fig, height=height, legend=False)
    fig.update_layout(bargap=0.25)
    fig.update_xaxes(title=f"Cell rank by occupancy (top {len(shown)} of {len(p)} visited)")
    fig.update_yaxes(title="Occupancy probability p")
    return fig


def curves_by_condition(curves: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, height: int = 340,
                        log_x: bool = False) -> go.Figure:
    """Faint line per participant + bold condition mean. ``curves``: participant_id, condition, x, y."""
    conds = condition_colors(curves["condition"].unique())
    fig = go.Figure()
    shown: set[str] = set()
    for pid, g in curves.groupby("participant_id", sort=True):
        c = str(g["condition"].iloc[0])
        fig.add_trace(go.Scatter(x=g[x], y=g[y], mode="lines", line=dict(color=rgba(conds[c], 0.3), width=1),
                                 name=f"{c} (individual)", legendgroup=c + "i", showlegend=c not in shown, hovertemplate=f"{pid}<br>%{{x}}<br>%{{y:.3f}}<extra></extra>"))
        shown.add(c)
    mean = curves.groupby(["condition", x], as_index=False)[y].mean()
    for c, g in mean.groupby("condition"):
        fig.add_trace(go.Scatter(x=g[x], y=g[y], mode="lines+markers", name=f"{c} mean", line=dict(color=conds[str(c)], width=3),
                                 marker=dict(size=8, color=conds[str(c)], line=dict(color=SURFACE, width=2))))
    style(fig, height=height)
    fig.update_xaxes(title=xlabel, type="log" if log_x else "linear")
    fig.update_yaxes(title=ylabel)
    return fig


# ── clustering ──────────────────────────────────────────────────────────────────────────────────────────
def cluster_palette(sizes: dict[int, int]) -> dict[int, str]:
    """Largest clusters get the fixed categorical slots in order; the rest share one neutral colour."""
    pal: dict[int, str] = {}
    for rank, cid in enumerate(sorted(sizes, key=lambda c: -sizes[c])):
        pal[cid] = CATEGORICAL[rank] if rank < len(CATEGORICAL) else OTHER
    pal[-1] = NOISE
    return pal


def clusters_3d(points: pd.DataFrame, labels: np.ndarray, sizes: dict[int, int], coord_cols=POS_COLS, unit: str = "m",
                max_display: int = 15000, height: int = 600, seed: int = 0) -> tuple[go.Figure, dict]:
    pal = cluster_palette(sizes)
    n = len(points)
    idx = np.arange(n)
    if n > max_display:
        idx = np.sort(np.random.default_rng(seed).choice(n, max_display, replace=False))
    P = points.iloc[idx]
    L = np.asarray(labels)[idx]
    fig = go.Figure()
    ordered = sorted(sizes, key=lambda c: -sizes[c])
    folded = [c for c in ordered[len(CATEGORICAL):]]
    groups = [(c, f"Cluster {c} (n = {sizes[c]:,})", pal[c]) for c in ordered[:len(CATEGORICAL)]]
    for cid, name, col in groups:
        m = L == cid
        fig.add_trace(go.Scatter3d(x=P[coord_cols[0]][m], y=P[coord_cols[1]][m], z=P[coord_cols[2]][m], mode="markers",
                                   name=name, marker=dict(size=2.5, color=col, opacity=0.9), hoverinfo="name"))
    if folded:
        m = np.isin(L, folded)
        fig.add_trace(go.Scatter3d(x=P[coord_cols[0]][m], y=P[coord_cols[1]][m], z=P[coord_cols[2]][m], mode="markers",
                                   name=f"{len(folded)} smaller clusters", marker=dict(size=2.5, color=OTHER, opacity=0.9), hoverinfo="name"))
    m = L == -1
    fig.add_trace(go.Scatter3d(x=P[coord_cols[0]][m], y=P[coord_cols[1]][m], z=P[coord_cols[2]][m], mode="markers",
                               name=f"Noise (n = {int((np.asarray(labels) == -1).sum()):,})",
                               marker=dict(size=2, color=NOISE, opacity=0.35), hoverinfo="name"))
    axes = {f"{n_}axis": dict(title=dict(text=_ax(c.replace("position_", ""), unit), font=dict(color=INK2)), backgroundcolor=SURFACE,
                              gridcolor=GRID, linecolor=AXIS, showbackground=True, tickfont=dict(color=MUTED), zerolinecolor=AXIS)
            for n_, c in zip("xyz", coord_cols)}
    lo = np.array([float(points[c].min()) for c in coord_cols])
    hi = np.array([float(points[c].max()) for c in coord_cols])
    span = np.maximum(hi - lo, 1e-9)
    big = float(span.max())
    half = np.maximum(span, 0.2 * big) / 2                 # never let an axis collapse to a sliver; ticks stay true to the data
    mid = (lo + hi) / 2
    for k, n_ in enumerate("xyz"):
        axes[f"{n_}axis"]["range"] = [float(mid[k] - half[k] * 1.04), float(mid[k] + half[k] * 1.04)]
    aspect = dict(zip("xyz", (2 * half / (2 * half).max()).tolist()))
    fig.update_layout(height=height, paper_bgcolor=SURFACE, font=dict(family=FONT, color=INK2, size=12),
                      scene=dict(aspectmode="manual", aspectratio=aspect, **axes, camera=dict(eye=dict(x=1.2, y=-1.3, z=0.85))),
                      margin=dict(l=0, r=0, t=36, b=0), uirevision="clusters3d",
                      legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                                  font=dict(color=INK2, size=12), itemsizing="constant"))
    return fig, {"displayed": int(len(idx)), "total": int(n)}


def cluster_size_bars(sizes: dict[int, int], n_noise: int, height: int = 320) -> go.Figure:
    pal = cluster_palette(sizes)
    ordered = sorted(sizes, key=lambda c: -sizes[c])
    labels = [f"C{c}" for c in ordered] + (["noise"] if n_noise else [])
    vals = [sizes[c] for c in ordered] + ([n_noise] if n_noise else [])
    cols = [pal[c] for c in ordered] + ([NOISE] if n_noise else [])
    fig = go.Figure(go.Bar(x=labels, y=vals, marker=dict(color=cols, cornerradius=4), hovertemplate="%{x}<br>%{y:,} points<extra></extra>"))
    style(fig, height=height, legend=False)
    fig.update_layout(bargap=0.3)
    fig.update_xaxes(title="Cluster (largest first)", type="category")
    fig.update_yaxes(title="Points")
    return fig


def k_distance_fig(kd: np.ndarray, eps: float, k: int, unit: str = "", height: int = 320) -> go.Figure:
    idx = decimate_indices(len(kd), 1500)
    fig = go.Figure(go.Scatter(x=idx, y=kd[idx], mode="lines", line=dict(color=CYAN, width=2),
                               hovertemplate="point rank %{x}<br>distance %{y:.3f}<extra></extra>"))
    fig.add_hline(y=eps, line=dict(color=VIOLET, width=2), annotation_text=f"eps = {eps:g}", annotation_font_color=INK2,
                  annotation_position="top left")
    style(fig, height=height, legend=False)
    fig.update_xaxes(title="Points sorted by k-distance")
    fig.update_yaxes(title=_ax(f"Distance to {k}-th nearest neighbour", unit))
    return fig


def pca_scatter(emb: np.ndarray, table: pd.DataFrame, labels: np.ndarray, explained, height: int = 400) -> go.Figure:
    """Participant-level clusters: colour = cluster, symbol = experimental condition."""
    cats = {c: CATEGORICAL[i % len(CATEGORICAL)] for i, c in enumerate(sorted(set(labels[labels >= 0])))}
    syms = condition_symbols(table["condition"].unique())
    fig = go.Figure()
    for lab in sorted(cats):
        for cond in sorted(table["condition"].unique()):
            m = (labels == lab) & (table["condition"].to_numpy() == cond)
            if not m.any():
                continue
            fig.add_trace(go.Scatter(x=emb[m, 0], y=emb[m, 1], mode="markers", name=f"cluster {lab} · {cond}",
                                     text=table.loc[m, "participant_id"],
                                     marker=dict(size=12, color=cats[lab], symbol=syms[str(cond)], line=dict(color=SURFACE, width=2)),
                                     hovertemplate="%{text}<extra>cluster " + str(lab) + " · " + str(cond) + "</extra>"))
    style(fig, height=height)
    fig.update_xaxes(title=f"PC1 ({100 * explained[0]:.0f} % of variance)")
    fig.update_yaxes(title=f"PC2 ({100 * explained[1]:.0f} % of variance)")
    return fig


# ── statistics ──────────────────────────────────────────────────────────────────────────────────────────
def box_with_points(table: pd.DataFrame, metric: str, ylabel: str, groups: list[str], height: int = 360) -> go.Figure:
    conds = condition_colors(groups)
    syms = condition_symbols(groups)
    fig = go.Figure()
    for g in groups:
        sub = table[table["condition"] == g]
        v = sub[metric].to_numpy(float)
        fig.add_trace(go.Box(
            y=v, x=[g] * len(v), name=g, boxpoints="all", jitter=0.45, pointpos=0, whiskerwidth=0.6, boxmean=False,
            line=dict(color=conds[g], width=2), fillcolor=rgba(conds[g], 0.12),
            marker=dict(size=9, color=conds[g], symbol=syms[g], line=dict(color=SURFACE, width=1.5)),
            text=sub["participant_id"], hovertemplate="%{text}<br>%{y:.4g}<extra>" + g + "</extra>", showlegend=False))
        n = np.isfinite(v).sum()
        if n > 1:
            from scipy import stats as sps
            m, sd = np.nanmean(v), np.nanstd(v, ddof=1)
            h = sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
            fig.add_trace(go.Scatter(x=[g], y=[m], mode="markers", name="mean ± 95 % CI" if g == groups[0] else None,
                                     showlegend=g == groups[0], error_y=dict(type="constant", value=h, color=INK, thickness=2, width=8),
                                     marker=dict(size=11, color=INK, symbol="line-ew", line=dict(color=INK, width=3)),
                                     offsetgroup="m", hovertemplate=f"mean = {m:.4g}<br>95 % CI [{m - h:.4g}, {m + h:.4g}]<extra>{g}</extra>"))
    style(fig, height=height)
    fig.update_xaxes(type="category")
    fig.update_yaxes(title=ylabel)
    return fig


def forest_plot(cmp: pd.DataFrame, labels: dict[str, str], height: int | None = None) -> go.Figure:
    """Hedges' g with approximate 95 % CI per metric (participant-level)."""
    d = cmp.iloc[::-1]
    names = [labels.get(m, m) for m in d["metric"]]
    fig = go.Figure()
    fig.add_vline(x=0, line=dict(color=AXIS, width=2))
    fig.add_trace(go.Scatter(
        x=d["hedges_g"], y=names, mode="markers", name="Hedges' g (95 % CI, approx.)",
        error_x=dict(type="data", symmetric=False, array=(d["g_ci_high"] - d["hedges_g"]).to_numpy(),
                     arrayminus=(d["hedges_g"] - d["g_ci_low"]).to_numpy(), color=CYAN, thickness=2, width=0),
        marker=dict(size=11, color=CYAN, line=dict(color=SURFACE, width=2)),
        hovertemplate="%{y}<br>g = %{x:.2f}<extra></extra>"))
    style(fig, height=height or 80 + 38 * len(d), legend=False)
    fig.update_xaxes(title="Hedges' g (B − A)")
    fig.update_yaxes(title="")
    return fig


def qc_trajectory(flags: pd.DataFrame, u_col: str, v_col: str, height: int = 420) -> go.Figure:
    """Top-down view of one raw recording with flagged samples highlighted (status colours + symbols)."""
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=flags[u_col], y=flags[v_col], mode="lines+markers", name="samples",
                               line=dict(color=rgba(CYAN, 0.5), width=1), marker=dict(size=3, color=CYAN)))
    for col, name, color, sym in (("flag_outlier", "statistical outlier (Tukey)", STATUS["serious"], "diamond"),
                                  ("flag_discontinuity", "implausible jump (> max speed)", STATUS["warning"], "triangle-up"),
                                  ("flag_spike", "isolated spike", STATUS["critical"], "x")):
        m = flags[col].to_numpy(bool)
        if m.any():
            fig.add_trace(go.Scatter(x=flags.loc[m, u_col], y=flags.loc[m, v_col], mode="markers", name=f"{name} ({int(m.sum())})",
                                     marker=dict(size=11, color=color, symbol=sym, line=dict(color=SURFACE, width=1.5)),
                                     text=[f"t = {t:.2f} s" for t in flags.loc[m, "timestamp"]], hoverinfo="text"))
    style(fig, height=height)
    fig.update_xaxes(title=u_col)
    fig.update_yaxes(title=v_col, scaleanchor="x", scaleratio=1)
    return fig
