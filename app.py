"""6-DoF Explorer - Human Movement & Exploration Analytics (Streamlit dashboard).

All scientific computation lives in ``src/``; this file only handles layout,
caching and presentation. Run with:  streamlit run app.py
"""
from __future__ import annotations

import html
import inspect
import json
import platform
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from src import clustering as cl
from src import entropy as ent
from src import statistics as stt
from src import visualization as viz
from src.data_generator import generate_dataset
from src.data_loader import POSITION_UNITS, TIME_UNITS, ColumnMapping, Dataset, ImportSettings, guess_mapping, parse_table, read_table
from src.features import (POS_COLS, FeatureConfig, add_euler_columns, horizontal_axes, metric_catalogue, summarize_kinematics)
from src.pipeline import default_stat_metrics, merge_metrics, stage_exploration, stage_kinematics, stage_prepare
from src.preprocessing import PreprocessConfig, flag_samples

st.set_page_config(page_title="6-DoF Explorer", page_icon="🧭", layout="wide",
                   initial_sidebar_state="expanded")

PAGES = ["Research Overview", "3D Trajectory Explorer", "Movement Analytics", "Exploration Entropy",
         "Unsupervised Clustering", "Statistical Comparison", "Methodology & Reproducibility"]
SRC_SYN, SRC_UP = "Synthetic demo", "Uploaded CSV"

# ═════════════════════════════════════════════════════════════════════════════════════════════
# Styling
# ═════════════════════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
:root{--bg:#0b1020;--surface:#111a30;--surface2:#16223f;--line:rgba(148,163,184,.16);--ink:#e6eaf5;--ink2:#b4bdd6;
--muted:#8792b0;--cyan:#22d3ee;--violet:#a78bfa;}
.stApp{background:var(--bg);color:var(--ink);}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2.4rem;padding-bottom:4rem;padding-left:2.2rem;padding-right:2.2rem;max-width:1400px;}
section[data-testid="stSidebar"]{background:#0d1428;border-right:1px solid var(--line);}
section[data-testid="stSidebar"] .block-container{padding-top:1.4rem;}
h1,h2,h3,h4{letter-spacing:-.012em;font-weight:600;color:var(--ink) !important;}
h2{font-size:1.35rem;margin-top:1.6rem;} h3{font-size:1.08rem;color:var(--ink);}
p,li,label,.stMarkdown{color:var(--ink2);}
hr{border-color:var(--line);}
a{color:var(--cyan);}
.brand{display:flex;align-items:center;gap:.65rem;margin-bottom:.9rem;}
.brand-mark{width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#17a2c4,#8b7bea);display:flex;
align-items:center;justify-content:center;color:#0b1020;font-weight:700;font-size:.8rem;letter-spacing:.02em;}
.brand-name{font-weight:650;color:var(--ink);font-size:1.02rem;line-height:1.15;}
.brand-sub{color:var(--muted);font-size:.74rem;}
.side-label{color:var(--muted);font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;margin:1.1rem 0 .35rem;}
.eyebrow{color:var(--cyan);font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;font-weight:600;margin-bottom:.15rem;}
.page-title{font-size:2rem;font-weight:650;color:var(--ink);letter-spacing:-.02em;line-height:1.15;margin:0 0 .35rem;}
.page-sub{color:var(--ink2);max-width:62rem;font-size:1rem;margin-bottom:.9rem;}
.chip{display:inline-flex;align-items:center;gap:.4rem;padding:.2rem .65rem;border-radius:999px;border:1px solid var(--line);
background:var(--surface);color:var(--ink2);font-size:.76rem;margin:0 .4rem .4rem 0;white-space:nowrap;}
.chip b{color:var(--ink);font-weight:600;}
.chip.syn{border-color:rgba(167,139,250,.5);background:rgba(139,123,234,.10);color:#d4cbfb;}
.chip.up{border-color:rgba(34,211,238,.45);background:rgba(23,162,196,.10);color:#b9f2fb;}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));gap:.7rem;margin:.4rem 0 1rem;}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:.8rem .9rem .75rem 1rem;position:relative;overflow:hidden;}
.kpi:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:linear-gradient(180deg,#17a2c4,#8b7bea);}
.kpi-label{color:var(--muted);font-size:.68rem;letter-spacing:.06em;text-transform:uppercase;}
.kpi-value{color:var(--ink);font-size:1.5rem;font-weight:620;line-height:1.25;letter-spacing:-.01em;}
.kpi-sub{color:var(--muted);font-size:.76rem;}
.callout{border:1px solid var(--line);border-left:3px solid var(--cyan);background:var(--surface);border-radius:10px;
padding:.75rem 1rem;margin:.6rem 0 1rem;color:var(--ink2);font-size:.92rem;}
.callout b{color:var(--ink);}
.callout.warn{border-left-color:#fab219;background:rgba(250,178,25,.06);}
.callout.violet{border-left-color:var(--violet);}
.badge{display:inline-flex;align-items:center;gap:.45rem;padding:.3rem .7rem;border-radius:8px;border:1px solid var(--line);
background:var(--surface);font-size:.8rem;color:var(--ink2);margin:0 .45rem .45rem 0;}
.badge .ico{font-weight:700;font-size:.85rem;}
.badge.good .ico{color:#0ca30c;} .badge.warning .ico{color:#fab219;} .badge.critical .ico{color:#d03b3b;}
.badge b{color:var(--ink);}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:1rem 1.15rem;height:100%;}
.card h4{margin:0 0 .35rem;font-size:1rem;} .card p,.card li{font-size:.9rem;color:var(--ink2);}
.card .tag{font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);}
div[data-testid="stPlotlyChart"]{border:1px solid var(--line);border-radius:12px;overflow:hidden;background:var(--surface);}
div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:10px;overflow:hidden;}
div[data-testid="stExpander"]{border:1px solid var(--line);border-radius:10px;background:var(--surface);}
div[data-testid="stExpander"] summary{color:var(--ink2);}
button[data-baseweb="tab"]{color:var(--muted);} button[data-baseweb="tab"][aria-selected="true"]{color:var(--ink);}
.stRadio [role="radiogroup"]{gap:.15rem;}
section[data-testid="stSidebar"] .stRadio label{padding:.18rem 0;}
.small{color:var(--muted);font-size:.8rem;}
code{color:#b9f2fb;background:rgba(23,162,196,.10);padding:.05rem .3rem;border-radius:4px;}
</style>
"""

def _stretch_kw(fn) -> dict:
    """Full-width keyword that works on both old (use_container_width) and new (width='stretch') Streamlit."""
    p = inspect.signature(fn).parameters.get("width")
    return {"width": "stretch"} if p is not None and isinstance(p.default, str) else {"use_container_width": True}


_PLOT_W = _stretch_kw(st.plotly_chart)
_DF_W = _stretch_kw(st.dataframe)


def show(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, key=key, config={"displaylogo": False, "scrollZoom": True,
                                          "modeBarButtonsToRemove": ["lasso2d", "select2d"]}, **_PLOT_W)


def show_df(df: pd.DataFrame, height: int | None = None, **kw) -> None:
    df = df.copy()
    for c in df.columns[df.dtypes == object]:            # mixed int/str columns (e.g. cluster id vs 'noise') break Arrow
        if df[c].map(type).nunique() > 1:
            df[c] = df[c].astype(str)
    if height is not None:
        kw["height"] = int(height)
    st.dataframe(df, hide_index=True, **_DF_W, **kw)


def show_ops(ops: pd.DataFrame) -> None:
    """Static table so the long 'effect on analysis' sentences wrap instead of being truncated."""
    st.table(ops.set_index("operation"))


def md(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def callout(text: str, kind: str = "") -> None:
    md(f'<div class="callout {kind}">{text}</div>')


def kpis(items: list[tuple[str, str, str]]) -> None:
    cells = "".join(f'<div class="kpi"><div class="kpi-label">{html.escape(l)}</div><div class="kpi-value">{html.escape(str(v))}</div>'
                    f'<div class="kpi-sub">{html.escape(s)}</div></div>' for l, v, s in items)
    md(f'<div class="kpi-grid">{cells}</div>')


def badge(label: str, count, status: str) -> str:
    ico = {"good": "✓", "warning": "▲", "critical": "✕"}[status]
    return f'<span class="badge {status}"><span class="ico">{ico}</span><span>{html.escape(label)}</span><b>{count}</b></span>'


def header(eyebrow: str, title: str, sub: str, ctx: "Ctx | None") -> None:
    md(f'<div class="eyebrow">{html.escape(eyebrow)}</div><div class="page-title">{html.escape(title)}</div>'
       f'<div class="page-sub">{sub}</div>')
    if ctx is not None:
        chips = ('<span class="chip syn"><b>SYNTHETIC DATA</b> simulated · not real VR observations</span>' if ctx.ds.is_synthetic else
                 f'<span class="chip up"><b>UPLOADED</b> {html.escape(ctx.ds.name)}</span>')
        chips += (f'<span class="chip">{ctx.n_part} participants</span><span class="chip">{len(ctx.metrics)} analysed recordings</span>'
                  f'<span class="chip">{len(ctx.clean):,} samples</span>')
        md(f'<div>{chips}</div>')


def fmt_p(p: float) -> str:
    return "n/a" if not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def fmt(x: float, d: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}g}" if abs(x) < 1000 else f"{x:,.0f}"


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Cached computation stages
# ═════════════════════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Generating synthetic recordings …", max_entries=4)
def cached_synthetic(seed: int, n: int, artifacts: bool) -> Dataset:
    return generate_dataset(n_participants=n, seed=seed, inject_artifacts=artifacts)


@st.cache_data(show_spinner="Preprocessing …", max_entries=8)
def cached_prepare(ds_id: str, pre_json: str, window: float | None, _ds: Dataset):
    return stage_prepare(_ds, PreprocessConfig(**json.loads(pre_json)), window)


@st.cache_data(show_spinner="Extracting kinematic features …", max_entries=8)
def cached_kinematics(ds_id: str, pre_json: str, window: float | None, feat_json: str, has_rot: bool, _clean: pd.DataFrame):
    return stage_kinematics(_clean, has_rot, FeatureConfig(**json.loads(feat_json)))


@st.cache_data(show_spinner=False, max_entries=24)
def cached_exploration(ds_id: str, pre_json: str, window: float | None, feat_json: str, cell: float, _ds: Dataset, _clean: pd.DataFrame):
    return stage_exploration(_ds, _clean, FeatureConfig(**json.loads(feat_json)), cell)


@st.cache_data(show_spinner=False, max_entries=8)
def min_duration(ds_id: str, _ds: Dataset) -> float:
    d = _ds.df.groupby(["participant_id", "condition"])["timestamp"].agg(lambda s: s.max() - s.min())
    return float(d.min())


@dataclass
class Ctx:
    ds: Dataset
    ds_id: str
    pre_json: str
    feat_json: str
    window: float | None
    clean: pd.DataFrame
    qc: object
    kin: pd.DataFrame
    skipped: dict
    movement: pd.DataFrame
    grid: ent.SpatialGrid
    expl: pd.DataFrame
    metrics: pd.DataFrame
    cell: float
    fcfg: FeatureConfig

    @property
    def n_part(self) -> int:
        return int(self.ds.df["participant_id"].nunique())

    @property
    def unit(self) -> str:
        return self.ds.unit_label

    @property
    def labels(self) -> dict:
        return metric_catalogue(self.unit)

    def lbl(self, m: str, unit: bool = True) -> str:
        info = self.labels.get(m, {"label": m, "unit": ""})
        return f"{info['label']} ({info['unit']})" if unit and info["unit"] else info["label"]

    @property
    def conditions(self) -> list[str]:
        return sorted(self.clean["condition"].astype(str).unique())

    @property
    def recordings(self) -> list[tuple[str, str]]:
        return [(r.participant_id, r.condition) for r in self.metrics[["participant_id", "condition"]].itertuples()]


def recording(ctx: Ctx, pid: str) -> pd.DataFrame:
    return ctx.clean[ctx.clean["participant_id"] == pid].sort_values("timestamp", kind="stable")


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════════════════════
def sidebar_controls() -> dict:
    s = {}
    with st.sidebar:
        md('<div class="brand"><div class="brand-mark">6D</div><div><div class="brand-name">6-DoF Explorer</div>'
           '<div class="brand-sub">Human movement &amp; exploration analytics</div></div></div>')
        s["page"] = st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")

        md('<div class="side-label">Data</div>')
        s["source"] = st.radio("Data source", [SRC_SYN, SRC_UP], key="source", label_visibility="collapsed")
        with st.expander("Synthetic generator", expanded=False):
            s["seed"] = int(st.number_input("Random seed", 0, 1_000_000, 42, 1, key="seed"))
            s["n_part"] = int(st.slider("Participants", 4, 40, 20, 2, key="n_part"))
            s["artifacts"] = st.toggle("Inject tracking artifacts (QC demo)", value=False, key="artifacts",
                                       help="Adds spikes, gaps, duplicates, NaNs and non-unit quaternions to some "
                                            "recordings so the quality-control tools have something to find.")
        with st.expander("Preprocessing", expanded=False):
            c1, c2 = st.columns(2)
            drop_missing = c1.toggle("Drop missing", True, key="pp_miss")
            drop_dup = c2.toggle("Drop dup. time", True, key="pp_dup")
            renorm = st.toggle("Renormalise quaternions", True, key="pp_renorm")
            max_speed = st.number_input("Max plausible speed (units/s)", 0.5, 100.0, 6.0, 0.5, key="pp_vmax",
                                        help="Steps faster than this are flagged as position discontinuities.")
            iqr_k = st.number_input("Outlier fence (× IQR)", 1.0, 10.0, 3.0, 0.5, key="pp_iqr")
            repair = st.toggle("Repair isolated spikes", False, key="pp_repair",
                               help="Linear interpolation of single-sample excursions only. Off by default - flagged, not removed.")
            rs = st.selectbox("Resample", ["Off", "15 Hz", "30 Hz", "60 Hz"], key="pp_rs")
            sm = st.slider("Gaussian smoothing σ (s)", 0.0, 0.30, 0.0, 0.01, key="pp_sm")
            s["pp"] = dict(drop_missing=drop_missing, drop_duplicate_timestamps=drop_dup, renormalize_quaternions=renorm,
                           max_speed=float(max_speed), outlier_iqr_k=float(iqr_k), repair_spikes=bool(repair),
                           resample_hz=None if rs == "Off" else float(rs.split()[0]), smooth_sigma_s=float(sm))
        with st.expander("Features & analysis window", expanded=False):
            s["gap"] = float(st.number_input("Tracking-gap threshold (× median Δt)", 2.0, 100.0, 10.0, 1.0, key="ft_gap"))
            s["turn"] = float(st.slider("Direction-change threshold (deg)", 15, 120, 45, 5, key="ft_turn"))
            s["stride"] = float(st.slider("Direction stride (s)", 0.2, 2.0, 0.5, 0.1, key="ft_stride",
                                          help="Positions are sub-sampled at this interval before turning angles are computed."))
            s["use_window"] = st.toggle("Common analysis window", False, key="win_on",
                                        help="Analyse only the first T seconds of every recording so that entropy and "
                                             "coverage are compared at equal observation time.")
    return s


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Context assembly
# ═════════════════════════════════════════════════════════════════════════════════════════════
def get_dataset(s: dict) -> Dataset | None:
    if s["source"] == SRC_SYN:
        return cached_synthetic(s["seed"], s["n_part"], s["artifacts"])
    return st.session_state.get("uploaded_ds")


def build_ctx(s: dict) -> Ctx | None:
    ds = get_dataset(s)
    if ds is None:
        return None
    window = None
    if s["use_window"]:
        mx = max(5.0, float(np.floor(min_duration(ds.dataset_id, ds))))
        with st.sidebar:
            window = float(st.slider("Window length T (s)", 5.0, mx, mx, 1.0, key="win_T")) if mx > 5.0 else 5.0
    pre = PreprocessConfig(**s["pp"], gap_factor=s["gap"])
    feat = FeatureConfig(gap_factor=s["gap"], direction_stride_s=s["stride"], turn_threshold_deg=s["turn"])
    pre_json = json.dumps(pre.to_dict(), sort_keys=True)
    feat_json = json.dumps(feat.__dict__, sort_keys=True)
    clean, qc = cached_prepare(ds.dataset_id, pre_json, window, ds)
    kin, skipped, movement = cached_kinematics(ds.dataset_id, pre_json, window, feat_json, ds.has_rotation, clean)
    cell = float(st.session_state.get("_p_cell", 1.0))
    grid, expl = cached_exploration(ds.dataset_id, pre_json, window, feat_json, cell, ds, clean)
    metrics = merge_metrics(movement, expl) if len(movement) else movement
    return Ctx(ds, ds.dataset_id, pre_json, feat_json, window, clean, qc, kin, skipped, movement, grid, expl, metrics, cell, feat)


def focus_selector(ctx: Ctx) -> str:
    opts = [pid for pid, _ in ctx.recordings]
    cond = dict(ctx.recordings)
    with st.sidebar:
        md('<div class="side-label">Focus recording</div>')
        pid = st.selectbox("Focus recording", opts, key="focus", format_func=lambda p: f"{p} · {cond[p]}",
                           label_visibility="collapsed")
        md('<div class="small">Used by the single-recording views.</div>')
    return pid


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Page 1 - overview
# ═════════════════════════════════════════════════════════════════════════════════════════════
def condition_cards(ctx: Ctx) -> None:
    desc = {
        "Structured": ("Structured Exploration", ["Boustrophedon sweep through 25 navigation targets (second lap reversed)",
                                                  "Low heading noise, directed transitions between regions",
                                                  "Moderate head rotation, slower gaze shifts"]),
        "Free": ("Free Exploration", ["Random hops between targets with a strong tendency to return to known ones",
                                      "Irregular re-targeting and high heading variability",
                                      "Larger and faster head rotations"]),
    }
    conds = ctx.conditions
    cols = st.columns(max(len(conds), 1))
    colors = viz.condition_colors(conds)
    for col, c in zip(cols, conds):
        n = int((ctx.metrics["condition"] == c).sum()) if len(ctx.metrics) else 0
        title, bullets = desc.get(c, (c, []))
        if not ctx.ds.is_synthetic:
            bullets = []
        li = "".join(f"<li>{b}</li>" for b in bullets)
        with col:
            md(f'<div class="card" style="border-top:3px solid {colors[c]}"><div class="tag">Condition · n = {n}</div>'
               f'<h4>{html.escape(title)}</h4><ul style="padding-left:1.1rem;margin:.3rem 0 0">{li}</ul></div>')


def import_ui() -> None:
    st.caption("Upload a CSV with one row per tracking sample. Nothing is uploaded anywhere - the file stays on this computer.")
    f = st.file_uploader("CSV file", type=["csv", "txt", "tsv"], key="uploader")
    if f is None:
        return
    data = f.getvalue()
    try:
        raw = read_table(data)
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        return
    st.markdown(f"**{f.name}** · {len(raw):,} rows × {raw.shape[1]} columns")
    show_df(raw.head(6))
    cols = ["— none —"] + [str(c) for c in raw.columns]
    g = guess_mapping([str(c) for c in raw.columns])

    def sel(label, key, req=False):
        default = g.get(key)
        idx = cols.index(default) if default in cols else 0
        if req and idx == 0 and len(cols) > 1:
            idx = 1
        return st.selectbox(label, cols if not req else cols[1:], index=max(idx - (1 if req else 0), 0), key=f"map_{key}")

    st.markdown("##### Column mapping")
    c1, c2, c3 = st.columns(3)
    with c1:
        pid = sel("Participant ID", "participant")
        ts = sel("Timestamp", "timestamp", True)
        cond = sel("Condition (optional)", "condition")
    with c2:
        x, y, z = sel("Position X", "x", True), sel("Position Y", "y", True), sel("Position Z", "z", True)
    with c3:
        qx, qy, qz, qw = (sel("Quaternion x", "qx"), sel("Quaternion y", "qy"), sel("Quaternion z", "qz"), sel("Quaternion w", "qw"))
    st.markdown("##### Units and conventions - declare, do not assume")
    d1, d2, d3, d4 = st.columns(4)
    tu = d1.selectbox("Time unit", list(TIME_UNITS), key="imp_tu")
    pu = d2.selectbox("Position unit", list(POSITION_UNITS), key="imp_pu")
    up = d3.selectbox("Vertical (up) axis", ["z", "y", "x"], key="imp_up",
                      help="Defines the horizontal plane used for occupancy grids and top-down plots.")
    fw = d4.selectbox("Local forward axis", ["+x", "-x", "+y", "-y", "+z", "-z"], key="imp_fw",
                      help="Only used to draw head-direction arrows; never used in metrics.")
    none = lambda v: None if v == "— none —" else v
    st.caption("Quaternions are mapped per component, so the order in your file does not matter. Rotation angles do not depend on "
               "handedness or on body→world vs world→body convention. Leave all four quaternion fields empty if you have no orientation data.")

    def do_import():
        mp = ColumnMapping(participant=none(pid), timestamp=ts, x=x, y=y, z=z, qx=none(qx), qy=none(qy), qz=none(qz), qw=none(qw),
                           condition=none(cond))
        ds, rep = parse_table(raw, mp, ImportSettings(time_unit=tu, position_unit=pu, up_axis=up, forward_axis=fw), f.name)
        st.session_state["upload_report"] = rep
        if ds is not None:
            st.session_state["uploaded_ds"] = ds
            st.session_state["source"] = SRC_UP          # safe: callbacks run before widgets are instantiated

    st.button("Validate and load", type="primary", on_click=do_import, key="btn_import")
    rep = st.session_state.get("upload_report")
    if rep is not None:
        for e in rep.errors:
            st.error(e)
        for w in rep.warnings:
            st.warning(w)
        if rep.ok:
            st.success(f"Loaded {rep.info['rows']:,} rows · {rep.info['participants']} participants · "
                       f"{rep.info['recordings']} recordings · ≈{rep.info['median_rate_hz']:.1f} Hz. "
                       "The data source has been switched to 'Uploaded CSV'; synthetic data is kept separate.")


def page_overview(ctx: Ctx | None, s: dict) -> None:
    header("Research overview", "6-DoF Explorer",
           "Scientific analysis of human exploration behaviour from six-degrees-of-freedom tracking: three translational "
           "and three rotational degrees of freedom, sampled at tens of Hertz.", ctx)
    if ctx is None:
        callout("No uploaded dataset yet. Use <b>Import your own data</b> below, or switch the data source in the sidebar.", "warn")
        with st.expander("Import your own data (CSV)", expanded=True):
            import_ui()
        return
    rep_up = st.session_state.get("upload_report")
    if not ctx.ds.is_synthetic and rep_up is not None and rep_up.warnings:
        callout("<b>Import notes</b> (from the column mapping and validation step):<ul style='margin:.3rem 0 0;padding-left:1.1rem'>" +
                "".join(f"<li>{html.escape(w)}</li>" for w in rep_up.warnings[:8]) + "</ul>", "warn")
    if ctx.ds.is_synthetic:
        callout("<b>Synthetic demonstration data.</b> Every trajectory in this dashboard was generated by a simulation with fixed "
                "random seed. It is <b>not</b> an observation of real VR experiments, and group differences reflect the generator "
                "settings - not evidence about human behaviour.", "warn")
    if ctx.ds.is_synthetic:
        st.caption("To analyse your own recordings, open **Import your own data (CSV)** at the bottom of this page, or choose "
                   "*Uploaded CSV* under *Data* in the sidebar. Synthetic and uploaded data are never mixed.")
    md("## Motivation")
    st.markdown(
        "Head-mounted displays and motion-capture systems record **position** (x, y, z) and **orientation** (a unit quaternion) "
        "of the head continuously. From these streams, behavioural research asks *how* people explore an environment: how far and "
        "how fast they move, how much they turn their head, which regions they cover, how evenly they distribute their time and "
        "how often they return to places. This demo shows an end-to-end, reproducible workflow for such questions: validated "
        "preprocessing, mathematically careful feature extraction (quaternion-relative rotations, time-weighted statistics), "
        "unsupervised learning, and participant-level inference.")
    md("## Dataset summary")
    t = ctx.qc.totals
    durs = ctx.metrics["duration_s"] if len(ctx.metrics) else pd.Series(dtype=float)
    kpis([("Participants", str(ctx.n_part), ", ".join(f"{(ctx.metrics['condition'] == c).sum()} {c}" for c in ctx.conditions)),
          ("Recordings", str(t["n_recordings"]), f"{len(ctx.metrics)} analysed"),
          ("Total samples", f"{t['n_rows']:,}", f"{len(ctx.clean):,} after preprocessing"),
          ("Sampling rate", f"{t.get('median_rate_hz', float('nan')):.1f} Hz", f"max Δt irregularity CV {fmt(t.get('max_dt_cv', np.nan), 2)}"),
          ("Recording length", f"{durs.mean():.0f} s" if len(durs) else "n/a", f"range {durs.min():.0f}–{durs.max():.0f} s" if len(durs) else ""),
          ("Orientation data", "yes" if ctx.ds.has_rotation else "no", "rotation metrics " + ("enabled" if ctx.ds.has_rotation else "disabled"))])
    if ctx.skipped:
        callout("<b>Recordings excluded from analysis</b> (not silently dropped): " +
                "; ".join(f"{html.escape(str(k))}: {html.escape(v)}" for k, v in ctx.skipped.items()), "warn")

    md("## Data quality")
    badges = (badge("Missing / non-finite rows", t["n_missing_rows"], "good" if not t["n_missing_rows"] else "warning") +
              badge("Duplicate timestamps", t["n_duplicate_timestamps"], "good" if not t["n_duplicate_timestamps"] else "warning") +
              badge("Out-of-order timestamps", t["n_nonmonotonic"], "good" if not t["n_nonmonotonic"] else "warning") +
              badge("Tracking gaps", t["n_gaps"], "good" if not t["n_gaps"] else "warning") +
              badge("Position jumps flagged", t["n_discontinuities"], "good" if not t["n_discontinuities"] else "warning") +
              badge("Statistical outliers flagged", t["n_outliers"], "good" if not t["n_outliers"] else "warning") +
              (badge("Quaternion norm off", t["n_quat_norm_bad"], "good" if not t["n_quat_norm_bad"] else "warning") if ctx.ds.has_rotation else ""))
    md(badges)
    for w in ctx.qc.warnings:
        st.caption("▲ " + w)
    with st.expander("Per-recording quality table (raw data, before preprocessing)"):
        per = ctx.qc.per_recording.copy()
        show_df(per.round(4))
    with st.expander("Preprocessing operations and their effect on the analysis", expanded=False):
        show_ops(ctx.qc.operations)
        st.caption("Nothing is removed silently: every operation reports how many rows it touched. Flagged samples are kept unless "
                   "spike repair is enabled in the sidebar.")
    with st.expander("Inspect flagged samples of one raw recording"):
        pid = st.selectbox("Recording", [p for p, _ in ctx.recordings], key="qc_pid")
        raw = ctx.ds.df[ctx.ds.df["participant_id"] == pid]
        pre = PreprocessConfig(**json.loads(ctx.pre_json))
        flags = flag_samples(raw, pre)
        u, v = horizontal_axes(ctx.ds.up_axis)
        show(viz.qc_trajectory(flags, u, v), key="qc_traj")
        if ctx.ds.is_synthetic and not ctx.ds.meta["injected"].empty:
            st.markdown("**Ground truth of injected artifacts vs. detection** (synthetic data only)")
            inj = ctx.ds.meta["injected"].merge(ctx.qc.per_recording[["participant_id", "n_spikes", "n_missing_rows",
                                                "n_duplicate_timestamps", "n_quat_norm_bad", "n_gaps"]], on="participant_id")
            show_df(inj.rename(columns={"spikes": "injected spikes", "n_spikes": "detected spikes", "missing_rows": "injected NaN rows",
                                        "n_missing_rows": "detected NaN rows", "duplicate_rows": "injected duplicates",
                                        "n_duplicate_timestamps": "detected duplicates", "quat_non_unit_rows": "injected non-unit q",
                                        "n_quat_norm_bad": "detected non-unit q", "gap_rows_removed": "injected gap rows",
                                        "n_gaps": "detected gaps"}).round(3))

    md("## Experimental conditions")
    condition_cards(ctx)
    md("## Research questions")
    st.markdown("1. Do conditions differ in **translational** behaviour (path length, speed, acceleration)?\n"
                "2. Do they differ in **rotational** behaviour (angular velocity, cumulative head rotation)?\n"
                "3. How is time distributed over space - **coverage, entropy, revisitation**?\n"
                "4. Where do people dwell (point-level density clusters), and do **participants** fall into behavioural types?")
    with st.expander("Import your own data (CSV)", expanded="upload_report" in st.session_state):
        import_ui()


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Page 2 - 3D trajectories
# ═════════════════════════════════════════════════════════════════════════════════════════════
def page_trajectory(ctx: Ctx, focus: str) -> None:
    header("Visual centrepiece", "Interactive 3D Trajectory Explorer",
           "Position and head orientation of each recording in space. Drag to rotate, scroll to zoom, double-click to reset.", ctx)
    rec_cond = dict(ctx.recordings)
    c1, c2, c3 = st.columns([1.1, 1.8, 1.1])
    conds = c1.multiselect("Condition", ctx.conditions, default=ctx.conditions, key="t_cond")
    options = [p for p, c in ctx.recordings if c in conds]
    default = [focus] if focus in options else options[:1]
    parts = c2.multiselect("Participants", options, default=default, key="t_parts")
    modes = {"Time": "t", "Speed": "speed", "Angular velocity": "ang_vel_dps", "Condition": "condition"}
    avail = [k for k, v in modes.items() if not (v == "ang_vel_dps" and not ctx.ds.has_rotation)]
    color_label = c3.selectbox("Colour by", avail, index=avail.index("Condition") if len(parts) > 1 else 0, key="t_color")
    if not parts:
        st.info("Select at least one participant.")
        return
    kin = ctx.kin[ctx.kin["participant_id"].isin(parts)]
    tmax = float(np.ceil(kin["t"].max()))
    r1, r2 = st.columns([1, 2.2])
    with r1:
        with st.expander("Display options", expanded=False):
            max_pts = st.slider("Max points per trajectory (display only)", 300, 6000, 1500, 100, key="t_pts")
            exag = st.toggle("Exaggerate vertical axis", False, key="t_exag",
                             help="Off = true scale (head height above the floor). On = zoom into the centimetre-scale height variation; "
                                  "the walking bob then looks much larger than it is.")
            gaze = st.toggle("Show head direction", ctx.ds.has_rotation, key="t_gaze", disabled=not ctx.ds.has_rotation)
            gaze_dt = st.slider("Arrow interval (s)", 0.5, 5.0, 1.5, 0.5, key="t_gaze_dt", disabled=not ctx.ds.has_rotation)
    with r2:
        t0, t1 = st.slider("Time range (s)", 0.0, tmax, (0.0, tmax), 1.0, key="t_range") if tmax > 1 else (0.0, tmax)
    color_by = modes[color_label]
    full_range = viz._robust_range(kin, color_by) if color_by != "condition" else None
    kw = kin[(kin["t"] >= t0) & (kin["t"] <= t1)]
    if kw.empty:
        st.warning("No samples in the selected time range.")
        return
    pois = ctx.ds.meta.get("pois") if ctx.ds.is_synthetic else None
    fig, notes = viz.trajectory_3d(kw, color_by, ctx.ds.up_axis, ctx.unit, max_pts, exag, gaze_dt if gaze else None,
                                   ctx.ds.forward_axis, ctx.ds.arena_bounds, pois, height=600, color_range=full_range)
    show(fig, key="traj3d")
    notes_txt = []
    if notes["decimated"]:
        notes_txt.append(f"Display decimation: {notes['shown_points']:,} of {notes['total_points']:,} samples drawn. "
                         "All metrics are computed on the full-rate data.")
    if exag and notes["exaggeration"] > 1.5:
        notes_txt.append(f"Vertical axis exaggerated ×{notes['exaggeration']:.0f} for visibility.")
    if notes_txt:
        st.caption(" · ".join(notes_txt))

    md("### Top-down projection and window metrics")
    left, right = st.columns([1.15, 1])
    u, v = horizontal_axes(ctx.ds.up_axis)
    with left:
        show(viz.topdown_2d(kw, u, v, color_by, ctx.unit, 3000, ctx.ds.arena_bounds, pois, color_range=full_range),
             key="topdown")
    with right:
        callout("<b>Reading the plots.</b> Colour encodes the chosen quantity along the path (single-hue ramp: darker = lower). "
                "Circles mark starts, open diamonds ends, grey arrows the head's forward direction derived from the quaternion. "
                "The 3D view is drawn at true scale by default: head height varies by only a few centimetres over a floor "
                "that spans metres, so the path looks almost planar. Use <i>Display options</i> to exaggerate the vertical axis.")
        callout("<b>Metrics below</b> are computed from the full-rate samples inside the selected time range, not from the decimated drawing. "
                "Maximum values are sensitive to tracking glitches (see the data-quality section on the overview page).", "violet")
    rows = []
    for pid, g in kw.groupby("participant_id"):
        sm = summarize_kinematics(g.reset_index(drop=True), ctx.fcfg)
        row = {"participant": pid, "condition": rec_cond.get(pid, ""), "window (s)": round(float(g["t"].max() - g["t"].min()), 1),
               f"path ({ctx.unit})": sm["path_length"], f"mean speed ({ctx.unit}/s)": sm["mean_speed"], f"max speed ({ctx.unit}/s)": sm["max_speed"]}
        if ctx.ds.has_rotation:
            row["mean ang. vel. (deg/s)"] = sm["mean_ang_vel_dps"]
            row["cum. rotation (deg)"] = sm["cum_rotation_deg"]
        rows.append(row)
    st.markdown("**Metrics within the selected time window**")
    show_df(pd.DataFrame(rows).round(3), height=min(80 + 35 * len(rows), 360))


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Page 3 - movement analytics
# ═════════════════════════════════════════════════════════════════════════════════════════════
def page_movement(ctx: Ctx, focus: str) -> None:
    header("Kinematics", "Movement Analytics",
           "Translational and rotational movement features computed from actual timestamp intervals; rotations from relative quaternions.", ctx)
    cond = dict(ctx.recordings)[focus]
    colors = viz.condition_colors(ctx.conditions)
    g = ctx.kin[ctx.kin["participant_id"] == focus].reset_index(drop=True)
    sm = ctx.metrics[ctx.metrics["participant_id"] == focus].iloc[0]
    items = [("Path length", f"{sm['path_length']:.1f} {ctx.unit}", f"net displacement {sm['net_displacement']:.1f} {ctx.unit}"),
             ("Mean speed", f"{sm['mean_speed']:.2f} {ctx.unit}/s", f"max {sm['max_speed']:.2f} · CV {sm['speed_cv']:.2f}"),
             ("Mean acceleration", f"{sm['mean_accel']:.2f} {ctx.unit}/s²", "|Δv| / Δt, raw data")]
    if ctx.ds.has_rotation:
        items += [("Mean angular velocity", f"{sm['mean_ang_vel_dps']:.0f} °/s", f"max {sm['max_ang_vel_dps']:.0f} °/s"),
                  ("Cumulative rotation", f"{sm['cum_rotation_deg']:,.0f}°", f"{sm['cum_rotation_deg'] / 360:.1f} full turns")]
    items.append(("Direction changes", f"{sm['turn_rate_per_min']:.1f} /min", f"mean turn {sm['mean_turn_angle_deg']:.0f}°"))
    st.markdown(f"**Inspecting recording {focus}** · {cond}  (change it with *Focus recording* in the sidebar)")
    kpis(items)
    tabs = st.tabs(["Time series", "Distributions", "Cumulative distance", "Per-participant summary", "Condition comparison"])
    with tabs[0]:
        smooth = st.slider("Moving-average window (s)", 0.1, 3.0, 0.5, 0.1, key="mv_smooth")
        st.markdown("**Speed over time**")
        show(viz.timeseries(g, "speed", f"Speed ({ctx.unit}/s)", colors[cond], smooth), key="ts_speed")
        if ctx.ds.has_rotation:
            st.markdown("**Angular velocity over time** (magnitude of the relative rotation between consecutive samples)")
            g2 = g.assign(ang_vel_dps=np.degrees(g["ang_vel"]))
            show(viz.timeseries(g2, "ang_vel_dps", "Angular velocity (deg/s)", colors[cond], smooth), key="ts_av")
        else:
            callout("No quaternion columns were mapped, so angular velocity and all rotation-based metrics are disabled.", "warn")
        st.caption("Amber bands mark tracking gaps (excluded from statistics). Raw data are faint; the solid line is a centred moving average.")
        if ctx.ds.has_rotation:
            with st.expander("Head orientation as Euler angles (display only)"):
                e = add_euler_columns(g)
                e = e.assign(yaw_unwrapped=np.degrees(np.unwrap(np.radians(e["euler_yaw_deg"]))))
                import plotly.graph_objects as go
                fig = go.Figure()
                for col, name, c in (("yaw_unwrapped", "yaw (unwrapped)", viz.CYAN), ("euler_pitch_deg", "pitch", viz.VIOLET),
                                     ("euler_roll_deg", "roll", "#c98500")):
                    fig.add_trace(go.Scatter(x=e["t"], y=e[col], mode="lines", name=name, line=dict(color=c, width=2)))
                viz.style(fig, 320)
                fig.update_xaxes(title="Time (s)")
                fig.update_yaxes(title="Angle (deg)")
                show(fig, key="euler")
                st.caption("Euler angles (ZYX convention assumed) suffer from wrapping and gimbal lock; they are shown for orientation "
                           "intuition only. All rotation metrics use quaternion-relative rotations.")
    with tabs[1]:
        callout("Histograms pool all valid samples of every recording in a condition. They describe the *shape* of movement; "
                "because samples from one participant are strongly autocorrelated they must not be used for significance tests.")
        c1, c2 = st.columns(2)
        sp = {str(c): x["speed"].dropna().to_numpy() for c, x in ctx.kin.groupby("condition")}
        with c1:
            st.markdown("**Speed distribution**")
            show(viz.histogram_by_condition(sp, f"Speed ({ctx.unit}/s)"), key="h_speed")
        with c2:
            if ctx.ds.has_rotation:
                st.markdown("**Angular velocity distribution**")
                av = {str(c): np.degrees(x["ang_vel"].dropna().to_numpy()) for c, x in ctx.kin.groupby("condition")}
                show(viz.histogram_by_condition(av, "Angular velocity (deg/s)"), key="h_av")
    with tabs[2]:
        show(viz.cumulative_distance(ctx.kin, ctx.unit), key="cumdist")
        st.caption("Thin lines: individual recordings. Thick lines: condition mean up to the shortest recording.")
    with tabs[3]:
        cols = ["participant_id", "condition", "duration_s", "path_length", "mean_speed", "max_speed", "speed_cv", "mean_accel",
                "net_displacement", "straightness", "turn_rate_per_min", "mean_turn_angle_deg"]
        if ctx.ds.has_rotation:
            cols[10:10] = ["cum_rotation_deg", "mean_ang_vel_dps", "max_ang_vel_dps"]
        tab = ctx.metrics[[c for c in cols if c in ctx.metrics.columns]]
        show_df(tab.round(3), height=420)
        st.download_button("Download participant metrics (CSV)", tab.to_csv(index=False).encode(), "participant_metrics.csv", "text/csv")
    with tabs[4]:
        opts = [m for m in ["mean_speed", "path_length", "speed_cv", "mean_accel", "straightness", "turn_rate_per_min",
                            "mean_ang_vel_dps", "cum_rotation_deg", "ang_vel_p95_dps"] if m in ctx.metrics.columns]
        m = st.selectbox("Metric", opts, format_func=ctx.lbl, key="mv_metric")
        show(viz.participant_dotplot(ctx.metrics, m, ctx.lbl(m)), key="dot_mv")
        st.caption("One marker per participant (the statistical unit); the line is the condition mean. Formal tests: *Statistical Comparison*.")


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Page 4 - entropy
# ═════════════════════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, max_entries=8)
def resolution_curves(ds_id: str, pre_json: str, window, gap: float, bounds: tuple, up: str, _clean: pd.DataFrame) -> pd.DataFrame:
    u, v = horizontal_axes(up)
    sizes = [5.0, 2.5, 1.25, 1.0, 0.75, 0.5, 0.25]
    rows = []
    for (pid, cond), g in _clean.groupby(["participant_id", "condition"]):
        g = g.sort_values("timestamp")
        t = g["timestamp"].to_numpy()
        if len(t) < 3:
            continue
        cur = ent.entropy_vs_resolution(g[u], g[v], t, bounds, sizes, gap * float(np.median(np.diff(t))))
        cur.insert(0, "condition", cond)
        cur.insert(0, "participant_id", pid)
        rows.append(cur)
    return pd.concat(rows, ignore_index=True)


@st.cache_data(show_spinner=False, max_entries=8)
def duration_curves(ds_id: str, pre_json: str, window, gap: float, cell: float, bounds: tuple, up: str, _clean: pd.DataFrame) -> pd.DataFrame:
    u, v = horizontal_axes(up)
    grid = ent.SpatialGrid.from_bounds(*bounds, cell_size=cell)
    rows = []
    for (pid, cond), g in _clean.groupby(["participant_id", "condition"]):
        g = g.sort_values("timestamp")
        t = g["timestamp"].to_numpy()
        if len(t) < 3:
            continue
        T = [x for x in (5, 10, 15, 20, 30, 40, 50, 60, 75, 90, 105, 120) if x <= t[-1] - t[0] + 1e-9]
        cur = ent.entropy_vs_duration(g[u], g[v], t, grid, T, gap * float(np.median(np.diff(t))))
        cur.insert(0, "condition", cond)
        cur.insert(0, "participant_id", pid)
        rows.append(cur)
    return pd.concat(rows, ignore_index=True)


def grid_bounds(ctx: Ctx) -> tuple:
    return ctx.ds.arena_bounds or ent.bounds_from_data(ctx.clean, *horizontal_axes(ctx.ds.up_axis))


def page_entropy(ctx: Ctx, focus: str) -> None:
    header("Exploration metrics", "Exploration Entropy",
           "How evenly is observation time spread over space? Occupancy, Shannon entropy, coverage and revisitation on a user-chosen grid.", ctx)
    u, v = horizontal_axes(ctx.ds.up_axis)
    cs = st.slider(f"Spatial grid cell size ({ctx.unit})", 0.25, 5.0, float(ctx.cell), 0.25, key="w_cell")
    st.session_state["_p_cell"] = cs
    if abs(cs - ctx.cell) > 1e-9:
        st.rerun()                                      # recompute spatial metrics for the new resolution
    grid = ctx.grid
    cond = dict(ctx.recordings)[focus]
    rec = recording(ctx, focus)
    t = rec["timestamp"].to_numpy()
    max_dt = ctx.fcfg.gap_factor * float(np.median(np.diff(t)))
    occ = ent.occupancy_time(rec[u], rec[v], t, grid, max_dt)
    H = ent.shannon_entropy(occ)
    row = ctx.metrics[ctx.metrics["participant_id"] == focus].iloc[0]
    kpis([("Grid", f"{grid.nx} × {grid.ny}", f"K = {grid.n_cells} cells of {cs:g} {ctx.unit}"),
          ("Shannon entropy H", f"{H:.2f} bits", f"upper bound log₂K = {ent.max_entropy(grid.n_cells):.2f}"),
          ("Normalised entropy", fmt(row["entropy_norm"], 3), "H / log₂K (see definition)"),
          ("Effective cells 2^H", f"{2 ** H:.1f}", "perplexity of the occupancy"),
          ("Spatial coverage", f"{100 * row['coverage']:.0f} %", f"{int(row['n_unique_cells'])} of {grid.n_cells} cells occupied"),
          ("Revisit ratio", f"{row['revisit_ratio']:.2f}", f"{row['revisits_per_min']:.1f} revisits / min")])
    callout(f"<b>Discretisation.</b> Positions are projected on the horizontal plane ({u.replace('position_', '')}, {v.replace('position_', '')}) "
            f"and binned on a square grid of {cs:g} {ctx.unit} cells spanning a bounding box <i>shared by all recordings</i>. "
            "Occupancy is the fraction of <i>observed time</i> per cell (trapezoidal sample durations, tracking gaps excluded). "
            "Entropy grows with finer grids and longer observation, so values are only comparable at identical grid size and "
            "window; normalisation by log₂K is valid, but 1.0 is unattainable if some cells cannot be reached.")
    tabs = st.tabs(["Occupancy of one recording", "Condition maps", "Comparison across participants", "Sensitivity", "Coverage vs entropy"])
    with tabs[0]:
        prob = st.toggle("Show probability instead of seconds", False, key="occ_prob")
        c1, c2 = st.columns([1.25, 1])
        with c1:
            st.markdown(f"**Occupancy heatmap · {focus} ({cond})**")
            show(viz.occupancy_heatmap(occ, grid, prob, ctx.unit, u_name=u.replace("position_", ""), v_name=v.replace("position_", "")), key="occ_map")
            st.caption("Empty (background) cells were never visited; filled cells use a single-hue ramp (darker = less time).")
        with c2:
            st.markdown("**Occupancy probability distribution**")
            show(viz.rank_occupancy(occ), key="occ_rank")
            with st.expander("Show the entropy calculation"):
                p = (occ / occ.sum()).ravel()
                idx = np.argsort(p)[::-1][:12]
                tab = pd.DataFrame({"cell (ix, iy)": [f"({i % grid.nx}, {i // grid.nx})" for i in idx], "p": p[idx],
                                    "−p·log₂p": -p[idx] * np.log2(np.where(p[idx] > 0, p[idx], 1))})
                show_df(tab.round(4))
                st.latex(r"H=-\sum_k p_k\log_2 p_k=" + f"{H:.3f}" + r"\ \text{bits},\qquad H_{\max}=\log_2 K=" + f"{ent.max_entropy(grid.n_cells):.3f}")
                st.caption(f"Top 12 of {int((p > 0).sum())} visited cells shown; the remaining cells contribute "
                           f"{H + float((p[idx] * np.log2(np.where(p[idx] > 0, p[idx], 1))).sum()):.3f} bits.")
    with tabs[1]:
        st.markdown("**Mean occupancy probability per condition** (average over participants, shared colour scale)")
        maps, zmax = {}, 0.0
        for c in ctx.conditions:
            acc = []
            for pid, cc in ctx.recordings:
                if cc != c:
                    continue
                r = recording(ctx, pid)
                tt = r["timestamp"].to_numpy()
                o = ent.occupancy_time(r[u], r[v], tt, grid, ctx.fcfg.gap_factor * float(np.median(np.diff(tt))))
                acc.append(o / o.sum())
            if acc:
                maps[c] = np.mean(acc, axis=0)
                zmax = max(zmax, maps[c].max())
        cols = st.columns(max(len(maps), 1))
        for col, (c, m) in zip(cols, maps.items()):
            with col:
                st.markdown(f"**{c}**")
                show(viz.occupancy_heatmap(m, grid, True, ctx.unit, zmax=zmax, height=400, cbar=c == list(maps)[-1],
                                           u_name=u.replace("position_", ""), v_name=v.replace("position_", "")), key=f"cmap_{c}")
    with tabs[2]:
        st.markdown("**Normalised entropy and coverage per participant** (the unit of analysis)")
        c1, c2 = st.columns(2)
        with c1:
            show(viz.participant_dotplot(ctx.metrics, "entropy_norm", "Normalised spatial entropy"), key="dot_ent")
        with c2:
            show(viz.participant_dotplot(ctx.metrics, "coverage", "Spatial coverage (fraction of cells)"), key="dot_cov")
        tab = ctx.metrics[["participant_id", "condition", "duration_s", "entropy_bits", "entropy_norm", "perplexity", "coverage",
                           "n_unique_cells", "revisit_ratio", "revisits_per_min"]].rename(columns={"n_unique_cells": "occupied_cells"})
        show_df(tab.round(3), height=360)
        st.caption("Longer recordings can reach more cells; compare with the duration column or enable the common analysis window in the sidebar.")
    with tabs[3]:
        b = grid_bounds(ctx)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Entropy depends on grid resolution**")
            rc = resolution_curves(ctx.ds_id, ctx.pre_json, ctx.window, ctx.fcfg.gap_factor, b, ctx.ds.up_axis, ctx.clean)
            show(viz.curves_by_condition(rc, "cell_size", "entropy_bits", f"Cell size ({ctx.unit})", "Shannon entropy (bits)", log_x=True),
                 key="sens_res")
            st.caption("Finer cells -> more distinguishable locations -> higher entropy, for every participant.")
        with c2:
            st.markdown("**Entropy depends on observation duration**")
            dc = duration_curves(ctx.ds_id, ctx.pre_json, ctx.window, ctx.fcfg.gap_factor, ctx.cell, b, ctx.ds.up_axis, ctx.clean)
            show(viz.curves_by_condition(dc, "window_s", "entropy_bits", "Observation window: first T seconds", "Shannon entropy (bits)"),
                 key="sens_dur")
            st.caption(f"Computed on the first T seconds of each recording at {cs:g} {ctx.unit} cells. Comparing recordings of unequal length confounds duration with behaviour.")
    with tabs[4]:
        c1, c2 = st.columns([1, 1.1])
        with c1:
            st.markdown("**Coverage and entropy answer different questions**")
            st.markdown("- **Coverage** = fraction of cells visited *at all*. A single brief visit counts as much as an hour of dwelling.\n"
                        "- **Entropy** = how *evenly* the observed time is distributed. It rewards balanced allocation and penalises concentration.\n"
                        "- Entropy is therefore **not** a direct measure of exploration quality: a high value can come from aimless wandering, "
                        "a low value from deliberate, prolonged inspection of few objects.")
            ex = {"A: time evenly spread": np.array([.25, .25, .25, .25]), "B: 97 % in one cell": np.array([.97, .01, .01, .01])}
            show_df(pd.DataFrame([{"example (4 cells)": k, "coverage": ent.spatial_coverage(p), "entropy (bits)": round(ent.shannon_entropy(p), 3),
                                   "normalised": round(ent.normalized_entropy(p), 3)} for k, p in ex.items()]))
            st.caption("Both examples have 100 % coverage; entropy separates them.")
        with c2:
            show(viz.metric_scatter(ctx.metrics, "coverage", "entropy_norm", "Spatial coverage", "Normalised entropy"), key="cov_ent")


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Page 5 - clustering
# ═════════════════════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Running DBSCAN …", max_entries=12)
def cached_dbscan(ds_id: str, pre_json: str, window, parts: tuple, eps: float, ms: int, scaling: str, axes: tuple, stride: int, _kin: pd.DataFrame):
    k = _kin[_kin["participant_id"].isin(parts)]
    k = k[(k.groupby("participant_id").cumcount() % stride) == 0]
    pts = k[["participant_id", "condition", "t", *POS_COLS]].reset_index(drop=True)
    X = pts[list(axes)].to_numpy(float)
    res = cl.run_dbscan(X, eps, ms, scaling)
    return pts, X, res


def page_clustering(ctx: Ctx, focus: str) -> None:
    header("Unsupervised learning", "Unsupervised Clustering",
           "Two different questions: where do samples concentrate in space (point level), and which participants behave alike (participant level)?", ctx)
    callout("<b>Point-level vs participant-level clustering.</b> <i>Point level</i> clusters individual 6-DoF samples by position: it finds "
            "<b>dwell regions</b> (dense places) within the selected recordings. The points are autocorrelated samples, not independent observations. "
            "<i>Participant level</i> clusters one feature vector per participant: it asks whether <b>people</b> form behavioural types and is the "
            "only level at which agreement with the experimental condition can be assessed. Neither result is validated ground truth.", "violet")
    tabs = st.tabs(["Point-level · DBSCAN on positions", "Participant-level · behavioural types"])
    with tabs[0]:
        c1, c2, c3 = st.columns([2, 1, 1])
        by_c = {}
        for p, c in ctx.recordings:
            by_c.setdefault(c, []).append(p)
        default = [v[0] for v in by_c.values()][:4]
        parts = c1.multiselect("Participants", [p for p, _ in ctx.recordings], default=default, key="db_parts",
                               format_func=lambda p: f"{p} · {dict(ctx.recordings)[p]}")
        eps = c2.slider(f"ε - neighbourhood radius ({ctx.unit})", 0.05, 2.0, 0.25, 0.05, key="db_eps")
        ms = c3.slider("min_samples", 5, 300, 60, 5, key="db_ms", help="Number of samples (not seconds) within ε needed for a core point.")
        d1, d2, d3 = st.columns([1, 1, 1])
        scaling = d1.selectbox("Feature scaling", ["none", "standard", "minmax"], key="db_scale",
                               help="'none' keeps ε in real units. Standardising inflates tiny-variance axes such as head height.")
        use3 = d2.toggle("Use all 3 position axes", True, key="db_3d")
        stride = d3.slider("Use every k-th sample", 1, 10, 1, key="db_stride", help="Explicit analysis choice; 1 = all samples.")
        if not parts:
            st.info("Select at least one participant.")
        else:
            axes = tuple(POS_COLS) if use3 else tuple(horizontal_axes(ctx.ds.up_axis))
            try:
                pts, X, res = cached_dbscan(ctx.ds_id, ctx.pre_json, ctx.window, tuple(parts), float(eps), int(ms), scaling, axes, int(stride), ctx.kin)
            except ValueError as exc:
                st.error(str(exc))
                pts = None
            if pts is not None:
                med_dt = float(ctx.kin["dt"].median() * stride)
                kpis([("Clusters found", str(res.n_clusters), f"min_samples ≈ {ms * med_dt:.1f} s of dwell"),
                      ("Noise points", f"{100 * res.noise_fraction:.1f} %", f"{res.n_noise:,} of {len(pts):,}"),
                      ("Core points", f"{res.n_core:,}", "density-reachable seeds"),
                      ("Silhouette", f"{res.silhouette:.2f}" if res.silhouette is not None else "n/a",
                       "non-noise points only (optimistic)" if res.silhouette is not None else "needs ≥ 2 clusters")])
                if scaling == "standard" and use3:
                    callout("<b>Scaling warning.</b> Standardising makes the nearly constant head-height axis as important as the floor axes, "
                            "so ε no longer corresponds to a distance in real units.", "warn")
                fig, info = viz.clusters_3d(pts, res.labels, res.sizes, list(POS_COLS), ctx.unit)
                show(fig, key="db3d")
                if info["displayed"] < info["total"]:
                    st.caption(f"Display: random subset of {info['displayed']:,} of {info['total']:,} points. Clustering used all of them.")
                t1, t2, t3 = st.tabs(["Cluster statistics", "Cluster sizes", "Parameter check"])
                with t1:
                    cols_ = [c for c in axes]
                    show_df(cl.cluster_summary(pts, res.labels, cols_, med_dt).round(3))
                    st.caption("Dwell time = points × sampling interval. Clusters of a single participant mean little beyond that participant's own pauses.")
                with t2:
                    show(viz.cluster_size_bars(res.sizes, res.n_noise), key="db_sizes")
                    st.caption("The six largest clusters get fixed colours; any smaller ones share one neutral colour. Noise is DBSCAN label −1.")
                with t3:
                    k1, k2 = st.columns(2)
                    with k1:
                        kd = cl.k_distance(X, ms, scaling)
                        show(viz.k_distance_fig(kd, eps, ms, ctx.unit if scaling == "none" else "scaled"), key="kdist")
                        st.caption("Heuristic: choose ε near the 'knee' of the sorted k-distance curve.")
                    with k2:
                        sens = cl.eps_sensitivity(X, [eps * f for f in (0.5, 0.75, 1.0, 1.5, 2.0)], ms, scaling)
                        show_df(sens.round(3))
                        st.caption("Large changes of cluster count or noise for ±25 % in ε signal an unstable partition.")
        with st.expander("Scientific note: what DBSCAN does - and does not - tell you"):
            st.markdown(
                "**Algorithm.** A point is a *core point* if at least `min_samples` points (itself included) lie within distance ε. "
                "Clusters are maximal sets of density-connected core points plus their border points; everything else is labelled *noise* (−1). "
                "The number of clusters is not specified in advance and clusters may have arbitrary shape.")
            st.latex(r"N_\varepsilon(p)=\{q:\ \|p-q\|\le\varepsilon\},\qquad p\ \text{core}\iff |N_\varepsilon(p)|\ge \text{min\_samples}")
            st.markdown(
                "**Interpretation for 6-DoF data.** Samples are evenly spaced in *time*, so density is high where a person lingers or moves slowly: "
                "clusters are **dwell regions**, and `min_samples / sampling rate` is the minimum dwell in seconds. With walking speed ≈ 1.2 m/s "
                "and 30 Hz, consecutive samples are ≈ 4 cm apart, so a large ε with a small `min_samples` simply links the whole path into one cluster.\n\n"
                "**Caveats.** (1) Labels are not validated behavioural categories. (2) Results depend on ε, `min_samples` and scaling - use the "
                "parameter check. (3) A single global ε cannot represent regions of different density. (4) Points are autocorrelated, "
                "so cluster sizes are not sample sizes. (5) Orientation is not used here; clustering on position only ignores half of the 6-DoF signal.")
    with tabs[1]:
        cand = [m for m in ["mean_speed", "speed_cv", "mean_accel", "straightness", "turn_rate_per_min", "mean_ang_vel_dps",
                            "cum_rotation_deg", "entropy_norm", "coverage", "revisit_ratio", "path_length"] if m in ctx.metrics.columns]
        defaults = [m for m in ["mean_speed", "mean_ang_vel_dps", "turn_rate_per_min", "entropy_norm", "coverage", "revisit_ratio"] if m in cand]
        feats = st.multiselect("Behavioural features (one value per participant)", cand, defaults, format_func=ctx.lbl, key="pl_feats")
        kmax = st.slider("Largest k to try", 2, 6, 4, key="pl_k")
        if len(feats) < 2 or len(ctx.metrics) < 4:
            st.info("Select at least two features and ensure there are at least four participants.")
        else:
            try:
                out = cl.participant_level_clustering(ctx.metrics, feats, k_range=(2, kmax))
            except ValueError as exc:
                st.error(str(exc))
            else:
                kpis([("Best k (silhouette)", str(out["best_k"]), "chosen among 2…" + str(kmax)),
                      ("Silhouette", f"{out['silhouettes'][out['best_k']]:.2f}", "> 0.5 reasonable · < 0.25 weak"),
                      ("Agreement with condition (ARI)", fmt(out["ari_vs_condition"], 2), "1 = identical partition · 0 = chance"),
                      ("PC1 + PC2", f"{100 * sum(out['explained']):.0f} %", "variance shown in the plot")])
                c1, c2 = st.columns([1.3, 1])
                with c1:
                    show(viz.pca_scatter(out["pca"], ctx.metrics, out["labels"], out["explained"]), key="pl_pca")
                    st.caption("Colour = cluster, symbol = experimental condition (circle / diamond). Features are standardised; PCA is for display only.")
                with c2:
                    sil = pd.DataFrame({"k": list(out["silhouettes"]), "silhouette": list(out["silhouettes"].values())})
                    show_df(sil.round(3))
                    st.markdown("**PCA loadings**")
                    show_df(out["loadings"].round(2).reset_index(names="feature").assign(feature=lambda d: d["feature"].map(ctx.lbl)))
                tbl = ctx.metrics[["participant_id", "condition"]].assign(cluster=out["labels"])
                with st.expander("Cluster membership vs experimental condition"):
                    show_df(pd.crosstab(tbl["cluster"], tbl["condition"]).reset_index())
                    if ctx.ds.is_synthetic:
                        st.caption("In the simulation the conditions differ by construction, so recovered structure mainly shows that the "
                                   "features are informative - not that real participants fall into these types.")


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Page 6 - statistics
# ═════════════════════════════════════════════════════════════════════════════════════════════
def interpret(ctx: Ctx, r: pd.Series, a: str, b: str) -> str:
    m = r["metric"]
    unit = ctx.labels.get(m, {}).get("unit", "")
    txt = (f"**{ctx.lbl(m, False)}**: {b} − {a} = {fmt(r['diff'])} {unit} (95 % CI [{fmt(r['diff_ci_low'])}, {fmt(r['diff_ci_high'])}]); "
           f"Hedges' g = {fmt(r['hedges_g'], 2)} ({r['effect_label']}); {r['test_param']} p = {fmt_p(r['p_param'])}, Holm-adjusted p = {fmt_p(r['p_param_holm'])}; "
           f"{r['test_nonparam']} p = {fmt_p(r['p_nonparam'])}.")
    if np.isfinite(r["p_param_holm"]) and r["p_param_holm"] < 0.05:
        txt += " The difference exceeds what chance variation between participants would typically produce"
    else:
        txt += " The data do not distinguish this difference from chance variation between participants"
    txt += (", **in this simulation by construction of the generator**." if ctx.ds.is_synthetic else
            " (statistical statement only; design, sample size and confounds determine what it means).")
    return txt


def page_stats(ctx: Ctx, focus: str) -> None:
    header("Inference", "Statistical Comparison",
           "Participant-level comparison of two conditions: descriptive statistics, confidence intervals, effect sizes and tests.", ctx)
    if ctx.ds.is_synthetic:
        callout("<b>Warning - simulated data.</b> A statistically significant difference here only shows that the simulation was configured "
                "to differ; it is <b>not</b> evidence of real behavioural differences between people.", "warn")
    conds = ctx.conditions
    if len(conds) < 2:
        st.info("Only one condition is present, so no group comparison is possible. Map a condition column when importing a CSV.")
        return
    c1, c2 = st.columns(2)
    a = c1.selectbox("Reference group (A)", conds, index=0 if "Structured" not in conds else conds.index("Structured"), key="st_a")
    b_opts = [c for c in conds if c != a]
    b = c2.selectbox("Comparison group (B)", b_opts, index=b_opts.index("Free") if "Free" in b_opts else 0, key="st_b")
    cand = [m for m in default_stat_metrics(ctx.ds.has_rotation) + ["duration_s", "max_speed", "entropy_bits", "perplexity", "revisits_per_min",
                                                                  "ang_vel_p95_dps"] if m in ctx.metrics.columns]
    defaults = [m for m in default_stat_metrics(ctx.ds.has_rotation) if m in cand]
    metrics = st.multiselect("Metrics (one multiple-comparison family, Holm-corrected)", cand, defaults, format_func=ctx.lbl, key="st_metrics")
    if not metrics:
        st.info("Select at least one metric.")
        return
    tab = ctx.metrics[ctx.metrics["condition"].isin([a, b])]
    try:
        cmp = stt.compare_metrics(tab, metrics, a, b)
    except ValueError as exc:
        st.error(str(exc))
        return
    design = cmp["design"].iloc[0]
    n_a, n_b = int(cmp["n_a"].max()), int(cmp["n_b"].max())
    mde = stt.minimal_detectable_d(n_a, n_b)
    kpis([("Design", "Paired" if design == "paired" else "Between" if design == "independent" else "Mixed",
           "same participants in both conditions" if design == "paired" else "each participant in one condition"),
          (f"n · {a}", str(n_a), "participants (statistical units)"), (f"n · {b}", str(n_b), "participants (statistical units)"),
          ("Detectable effect", f"|d| ≥ {mde:.2f}", "80 % power, α = 0.05, two-sided")])
    st.caption("Frames are never treated as independent observations: each participant contributes one value per metric. Direction: "
               f"differences are {b} − {a}.")
    if design == "mixed":
        callout("Some participants appear in both conditions and others in one. Independent-groups tests are used, which is conservative but ignores pairing.", "warn")

    md("### Descriptive statistics by condition")
    rows = []
    for _, r in cmp.iterrows():
        for g, sfx in ((a, "a"), (b, "b")):
            rows.append({"metric": ctx.lbl(r["metric"]), "group": g, "n": int(r[f"n_{sfx}"]), "mean ± SD": f"{fmt(r[f'mean_{sfx}'])} ± {fmt(r[f'sd_{sfx}'])}",
                         "95 % CI of mean": f"[{fmt(r[f'ci_low_{sfx}'])}, {fmt(r[f'ci_high_{sfx}'])}]",
                         "median [IQR]": f"{fmt(r[f'median_{sfx}'])} [{fmt(r[f'iqr_{sfx}'])}]"})
    show_df(pd.DataFrame(rows), height=min(60 + 35 * len(rows), 520))

    md("### Tests and effect sizes")
    res = pd.DataFrame({
        "metric": [ctx.lbl(m) for m in cmp["metric"]],
        f"Δ ({b} − {a})": [fmt(x) for x in cmp["diff"]],
        "95 % CI of Δ": [f"[{fmt(l)}, {fmt(h)}]" for l, h in zip(cmp["diff_ci_low"], cmp["diff_ci_high"])],
        "g [95 % CI]": [f"{fmt(g, 2)} [{fmt(l, 2)}, {fmt(h, 2)}]" for g, l, h in zip(cmp["hedges_g"], cmp["g_ci_low"], cmp["g_ci_high"])],
        "effect": cmp["effect_label"],
        "p (Welch/paired)": [fmt_p(x) for x in cmp["p_param"]],
        "p Holm": [fmt_p(x) for x in cmp["p_param_holm"]],
        "p (rank test)": [fmt_p(x) for x in cmp["p_nonparam"]],
        "Cliff's δ": [fmt(x, 2) for x in cmp["cliffs_delta"]]})
    show_df(res, height=min(60 + 35 * len(res), 520))
    st.caption("g = Hedges' g (small-sample corrected standardised mean difference). Rank test = Mann–Whitney U (independent) or Wilcoxon "
               "signed-rank (paired). 'p Holm' corrects the parametric p-values across the selected metrics.")
    st.download_button("Download test results (CSV)", cmp.to_csv(index=False).encode(), "group_comparison.csv", "text/csv")

    md("### Distributions and effect sizes")
    l, r_ = st.columns([1.1, 1])
    with l:
        mm = st.selectbox("Metric to plot", metrics, format_func=ctx.lbl, key="st_plot")
        show(viz.box_with_points(tab, mm, ctx.lbl(mm), [a, b]), key="st_box")
        st.caption("Each marker is one participant. Box: median and quartiles; bar: mean with 95 % t-interval.")
    with r_:
        st.markdown("&nbsp;")
        show(viz.forest_plot(cmp, {m: ctx.lbl(m, False) for m in metrics}), key="st_forest")
        st.caption("Hedges' g with approximate 95 % CI (normal approximation; wide with small n).")

    md("### Interpretation")
    row = cmp[cmp["metric"] == mm].iloc[0]
    st.markdown(interpret(ctx, row, a, b))
    with st.expander("Assumption checks"):
        chk = pd.DataFrame({"metric": [ctx.lbl(m) for m in cmp["metric"]], f"Shapiro–Wilk p ({a})": [fmt_p(x) for x in cmp["shapiro_p_a"]],
                            f"Shapiro–Wilk p ({b})": [fmt_p(x) for x in cmp["shapiro_p_b"]], "Levene p (equal variances)": [fmt_p(x) for x in cmp["levene_p"]]})
        show_df(chk)
        st.caption("With about 10 participants per group these checks have little power; Welch's test does not assume equal variances and the "
                   "rank-based test is reported as a sensitivity analysis. Both are shown so that conclusions can be checked for robustness.")
    md("### Limitations")
    st.markdown(
        f"- **Small samples.** With {n_a} vs {n_b} participants only large effects are detectable (|d| ≳ {mde:.1f}); non-significant results are *not* evidence of no difference, "
        "and significant effect-size estimates from small samples tend to be inflated.\n"
        "- **Multiplicity.** Holm correction controls the family-wise error across the selected metrics only; metrics are correlated, and the "
        "choice of metrics should be pre-specified.\n"
        "- **Metric definitions matter.** Entropy and coverage depend on grid size and observation time (see *Exploration Entropy*); mean speed includes pauses.\n"
        "- **Effect sizes are descriptive** and their confidence intervals use a normal approximation.\n"
        + ("- **Synthetic data.** Results reflect the generator's parameters and make no claim about human behaviour.\n" if ctx.ds.is_synthetic else
           "- **Causality.** Group differences in an uploaded dataset are only interpretable given the experimental design (randomisation, counterbalancing, confounds).\n"))


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Page 7 - methodology
# ═════════════════════════════════════════════════════════════════════════════════════════════
def page_method(ctx: Ctx, s: dict) -> None:
    header("Documentation", "Methodology & Reproducibility",
           "Definitions, assumptions and everything needed to reproduce the numbers in this dashboard.", ctx)
    if ctx.ds.is_synthetic:
        md("## Dataset generation (synthetic)")
        st.markdown(
            "- **Arena** 10 m × 10 m, *z* up, eye height ≈ 1.65 m (± 0.06 m between participants, ≈ 1 cm walking bob).\n"
            "- **Participants** 20 (10 per condition, random assignment), one recording each (between-subject design), 60–120 s at 30 Hz.\n"
            "- **Locomotion** Kinematic model: first-order speed dynamics (τ = 0.5 s), first-order turn-rate dynamics (τ = 0.25 s), soft wall repulsion, "
            "Ornstein–Uhlenbeck heading noise (σ differs by condition), dwell at each target. Cruise speed ~ N(1.15, 0.10) m/s in **both** conditions.\n"
            "- **Targets** 5 × 5 lattice (2 m spacing). *Structured*: boustrophedon sweep, then reversed. *Free*: random hops with return probability ≈ 0.3 and "
            "random re-targeting.\n"
            "- **Head orientation** body heading + critically damped gaze process (yaw, pitch) + slow roll; ZYX Euler → unit quaternion (x, y, z, w), "
            "canonical hemisphere *w* ≥ 0 (which creates real sign flips), plus small-angle jitter (≈ 0.01°) and 0.2 mm position jitter.\n"
            "- **Participant-level random effects** (cruise speed, head-movement vigour) overlap between conditions so the groups are not trivially separable.\n"
            "- **Not real data.** Differences between conditions exist because the generator was configured to produce them.")
    md("## Mathematical definitions")
    st.markdown("**Translational movement** (valid steps only, i.e. Δt ≤ gap threshold × median Δt)")
    st.latex(r"L=\sum_i \lVert \mathbf p_i-\mathbf p_{i-1}\rVert,\qquad v_i=\frac{\lVert \mathbf p_i-\mathbf p_{i-1}\rVert}{t_i-t_{i-1}},\qquad \bar v=\frac{L}{T_{\mathrm{obs}}}")
    st.latex(r"\mathbf a_i=\frac{\mathbf v_{i+1}-\mathbf v_{i}}{\tfrac12(\Delta t_i+\Delta t_{i+1})},\qquad \mathrm{CV}_v=\frac{\sigma_v}{\bar v},\qquad \text{straightness}=\frac{\lVert \mathbf p_N-\mathbf p_0\rVert}{L}")
    st.markdown("Speed variability uses time-weighted moments so irregular sampling does not bias them.")
    st.markdown("**Rotational movement** - from *relative quaternions*, never by differencing Euler angles")
    st.latex(r"\mathbf q^{\mathrm{rel}}_i=\mathbf q_{i-1}^{-1}\otimes \mathbf q_i=(\mathbf u,w),\qquad \theta_i=2\,\operatorname{atan2}\!\left(\lVert\mathbf u\rVert,\,|w|\right)\in[0,\pi]")
    st.latex(r"\omega_i=\frac{\theta_i}{t_i-t_{i-1}},\qquad \Theta=\sum_i\theta_i,\qquad \bar\omega=\frac{\Theta}{T_{\mathrm{obs}}}")
    st.markdown("`|w|` makes the angle invariant to the quaternion sign (**q** and −**q** are the same rotation); `atan2` is accurate for tiny and near-π angles. "
                "The angle is the same for body→world and world→body conventions.")
    st.markdown("**Direction changes** - positions are sub-sampled at the stride, short displacements are discarded, and for successive displacement vectors **a**, **b**:")
    st.latex(r"\psi=\operatorname{atan2}\!\left(\lVert \mathbf a\times\mathbf b\rVert,\ \mathbf a\cdot\mathbf b\right),\qquad \text{count } \psi>\psi_{\mathrm{thr}}")
    st.markdown("**Spatial occupancy, entropy and revisitation** - grid of K square cells over a bounding box shared by all recordings; each sample carries its trapezoidal duration *w*ᵢ.")
    st.latex(r"p_k=\frac{\sum_{i\in k}w_i}{\sum_i w_i},\qquad H=-\sum_{k}p_k\log_2 p_k,\qquad H_{\max}=\log_2 K,\qquad \tilde H=\frac{H}{\log_2 K}")
    st.latex(r"\text{coverage}=\frac{|\{k:p_k>0\}|}{K},\qquad \text{perplexity}=2^{H},\qquad \text{revisit ratio}=\frac{\#\text{visits}-\#\text{distinct cells in the visit sequence}}{\#\text{visits}}")
    st.markdown("A *visit* is a maximal run of samples in one cell; boundary flicker is suppressed with a hysteresis margin of 25 % of the cell size. "
                "**Entropy depends on the grid resolution and the observation duration** (see the sensitivity plots). "
                "Normalisation by log₂K is valid because H ≤ log₂K (equality for a uniform distribution over K cells), but it is only interpretable if all K cells are reachable. "
                "The plug-in estimator is used without bias correction because samples are strongly autocorrelated.")
    st.markdown("**Statistics** - participants are the units. Welch's test, effect size and Holm correction:")
    st.latex(r"t=\frac{\bar x_B-\bar x_A}{\sqrt{s_A^2/n_A+s_B^2/n_B}},\qquad \nu=\frac{(s_A^2/n_A+s_B^2/n_B)^2}{\frac{(s_A^2/n_A)^2}{n_A-1}+\frac{(s_B^2/n_B)^2}{n_B-1}}")
    st.latex(r"g=J(\nu')\,\frac{\bar x_B-\bar x_A}{s_p},\ \ J\approx 1-\frac{3}{4\nu'-1},\qquad \delta=\frac{\#(x_B>x_A)-\#(x_B<x_A)}{n_An_B},\qquad \tilde p_{(j)}=\max_{i\le j}\,(m-i+1)\,p_{(i)}")
    md("## Preprocessing methods and their effect")
    show_ops(ctx.qc.operations)
    st.markdown("Quality checks: missing values, timestamp validation (monotonic, duplicates), sampling interval (median Δt, CV), position discontinuities "
                "(step speed > limit), Tukey-fence outliers on step speed, quaternion-norm deviation. **Inspection never alters data; every modification is a logged, user-controlled operation.**")
    md("## Statistical assumptions")
    st.markdown("- Participants are independent; one value per participant and metric. Repeated measures (same participants in both conditions) are detected and analysed with paired tests.\n"
                "- Welch's *t* assumes approximately normal group means (reasonable for participant-level means, checked with Shapiro–Wilk); Mann–Whitney is a rank-based check.\n"
                "- Confidence intervals of the mean use the *t* distribution; CIs of Hedges' *g* use a normal approximation and are approximate for small *n*.\n"
                "- Multiple metrics are Holm-corrected as one family; choose the family before looking at the results.")
    md("## Limitations")
    st.markdown("- Entropy, coverage and revisits depend on grid size, bounding box and recording length; they are not measures of exploration *quality*.\n"
                "- DBSCAN clusters are density descriptors, not validated behavioural states.\n"
                "- Maximum speed / acceleration are noise-sensitive; the 95th percentile is provided as a robust alternative.\n"
                "- Smoothing trades jitter against attenuating real peaks; finite-difference acceleration at 30 Hz amplifies sensor noise.\n"
                "- Synthetic data cannot validate behavioural claims; it validates the *pipeline* (see the ground-truth check below).")
    md("## Research questions → methods")
    show_df(pd.DataFrame({
        "Research question": ["Do conditions differ in translational movement?", "Do conditions differ in head rotation?",
                              "How is time spread over space?", "Where do people dwell?", "Are there behavioural types of participants?",
                              "Is data quality sufficient?"],
        "Features": ["path length, speed, acceleration, straightness", "angular velocity, cumulative rotation, direction changes",
                     "occupancy, entropy, coverage, revisits", "point density", "participant feature vectors", "QC report"],
        "Method": ["participant-level Welch / Mann–Whitney, Hedges' g", "same", "grid-based Shannon entropy + sensitivity analysis",
                   "DBSCAN with parameter check", "k-means + silhouette + PCA, ARI vs condition", "inspection + logged preprocessing"]}))

    if ctx.ds.is_synthetic:
        md("## Pipeline validation against simulation ground truth")
        truth = ctx.ds.meta["participants"].merge(ctx.metrics, on=["participant_id", "condition"])
        if len(truth) > 3:
            r1 = np.corrcoef(truth["true_cruise_speed"], truth["speed_p95"])[0, 1]
            r2 = np.corrcoef(truth["true_gaze_yaw_sd_deg"], truth["mean_ang_vel_dps"])[0, 1] if "mean_ang_vel_dps" in truth else np.nan
            kpis([("r(true cruise speed, measured 95th-pct speed)", f"{r1:.2f}", "feature extraction recovers a known parameter"),
                  ("r(true gaze-yaw SD, measured mean angular velocity)", f"{r2:.2f}", "rotation metric tracks head-movement amplitude")])
            st.caption("Pearson correlations over participants. They test the pipeline, not behaviour: the generator's parameters are known.")
    md("## Reproducibility")
    ds = ctx.ds
    info = pd.DataFrame({"item": ["Dataset", "Dataset fingerprint (SHA-256, truncated)", "Python", "NumPy / pandas / SciPy / scikit-learn", "Streamlit / Plotly",
                                  "Preprocessing configuration", "Feature configuration", "Spatial grid", "Common analysis window"],
                         "value": [ds.name, ds.meta.get("fingerprint", "")[:24], sys.version.split()[0],
                                   " / ".join(_ver(m) for m in ("numpy", "pandas", "scipy", "sklearn")), f"{_ver('streamlit')} / {_ver('plotly')}",
                                   ctx.pre_json, ctx.feat_json, f"{ctx.grid.nx} × {ctx.grid.ny} cells of {ctx.cell:g} {ctx.unit}",
                                   "off" if ctx.window is None else f"first {ctx.window:g} s"]})
    if ds.is_synthetic:
        info.loc[len(info)] = ["Generator", f"seed {ds.meta['seed']}, n = {ds.meta['n_participants']}, {ds.meta['fs_hz']:g} Hz, v{ds.meta['generator_version']}"]
    info["value"] = info["value"].astype(str)
    show_df(info)
    st.markdown("Re-create the numbers from the command line with `python -m src.pipeline --seed 42`; the full test-suite runs with `pytest`.")
    d1, d2 = st.columns(2)
    if ds.is_synthetic:
        d1.download_button("Download synthetic dataset (CSV)", ds.df.to_csv(index=False).encode(), "synthetic_6dof.csv", "text/csv")
    d2.download_button("Download participant metrics (CSV)", ctx.metrics.to_csv(index=False).encode(), "participant_metrics.csv", "text/csv")


def _ver(mod: str) -> str:
    try:
        return __import__(mod).__version__
    except Exception:
        return "?"


# ═════════════════════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════════════════════
def main() -> None:
    md(CSS)
    s = sidebar_controls()
    ctx = None
    try:
        ctx = build_ctx(s)
    except Exception as exc:                                  # surface problems instead of a stack trace
        st.error(f"Analysis failed: {exc}")
        st.exception(exc)
        return
    if ctx is not None and ctx.metrics.empty:
        header("Data problem", "No analysable recordings", "", ctx)
        st.error("None of the recordings could be analysed.")
        for k, v in ctx.skipped.items():
            st.write(f"• {k}: {v}")
        return
    page = s["page"]
    if ctx is None:
        if s["source"] == SRC_UP:
            page_overview(None, s)
        return
    focus = focus_selector(ctx)
    {"Research Overview": lambda: page_overview(ctx, s),
     "3D Trajectory Explorer": lambda: page_trajectory(ctx, focus),
     "Movement Analytics": lambda: page_movement(ctx, focus),
     "Exploration Entropy": lambda: page_entropy(ctx, focus),
     "Unsupervised Clustering": lambda: page_clustering(ctx, focus),
     "Statistical Comparison": lambda: page_stats(ctx, focus),
     "Methodology & Reproducibility": lambda: page_method(ctx, s)}[page]()
    with st.sidebar:
        md('<div class="side-label">About</div><div class="small">Technical demonstration. Synthetic data are simulated and are not '
           'observations from real VR experiments. Not a replication of any published study.</div>')


main()
