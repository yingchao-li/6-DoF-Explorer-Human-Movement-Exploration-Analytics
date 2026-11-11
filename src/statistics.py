"""Participant-level statistical comparison of two experimental conditions.

Key rule: **the statistical unit is the participant**, never the time-series
frame. A recording with 3 000 highly autocorrelated samples is still one
observation. :func:`assert_participant_level` refuses tables that contain
several rows per (participant, condition).

Tests used
----------
* independent groups: Welch's t-test (primary, no equal-variance assumption)
  and Mann-Whitney U (rank-based sensitivity check);
* repeated measures (same participants in both conditions): paired t-test and
  Wilcoxon signed-rank test;
* effect sizes: Hedges' g (small-sample corrected standardised mean
  difference; approximate normal CI) and Cliff's delta (ordinal);
* multiplicity: Holm-Bonferroni adjustment across the family of metrics.

Direction convention: all differences are ``group_b - group_a``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats


def assert_participant_level(table: pd.DataFrame, participant_col: str = "participant_id",
                             condition_col: str = "condition") -> None:
    """Raise if the table is not one-row-per-participant-per-condition."""
    for c in (participant_col, condition_col):
        if c not in table.columns:
            raise ValueError(f"Column {c!r} missing: pass a participant-level table.")
    dup = table.duplicated(subset=[participant_col, condition_col], keep=False)
    if dup.any():
        raise ValueError(
            f"{int(dup.sum())} rows share a (participant, condition) pair: this looks like frame-level data. "
            "Aggregate to one row per participant first - frames of one participant are not independent observations."
        )


def detect_design(table: pd.DataFrame, group_a: str, group_b: str, participant_col: str = "participant_id",
                  condition_col: str = "condition") -> str:
    """'paired' if every participant appears in both groups, 'independent' if none does, else 'mixed'."""
    sub = table[table[condition_col].isin([group_a, group_b])]
    per = sub.groupby(participant_col)[condition_col].nunique()
    if len(per) and (per == 2).all():
        return "paired"
    if (per == 2).any():
        return "mixed"
    return "independent"


def holm_adjust(p) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1). NaNs are ignored."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def hedges_correction(df_total: float) -> float:
    """Small-sample bias correction J(df) (exact gamma-ratio form)."""
    if df_total <= 0:
        return np.nan
    from scipy.special import gammaln
    return float(np.exp(gammaln(df_total / 2) - 0.5 * np.log(df_total / 2) - gammaln((df_total - 1) / 2)))


def cliffs_delta(a, b) -> float:
    """P(b > a) - P(b < a) in [-1, 1]."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    diff = b[:, None] - a[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / (a.size * b.size))


def _interp_d(d: float) -> str:
    d = abs(d)
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"


def _interp_delta(d: float) -> str:
    d = abs(d)
    return "negligible" if d < 0.147 else "small" if d < 0.33 else "medium" if d < 0.474 else "large"


def describe(x) -> dict:
    """Mean, SD, median, IQR and the 95 % t-interval of the mean."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, median=np.nan, q1=np.nan, q3=np.nan, iqr=np.nan,
                    ci_low=np.nan, ci_high=np.nan)
    sd = float(x.std(ddof=1)) if n > 1 else np.nan
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    if n > 1:
        h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        lo, hi = x.mean() - h, x.mean() + h
    else:
        lo = hi = np.nan
    return dict(n=int(n), mean=float(x.mean()), sd=sd, median=float(med), q1=float(q1), q3=float(q3),
                iqr=float(q3 - q1), ci_low=float(lo), ci_high=float(hi))


def compare_two_groups(a, b, paired: bool = False) -> dict:
    """Compare group ``b`` against reference group ``a`` (difference = b - a)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if paired:
        if a.size != b.size:
            raise ValueError("Paired comparison needs equally many observations.")
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
    else:
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    na, nb = a.size, b.size
    res = {"n_a": na, "n_b": nb, "paired": paired}
    nan = np.nan
    if na < 2 or nb < 2:
        res.update(diff=nan, diff_ci_low=nan, diff_ci_high=nan, t=nan, df=nan, p_param=nan, p_nonparam=nan,
                   hedges_g=nan, g_ci_low=nan, g_ci_high=nan, cliffs_delta=nan, shapiro_p_a=nan, shapiro_p_b=nan,
                   levene_p=nan, note="fewer than 2 observations in a group")
        return res
    diff = float(b.mean() - a.mean())
    if paired:
        d = b - a
        sd_d = float(d.std(ddof=1))
        se = sd_d / np.sqrt(na)
        dfree = na - 1
        tcrit = stats.t.ppf(0.975, dfree)
        if sd_d == 0:
            t_stat, p_param = nan, nan
        else:
            t_stat, p_param = stats.ttest_rel(b, a)
        try:
            p_np = float(stats.wilcoxon(b, a, zero_method="wilcox", method="auto").pvalue)
        except ValueError:
            p_np = nan
        dz = diff / sd_d if sd_d > 0 else nan
        g = dz * hedges_correction(dfree) if np.isfinite(dz) else nan
        se_g = np.sqrt(1 / na + (g ** 2) / (2 * na)) if np.isfinite(g) else nan
        delta = nan
        res.update(test_param="Paired t-test", test_nonparam="Wilcoxon signed-rank")
    else:
        sa, sb = a.var(ddof=1), b.var(ddof=1)
        se = float(np.sqrt(sa / na + sb / nb))
        if se > 0:
            dfree = se ** 4 / ((sa / na) ** 2 / (na - 1) + (sb / nb) ** 2 / (nb - 1))
            tcrit = stats.t.ppf(0.975, dfree)
            t_stat, p_param = stats.ttest_ind(b, a, equal_var=False)
        else:
            dfree, tcrit, t_stat, p_param = nan, nan, nan, nan
        try:
            p_np = float(stats.mannwhitneyu(b, a, alternative="two-sided", method="auto").pvalue)
        except ValueError:
            p_np = nan
        sp = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2))
        d_c = diff / sp if sp > 0 else nan
        g = d_c * hedges_correction(na + nb - 2) if np.isfinite(d_c) else nan
        se_g = np.sqrt((na + nb) / (na * nb) + (g ** 2) / (2 * (na + nb))) if np.isfinite(g) else nan
        delta = cliffs_delta(a, b)
        res.update(test_param="Welch t-test", test_nonparam="Mann-Whitney U")
    res.update(
        diff=diff,
        diff_ci_low=diff - tcrit * se if np.isfinite(tcrit) else nan,
        diff_ci_high=diff + tcrit * se if np.isfinite(tcrit) else nan,
        t=float(t_stat) if np.isfinite(t_stat) else nan, df=float(dfree) if np.isfinite(dfree) else nan,
        p_param=float(p_param) if np.isfinite(p_param) else nan, p_nonparam=p_np,
        hedges_g=float(g) if np.isfinite(g) else nan,
        g_ci_low=float(g - 1.96 * se_g) if np.isfinite(g) else nan,
        g_ci_high=float(g + 1.96 * se_g) if np.isfinite(g) else nan,
        cliffs_delta=delta,
        effect_label=_interp_d(g) if np.isfinite(g) else "n/a",
        delta_label=_interp_delta(delta) if np.isfinite(delta) else "n/a",
        shapiro_p_a=float(stats.shapiro(a).pvalue) if na >= 3 and np.ptp(a) > 0 else nan,
        shapiro_p_b=float(stats.shapiro(b).pvalue) if nb >= 3 and np.ptp(b) > 0 else nan,
        levene_p=float(stats.levene(a, b, center="median").pvalue) if not paired and (np.ptp(a) > 0 or np.ptp(b) > 0) else nan,
    )
    return res


def compare_metrics(table: pd.DataFrame, metrics: list[str], group_a: str, group_b: str,
                    participant_col: str = "participant_id", condition_col: str = "condition",
                    design: str | None = None) -> pd.DataFrame:
    """Run :func:`compare_two_groups` for several metrics and Holm-adjust across them.

    ``table`` must be participant-level (checked). ``design`` is detected when
    not given; for ``'mixed'`` designs the independent-groups tests are used
    and a note is added.
    """
    assert_participant_level(table, participant_col, condition_col)
    if group_a == group_b:
        raise ValueError("Select two different conditions.")
    design = design or detect_design(table, group_a, group_b, participant_col, condition_col)
    paired = design == "paired"
    sub = table[table[condition_col].isin([group_a, group_b])]
    rows = []
    for m in metrics:
        if paired:
            wide = sub.pivot(index=participant_col, columns=condition_col, values=m).dropna()
            a, b = wide[group_a].to_numpy(), wide[group_b].to_numpy()
        else:
            a = sub.loc[sub[condition_col] == group_a, m].to_numpy(float)
            b = sub.loc[sub[condition_col] == group_b, m].to_numpy(float)
        r = compare_two_groups(a, b, paired=paired)
        da, db = describe(a), describe(b)
        r.update(metric=m, design=design,
                 mean_a=da["mean"], sd_a=da["sd"], median_a=da["median"], iqr_a=da["iqr"],
                 ci_low_a=da["ci_low"], ci_high_a=da["ci_high"],
                 mean_b=db["mean"], sd_b=db["sd"], median_b=db["median"], iqr_b=db["iqr"],
                 ci_low_b=db["ci_low"], ci_high_b=db["ci_high"])
        rows.append(r)
    out = pd.DataFrame(rows)
    out["p_param_holm"] = holm_adjust(out["p_param"].to_numpy())
    out["p_nonparam_holm"] = holm_adjust(out["p_nonparam"].to_numpy())
    return out


def minimal_detectable_d(n_a: int, n_b: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest standardised mean difference detectable with the given power (two-sided t-test)."""
    df = n_a + n_b - 2
    if df < 1:
        return float("nan")
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    k = np.sqrt(n_a * n_b / (n_a + n_b))

    def f(d: float) -> float:
        ncp = d * k
        lower = np.nan_to_num(stats.nct.cdf(-tcrit, df, ncp), nan=0.0)   # far tail: SciPy returns NaN, true value ~ 0
        return (stats.nct.sf(tcrit, df, ncp) + lower) - power

    try:
        return float(optimize.brentq(f, 1e-3, 8.0))
    except ValueError:                       # no root in range (tiny groups) or numerical failure
        return float("nan")
