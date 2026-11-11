# 6-DoF Explorer — Human Movement & Exploration Analytics

A small, fully local research dashboard for analysing **six-degrees-of-freedom (6-DoF)** tracking data:
three translational (x, y, z) and three rotational degrees of freedom (head orientation as a unit quaternion),
sampled at tens of Hertz. It covers the whole path from raw samples to participant-level statistics:
quality control → kinematic features → spatial exploration metrics → unsupervised learning → inference.

> **All built-in data are synthetic.** They are produced by a seeded simulation and are **not observations from
> real VR experiments**. Group differences in the demo data exist because the generator was configured to
> produce them; they say nothing about human behaviour. This project is a technical demonstration and does
> **not** replicate or reproduce any published study.

```
Python 3.11+ · Streamlit · Plotly · NumPy · Pandas · SciPy · scikit-learn · pytest · no cloud, no accounts, no paid services
```

---

## Motivation

Head-mounted displays and motion-capture systems record where a person is and where they look, many times
per second. Behavioural research on exploration asks *how* people move through an environment: how far and how
fast, how much they turn their head, which regions they cover, how evenly time is spread over space, and how
often they return to places. Answering such questions reliably is mostly a matter of **careful representation and
careful bookkeeping** — rotations are not vectors, samples are not independent observations, and entropy is not
a free-standing number. This project is a compact, reproducible implementation of those details.

## Research questions the dashboard is built around

1. Do conditions differ in **translational** behaviour (path length, speed, acceleration)?
2. Do they differ in **rotational** behaviour (angular velocity, cumulative head rotation)?
3. How is observation time distributed over space — **coverage, entropy, revisitation**?
4. Where do people dwell (point-level density clusters), and do **participants** form behavioural types?
5. Is the data quality sufficient to answer any of this?

## What the dashboard contains

| Page | Purpose |
|---|---|
| **Research Overview** | Motivation, dataset summary, data-quality badges and tables, preprocessing log, injected-vs-detected artifacts (synthetic only), CSV import |
| **3D Trajectory Explorer** | Interactive 3D paths coloured by time / speed / angular velocity / condition, head-direction arrows from the quaternion, camera presets, time-range slider, top-down projection, window metrics |
| **Movement Analytics** | Speed, acceleration, angular velocity, cumulative rotation, direction changes; time series, distributions, cumulative distance, per-participant table |
| **Exploration Entropy** | Occupancy maps, Shannon entropy, normalised entropy, coverage, revisits, **grid-size slider**, sensitivity to resolution and observation duration |
| **Unsupervised Clustering** | DBSCAN on positions (ε, `min_samples`, scaling, participant selection; k-distance plot, ε sensitivity, noise) and, separately, participant-level clustering with PCA |
| **Statistical Comparison** | Participant-level descriptives, CIs, Welch/paired tests and rank tests, Hedges' g, Cliff's δ, Holm correction, assumption checks, detectable effect size |
| **Methodology & Reproducibility** | Formulas (`st.latex`), preprocessing effects, assumptions, limitations, library versions, seed, dataset fingerprint, configuration, downloads |

## Architecture

Scientific code is independent of the UI and can be imported or run from the command line.

```
6dof-explorer/
├── app.py                    # Streamlit UI only: layout, caching, theme
├── requirements.txt
├── pytest.ini
├── .streamlit/config.toml    # dark theme, local-only settings
├── src/
│   ├── data_generator.py     # seeded synthetic 6-DoF recordings (+ optional tracking artifacts)
│   ├── data_loader.py        # CSV import: column mapping, units, validation, Dataset container
│   ├── preprocessing.py      # quality inspection (read-only) and logged preprocessing operations
│   ├── features.py           # quaternion maths, translational / rotational / direction-change features
│   ├── entropy.py            # spatial grid, occupancy, Shannon entropy, coverage, revisits, sensitivity
│   ├── clustering.py         # DBSCAN (point level) and k-means + PCA (participant level)
│   ├── statistics.py         # participant-level tests, effect sizes, Holm, power
│   ├── pipeline.py           # glue: stages used by the app and by `python -m src.pipeline`
│   └── visualization.py      # Plotly figure builders (validated dark palette)
└── tests/                    # 104 tests (features, entropy, preprocessing, clustering, statistics, generator, loader, pipeline)
```

## Installation and launch

```bash
cd 6dof-explorer
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. Nothing leaves the machine; uploaded CSV files are processed in memory.

Command-line analysis without the UI (prints QC, group means and the participant-level comparison):

```bash
python -m src.pipeline                           # default seed 42, 20 participants
python -m src.pipeline --seed 7 --artifacts      # inject tracking artifacts to exercise the QC code
python -m src.pipeline --csv synthetic.csv       # also write the synthetic dataset
```

Run the tests:

```bash
pytest
```

## Methods in brief

Full formulas are rendered on the *Methodology & Reproducibility* page.

**Quaternions.** Scalar-last `(x, y, z, w)`, Hamilton product. The rotation between consecutive samples is the
relative quaternion `q_rel = q_{i-1}⁻¹ ⊗ q_i`, and its angle is `θ = 2·atan2(‖v‖, |w|)`. Using `|w|` makes the
angle invariant to the quaternion sign (`q` and `−q` are the same rotation, and many trackers flip signs), and
`atan2` is accurate for very small and near-π angles. The angle does not depend on handedness or on
body→world versus world→body convention. **Euler angles are never subtracted**; they are computed only for display.

**Translational features** use the actual timestamp differences: path length, speed (mean, maximum, 95th
percentile, coefficient of variation), acceleration (centred difference on irregular spacing), net displacement and
straightness. Steps spanning a tracking gap (Δt > gap factor × median Δt) are excluded from speed, path and
angular statistics instead of being interpolated across.

**Rotational features:** angular displacement and velocity between samples, cumulative rotation, mean / maximum /
95th-percentile angular velocity.

**Direction changes:** displacement vectors on a coarse stride (default 0.5 s), tiny steps discarded, turning angle
`atan2(‖a×b‖, a·b)`, counted above a threshold (default 45°) and normalised per minute.

**Spatial entropy.** Positions are projected onto the horizontal plane and binned on a square grid whose bounding
box is **shared by all recordings** (so cells mean the same thing for everyone). Occupancy is the fraction of *observed
time* per cell (trapezoidal sample durations; gaps excluded). Plug-in Shannon entropy
`H = −Σ p log₂ p` is reported in bits, normalised entropy `H / log₂K`, effective cells `2^H`, and coverage (fraction
of occupied cells). Normalisation is valid because `H ≤ log₂K`, but 1 is only attainable if all `K` cells are
reachable. **Entropy depends on the grid resolution and on the observation duration** — the dashboard shows both
sensitivity curves and offers an optional *common analysis window* so that recordings are compared at equal
observation time. Revisits are counted on run-length visits with a small hysteresis margin (25 % of the cell size)
so that a path wobbling across a cell boundary is not counted as many revisits.

**Clustering.** DBSCAN on 3D positions (optionally standardised / min-max scaled). Because samples are evenly spaced in
time, density is high where a person lingers, so clusters are **dwell regions**, and `min_samples / sampling rate` is a
minimum dwell time. Guard rails bound memory (maximum number of points and neighbour pairs). Participant-level
clustering is a separate analysis: one standardised feature vector per participant → k-means (k chosen by
silhouette) with PCA for display, and agreement with the experimental condition measured by the adjusted Rand index.

**Statistics.** The participant is the statistical unit: frame-level tables are rejected with an error. Welch's *t*
test (or the paired *t* test when every participant appears in both conditions) with the Mann–Whitney U (or Wilcoxon
signed-rank) as a sensitivity analysis; Hedges' *g* with an approximate CI, Cliff's δ, Holm correction across the
selected metrics, Shapiro–Wilk and Levene checks, and the minimal detectable effect size for the given group sizes.
Differences are always *group B − group A*.

## Preprocessing and quality control

`inspect_quality` is read-only. `preprocess` runs a fixed, *logged* sequence — sort, drop non-finite rows, drop duplicate
timestamps, renormalise quaternions, flag position discontinuities and statistical outliers, optionally repair
**isolated** single-sample spikes, optionally resample (linear / SLERP, never across gaps) and optionally smooth
(Gaussian; normalised lerp for quaternions). Every operation records rows before / after / affected and what it does
to the analysis. **Nothing is removed silently**, flagged samples are kept unless repair is switched on, and
recordings that cannot be analysed are listed with the reason. Downsampling happens only for drawing (point
count shown in the UI) and never for calculations.

## Dataset assumptions

**Synthetic data** (`src/data_generator.py`): a 10 m × 10 m arena (z up, eye height ≈ 1.65 m), 20 participants
(10 per condition, between-subject), 60–120 s at 30 Hz, fixed seed. Locomotion is a smooth kinematic model with
first-order speed and turn-rate dynamics, wall repulsion and Ornstein–Uhlenbeck heading noise; head orientation is
body heading plus a critically damped gaze process, converted to unit quaternions and stored in the canonical `w ≥ 0`
hemisphere (which creates genuine sign flips that the analysis must handle). *Structured* participants sweep a 5 × 5
lattice of targets in a boustrophedon pattern and then reverse; *Free* participants hop between targets at random with a
tendency to return to visited ones. Cruise-speed distributions are identical across conditions; participant-level random
effects make the groups overlap. Optional artifacts (spikes, gaps, duplicates, NaNs, non-unit quaternions) come with
ground-truth counts.

**Uploaded CSV:** one row per sample. Columns are mapped explicitly (participant — optional, timestamp, x, y, z,
four quaternion components, condition — optional). Nothing is assumed about the coordinate system, units, quaternion
convention or sampling rate: you declare the time unit, position unit, vertical axis and (for drawing arrows only) the
local forward axis. Quaternion components are mapped one by one, so file order does not matter. Without quaternions
all rotation-based features are disabled, and without a condition column group comparisons are disabled. Synthetic and
uploaded data are held in separate objects with different dataset identifiers; they are never merged or mixed in
caches.

## Limitations

- Entropy, coverage and revisit counts depend on grid size, bounding box and recording length and are not measures of
  exploration *quality*. A high value can reflect aimless wandering; a low value, deliberate inspection of few places.
- DBSCAN clusters are density descriptors, not validated behavioural states; they depend on ε, `min_samples` and
  scaling, and the points are autocorrelated. The silhouette score is computed on non-noise points only and is optimistic.
- Maximum speed and acceleration are noise-sensitive; 95th-percentile alternatives are provided. Finite-difference
  acceleration at 30 Hz amplifies sensor noise, and smoothing trades jitter against flattened peaks.
- With about ten participants per group only large effects are detectable (the dashboard prints the minimal detectable
  effect size). Effect-size CIs use a normal approximation. Holm correction covers the selected metric family only.
- In the simulation, statistical significance reflects the generator settings. It is not evidence about people.
- Orientation is not used for DBSCAN and no gaze-target or object information is modelled.

## Reproducibility

All randomness is seeded (`numpy.random.default_rng`). The *Methodology & Reproducibility* page lists the Python and
library versions, the generator seed and settings, a SHA-256 fingerprint of the data, and the exact preprocessing and
feature configuration as JSON. The command-line pipeline reproduces the dashboard numbers
(`python -m src.pipeline --seed 42`). Tests fix seeds and compare against closed-form results.

## Tests

`pytest` runs 104 tests, including the checks that matter most for correctness:

- stationary position → path length 0; constant velocity → constant speed; irregular sampling handled with real Δt;
  circular motion → centripetal acceleration
- stationary orientation → zero angular displacement; quaternion sign flips → no jumps; known rotation rates recovered;
  small and near-π angles
- empty or invalid input → clear errors (`TrajectoryError`, `ValueError` with explanatory messages)
- entropy 0 for a single occupied bin, `log₂K` for a uniform distribution, refinement never lowers entropy,
  time-weighted occupancy with irregular sampling
- DBSCAN: well-separated blobs recovered, isolated points labelled noise, memory guards
- statistics: frame-level tables rejected, agreement with SciPy, Holm, paired designs, type-I error rate under the null
- preprocessing: exact detection counts for injected artifacts, logged removals, no interpolation across gaps
- CSV import: unit conversion, per-component quaternion mapping, validation errors, source separation
- pipeline: one row per participant, determinism, designed differences recovered with the right sign

## Relevance to 6-DoF Behavioral Research

The project is meant as a compact illustration of the practical issues in analysing 6-DoF behavioural data:

- **Rotations need rotation maths.** Orientation lives on SO(3); relative quaternions give sign-invariant, gimbal-free
  angular displacement, and the same code works regardless of the Euler convention.
- **Real tracking data are messy.** Dropouts, jitter, duplicate timestamps and non-unit quaternions are the rule; the
  pipeline measures them, reports them, and changes nothing without logging it.
- **Time, not sample count, is the measure.** Speed, occupancy and dwell are computed with actual timestamp intervals,
  so irregular sampling and gaps do not bias them.
- **Exploration metrics are definitions, not facts.** Entropy and coverage are explicit functions of the chosen grid
  and observation time, and the dashboard makes that dependence visible instead of hiding it.
- **The unit of inference is the participant.** Frames are autocorrelated; group comparisons use one value per person,
  report effect sizes with intervals and correct for multiple comparisons.
- **Generic structure, not a specific study.** The pipeline works for any dataset with positions, optional quaternions
  and an optional condition label, and it deliberately makes no claim to replicate any published result.

## Verification status

Reported honestly, as run in the build environment:

- `pytest`: 104 passed, on Python 3.13 with current libraries (Streamlit 1.67, pandas 3.0, NumPy 2.5, SciPy 1.18,
  scikit-learn 1.9) and on Python 3.11 with the minimum versions in `requirements.txt` (Streamlit 1.40, pandas 2.0.3,
  NumPy 1.26.4, SciPy 1.11.4, scikit-learn 1.3.2, Plotly 5.20).
- All seven pages were executed with Streamlit's `AppTest` harness on both stacks, across the sidebar options
  (artifact injection, resampling, smoothing, spike repair, analysis window, grid size, DBSCAN extremes), uploaded data
  with and without quaternions and with and without a condition column.
- The server was started headlessly and every page was rendered in Chrome (including WebGL 3D plots) to check layout.
- **Not verified:** other operating systems and browsers than the Linux/Chrome used here, Python 3.12, very large
  uploads (tens of millions of samples), and any real-world dataset. Synthetic data cannot validate behavioural claims; they
  validate the *pipeline* (for instance, measured speed percentiles correlate with the generator's known cruise speed).
