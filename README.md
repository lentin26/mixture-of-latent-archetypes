# mixture-of-latent-archetypes

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

All code assumes the repo root is on `sys.path` (imports look like `from src.mola import MoLA`),
so run commands and notebooks from the root, or `pip install -e .` once a package config is added.

To use the venv as a Jupyter kernel:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name mola --display-name "Python (mola)"
```

## Using MoLA directly

`MoLA` (`src/mola`) is a mixture of latent archetypes fit with EM. Each archetype
`m` carries a skill-mastery profile `mu[m]` (`P(skill mastered | archetype)`),
the mixture weights `pi` say how common each archetype is, and item easiness
`theta` relates to difficulty by `theta = (1 - d)`.

```python
import numpy as np
from src.mola import MoLA

# Q : (n_items, n_skills) binary item->skill map (the Q-matrix; row-normalized on init)
# X : (n_learners, n_items) responses; 1/0 = correct/incorrect, np.nan = not seen
model = MoLA(
    Q,
    n_components=4,              # number of latent archetypes
    max_iter=50,                 # max EM iterations
    tol=1e-5,                    # relative NLL convergence tolerance
    pseudo_likelihood=False,     # pseudo-likelihood M-step + Beta pseudo-counts
    random_state=746,            # seed for parameter initialization
)
model.fit(X)                     # dense ndarray (NaN = missing) or a scipy sparse matrix
```

Fitted parameters: `model.mu` `(n_components, n_skills)`, `model.pi`
`(n_components, 1)`, `model.theta` `(n_items, 1)`. `model.nll_trace` is the
negative log-likelihood at each EM step and decreases monotonically.

Read-outs (every method takes a dense **or** sparse `X` — no manual conversion):

```python
model.predict_proba(X)                    # (n_learners, n_components) archetype responsibilities
model.predict_item_proba(X)               # (n_learners, n_items) predicted P(correct)
model.predict_item_proba(X, item_idxs=[0, 5, 9])   # ... restricted to some items
model.predict_skill_proba(X, skill_idxs=None)       # (n_learners, n_skills) expected skill mastery
model.score(X)                            # mean predictive NLL per observed response
model.get_item_difficulty()              # 1 - theta, shape (n_items,); takes item_idxs too
model.get_user_ability(X)                # (n_learners, n_items) logit-scale ability
model.get_mu_slice(skill_idxs)           # archetype profiles restricted to skills
```

Diagnostics for whether archetypes have collapsed onto each other:

```python
model.effective_n_archetypes()   # participation ratio of mu (effective # of distinct archetypes)
model.redundancy_report(X)       # dict: effective_n_archetypes, effective_n_components,
                                 #   weight_min/argmin, closest_pair (+distance), posterior_usage
```

Covariance helpers: `get_item_cov()`, `get_skill_score_cov()`,
`convert_cov_to_corr(cov)`.

Fitting with `pseudo_likelihood=True` stores Beta pseudo-counts and unlocks the
posterior-sampling read-outs: `get_proficiency_cov(X)`,
`sample_user_skill_posterior(x, mask, n_samples)`, `get_skill_posterior(x, skill_idx)`,
`get_prior_exp_skill_prof()`.

## Running the simulation study

The `src/simulations` package fits MoLA across simulation conditions and scores
parameter and predictive recovery. Run it as a module from the repo root:

```bash
# fast smoke test: baseline condition, one replication
python -m src.simulations --design baseline --replications 1 --summarize

# one-factor-at-a-time sweep, 5 replications.
# Writes per-run rows to results/sim-results.csv and the aggregated
# mean/std table to results/sim-results_summary.csv.
python -m src.simulations --design ofat --replications 5 --out results/sim-results.csv

# restrict the sweep to specific factors and cap the number of conditions
python -m src.simulations --design ofat --factors n_archetypes n_learners --limit 6
```

See `python -m src.simulations --help` for all options.

### Visualizing component recovery

`src/simulations/viz.py` plots the assumed (data-generating) archetype profiles
against the recovered ones — proficiency `P(skill mastered)` on the y-axis, skill
index on the x-axis — after the same Hungarian alignment the recovery metrics use.

```python
from src.simulations import run_condition
from src.simulations.factors import SimulationCondition
from src.simulations.viz import plot_component_recovery

result = run_condition(
    SimulationCondition(n_archetypes=4, initialization="k-means"),
    seed=0,
    return_fit=True,          # needed: exposes result.model and result.data
)

plot_component_recovery(result, save="figures/component_recovery.pdf")   # one panel per component
plot_component_recovery(result, layout="overlay")                       # all on one axes
```

Also available: `plot_components(mu, ...)` for a single set of profiles, and
`plot_recovery_scatter(result)` for a recovered-vs-assumed diagonal plot.

### Five-experiment simulation study

`notebooks/simulation-study.ipynb` runs five experiments, each isolating one
claim about MoLA's archetype recovery (each needs a different axis held fixed
and a different axis varied, which is why they're separate rather than one
combined sweep):

1. **Archetype Recovery** — does MoLA recover the generating structure at
   all? `run_condition(..., return_fit=True)` + `plot_component_recovery` /
   `plot_recovery_scatter`.
2. **Estimation Stability** — does the solution depend on arbitrary
   initialization? `run_init_repeats` fixes one simulated dataset and its
   train/holdout split, then re-fits it many times varying only the EM
   initialization draw; `plot_component_recovery_distribution` shows the
   spread of recovered archetypes across those repeats.
3. **Robustness to Data Sparsity** — does recovery degrade gracefully as
   users (`n_learners`) or responses-per-user (`responses_per_learner`)
   shrink? `ofat_design` + `run_design` + `plot_ofat_sensitivity`.
4. **Sensitivity to the Number of Archetypes** — what happens when the fitted
   `n_components` doesn't match the true `n_archetypes`? `run_m_sweep` fits
   the same data across a grid of `n_components_fit` values;
   `plot_m_sensitivity` also reports `MoLA.effective_n_archetypes()` per fit,
   which should saturate near the true count even when over-specified.
5. **Computational Scalability** — how does fit time grow with problem size?
   The same `ofat_design` + `run_design` machinery as (3), plotted with
   `plot_ofat_sensitivity(..., metric="train_time_sec", log_x=True, log_y=True)`.

`plot_ofat_sensitivity` takes the **per-run** table from `run_design`
directly, not `summarize_results`'s aggregated output (which drops the
per-condition factor columns the sensitivity plot needs) — it computes the
mean/std per factor level itself. Use `log_x` and `log_y` together (not
`log_x` alone) when checking a scaling law like (5): on log-log axes a power
law `y ~ x**p` is a straight line of slope `p`, so linear cost reads as a
straight line. Plotted with a linear y-axis instead, genuinely linear cost
bends upward and can look quadratic or exponential — a straight line under
log-x/linear-y actually corresponds to *logarithmic* growth, not linear.

### Publication-quality figures

`src/style.py` is a standalone module (no dependency on `src.mola` or
`src.simulations`) for figures headed into a LaTeX paper: print-legible
fonts/line weights, vector-safe embedded text (`pdf.fonttype=42`, so text
stays real/searchable rather than a bitmap), and no dependency on a local
LaTeX toolchain (`text.usetex` stays `False`; a Times-like serif font plus
Matplotlib's `dejavuserif` mathtext set gets visually close to real LaTeX
text without it). Deliberately avoids Matplotlib's `"cmr10"`/`"Computer
Modern Roman"` font names in the serif fallback list — those are internal
mathtext-only assets, and using them as the plain-text body font drops
ordinary glyphs like the literal underscore (`"n_learners"` renders as
`"n˙learners"`).

```python
from src.style import apply, figsize

apply()  # every figure created for the rest of the session picks this up
fig = plot_component_recovery(result, figsize=figsize("acm-sigconf-text", aspect=0.32))
fig.savefig("figures/component_recovery.pdf")
```

`figsize(width, fraction=1.0, aspect=0.68)` sizes a figure in inches to
exactly fill a fraction of a LaTeX column/text width, so `\includegraphics`
doesn't rescale it (rescaling blurs text and distorts line weights relative
to the surrounding body text). `width` is a raw point value or a key into
`COLUMN_WIDTHS_PT` (a few common document classes, defaulting to
`acmart`'s `sigconf` layout); get the exact number for your own document
with `\typeout{\the\columnwidth}` (or `\the\textwidth`) next to a figure and
reading it from the compiled `.log`. Use `publication_style()` as a context
manager instead of `apply()` to scope the styling to only some figures in a
session, and `strip_titles(fig)` to drop the descriptive `fig.suptitle(...)`
some `viz.py` plots add (a submission figure's caption lives in the LaTeX
source, not baked into the image).

`notebooks/simulation-study.ipynb` applies this to all five experiment
figures.

`plot_ofat_sensitivity` and `plot_m_sensitivity` also never show a raw
`snake_case` column name: `src.simulations.viz.METRIC_LABELS` maps known
metric/factor names to readable labels (e.g. `mu_rmse` -> "Archetype
Recovery RMSE", `n_learners` -> "Number of Learners"). Pass
`labels={"mu_rmse": "..."}` to override or extend it for one figure;
anything not listed falls back to `src.style.humanize_label`'s generic
`snake_case` -> `Title Case` conversion (keeping a few statistical acronyms
like AUC/RMSE upper-case) rather than showing the raw name.

## Tests

```bash
pytest tests/ -q
```

`tests/test_mola.py` covers the `MoLA` model itself: construction, the EM loop
(parameter ranges, seed determinism, monotonic convergence), the
stability/stopping checks, and the posterior, predictive and
psi/difficulty/redundancy read-outs, all on a tiny fit.
`tests/test_simulations.py` covers the design grid, the data-generating process,
`run_condition` / `run_design` / `summarize_results` / `run_init_repeats` /
`run_m_sweep`, the `n_components_fit` misspecification validation, and the CLI
wiring, all on a tiny simulation condition so the suite finishes in a few
seconds.
`tests/test_viz.py` covers the component-recovery plots and the five-experiment
study's plots (`plot_component_recovery_distribution`, `plot_ofat_sensitivity`,
`plot_m_sensitivity`), headless (Agg backend).
`tests/test_style.py` covers `figsize`, `publication_style` / `apply`, and
`strip_titles`.
