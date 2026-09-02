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

# Q : (n_items, n_skills) binary item->skill map (dense Q is row-normalized on init)
# X : (n_learners, n_items) responses; 1/0 = correct/incorrect, np.nan = not seen
model = MoLA(
    Q_matrix=Q,
    n_components=4,              # number of latent archetypes
    n_iter=50,                   # max EM iterations
    tol=1e-5,                    # relative NLL convergence tolerance
    random_seed=746,             # seed for parameter initialization
    use_psuedo_likelihood=False,
)
model.fit(X)                     # accepts a dense ndarray (NaN = missing) or a scipy CSR
```

Fitted parameters: `model.mu` `(n_components, n_skills)`, `model.pi`
`(n_components, 1)`, `model.theta` `(n_items, 1)`. `model.nll_trace` is the
negative log-likelihood at each EM step and decreases monotonically.

Read-outs (all accept a dense or sparse `X`):

```python
model.get_posterior(X)                 # (n_learners, n_components) archetype responsibilities
model.pred_item_probas(X)              # (n_learners, n_items) predicted P(correct)
model.pred_item_probas(X, item_idxs=[0, 5, 9])   # ... restricted to some items
model.pred_skill_probas(X)             # (n_learners, n_skills) expected skill mastery
model.pred_attr_probas(X, skill_idxs=[1, 3], return_post=True)
model.get_pred_nll(X)                  # mean predictive NLL per response (pass a 0/1 X)
model.get_item_difficulty()           # 1 - theta, shape (n_items,); takes item_idxs too
model.get_user_ability(X)             # (n_learners, n_items) logit-scale ability
model.get_mu_slice(skill_idxs)        # archetype profiles restricted to skills
```

Diagnostics for whether components have collapsed onto each other:

```python
model.get_effective_attr_dim()        # participation ratio of mu (effective # of distinct archetypes)
model.component_redundancy_report(X)  # dict: effective_n_components, weight_min/argmin,
                                      #   closest_pair (+distance), posterior_usage
```

Covariance helpers: `get_item_cov()`, `get_skill_score_cov()`,
`convert_cov_to_corr(cov)`.

Fitting with `use_psuedo_likelihood=True` stores Beta pseudo-counts and unlocks
the posterior-sampling read-outs: `get_proficiency_cov(X)`,
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

## Tests

```bash
pytest tests/ -q
```

`tests/test_mola.py` covers the `MoLA` model itself: construction, the EM loop
(parameter ranges, seed determinism, monotonic convergence), the
stability/stopping checks, and the posterior, predictive and
psi/difficulty/redundancy read-outs, all on a tiny fit.
`tests/test_simulations.py` covers the design grid, the data-generating process,
`run_condition` / `run_design` / `summarize_results`, and the CLI wiring, all on a
tiny simulation condition so the suite finishes in a few seconds.
`tests/test_viz.py` covers the component-recovery plots (headless, Agg backend).
