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

`tests/test_simulations.py` covers the design grid, the data-generating process,
`run_condition` / `run_design` / `summarize_results`, and the CLI wiring, all on a
tiny simulation condition so the suite finishes in a few seconds.
`tests/test_viz.py` covers the component-recovery plots (headless, Agg backend).
