"""Fit MoLA under a simulation condition and score recovery."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.mola import MoLA
from src.simulations.dgp import SimulatedDataset, generate_dataset, resample_dataset
from src.simulations.factors import (
    BASELINE,
    SEPARATION_LEVELS,
    SimulationCondition,
    ofat_design,
    replicate_seeds,
)
from src.simulations.metrics import (
    align_components,
    assignment_ari,
    brier_score,
    masked_auc,
    masked_nll,
    mu_rmse,
    pi_mae,
    pi_rmse,
    repeat_alignment_spread,
    theta_rmse,
)


class MoLAWithInit(MoLA):
    """MoLA that can start from supplied (mu, theta, pi).

    Always forces `init="random"` on the base class regardless of what's
    passed in `kwargs`: the override hooks below (`_init_mu`/`_init_theta`/
    `_init_pi`) are only consulted by `_MoLABase._init_fit`'s "random"
    branch -- MoLA's own `init="k-means"` (the base class default as of
    this simulation harness's design) bypasses those hooks entirely and
    computes its own data-informed init directly, which would silently
    ignore `mu_init`/`theta_init`/`pi_init` here. This class exists
    specifically so `src.simulations` can supply its own initialization
    (including its own from-scratch k-means scheme, `_kmeans_init` below,
    computed on `X_train`/`Q_fit` before the model is even constructed) via
    `condition.initialization`, so it always needs the "random" dispatch
    path, never MoLA's built-in one.
    """

    def __init__(self, mu_init=None, theta_init=None, pi_init=None, **kwargs):
        kwargs["init"] = "random"
        super().__init__(**kwargs)
        self.mu_init = mu_init
        self.theta_init = theta_init
        self.pi_init = pi_init

    def _init_mu(self, n_skills, n_components):
        if self.mu_init is not None:
            mu = np.asarray(self.mu_init, dtype=float)
            if mu.shape != (n_components, n_skills):
                raise ValueError(
                    f"mu_init shape {mu.shape} != {(n_components, n_skills)}"
                )
            return np.clip(mu, 1e-6, 1 - 1e-6)
        return super()._init_mu(n_skills, n_components)

    def _init_theta(self, n_items):
        if self.theta_init is not None:
            theta = np.asarray(self.theta_init, dtype=float).reshape(n_items, 1)
            return np.clip(theta, 1e-6, 1 - 1e-6)
        return super()._init_theta(n_items)

    def _init_pi(self, n_components):
        if self.pi_init is not None:
            pi = np.asarray(self.pi_init, dtype=float).reshape(n_components, 1)
            pi = np.clip(pi, 1e-8, None)
            return pi / pi.sum()
        return super()._init_pi(n_components)


def _observed_skill_scores(X: np.ndarray, Q: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(X)
    X0 = np.where(mask, X, 0.0)
    num = X0 @ Q
    den = mask.astype(float) @ Q
    scores = np.divide(num, den, out=np.full_like(num, 0.5), where=den > 0)
    return scores


def _kmeans_init(
    X: np.ndarray, Q: np.ndarray, n_components: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = _observed_skill_scores(X, Q)
    km = KMeans(
        n_clusters=n_components,
        n_init=10,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    labels = km.fit_predict(scores)
    mu = np.clip(km.cluster_centers_, 0.05, 0.95)
    _, counts = np.unique(labels, return_counts=True)
    pi = np.full(n_components, 1.0 / n_components)
    pi[: counts.size] = counts / counts.sum()
    mask = ~np.isnan(X)
    item_mean = np.nanmean(X, axis=0, keepdims=True).T
    # 0.5: maximum-entropy fallback for an item with no observed responses at all.
    item_mean = np.where(np.isnan(item_mean), 0.5, item_mean)
    theta = np.clip(item_mean, 0.15, 0.95)
    if not np.any(mask):
        theta[:] = 0.5
    return mu, theta, pi.reshape(-1, 1)


def _informed_init(
    data: SimulatedDataset, n_fit_skills: int, n_components: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = min(data.mu.shape[1], n_fit_skills)
    mu = np.full((n_components, n_fit_skills), 0.5)
    m = min(n_components, data.mu.shape[0])
    mu[:m, :k] = data.mu[:m, :k]
    mu += rng.normal(0.0, 0.05, size=mu.shape)
    pi = np.full((n_components, 1), 1.0 / n_components)
    pi[: data.pi.size, 0] = data.pi.ravel()[:n_components]
    pi = pi / pi.sum()
    theta = np.clip(data.theta + rng.normal(0.0, 0.05, size=data.theta.shape), 0.1, 0.9)
    return np.clip(mu, 0.05, 0.95), theta, pi


def split_holdout(
    X: np.ndarray, holdout_frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Hide a fraction of observed responses for predictive evaluation."""
    observed = np.flatnonzero(~np.isnan(X))
    n_hold = int(np.floor(holdout_frac * observed.size))
    hold_ix = rng.choice(observed, size=max(n_hold, 1), replace=False)
    X_train = X.copy()
    hold_mask = np.zeros(X.shape, dtype=bool)
    hold_mask.ravel()[hold_ix] = True
    X_train[hold_mask] = np.nan
    return X_train, hold_mask


def _initial_params(
    data: SimulatedDataset, X_train: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    init = data.condition.initialization
    n_components = data.condition.resolved_n_components_fit()
    n_skills = data.Q_fit.shape[1]
    if init == "informed" and n_components != data.condition.n_archetypes:
        raise ValueError(
            f"initialization='informed' assumes the correct number of archetypes "
            f"(it seeds from the true mu/pi/theta), but this condition fits "
            f"n_components={n_components} against a true n_archetypes="
            f"{data.condition.n_archetypes}. Use initialization='random' or "
            f"'k-means' when n_components_fit misspecifies the number of "
            f"archetypes -- unlike 'informed', 'k-means' doesn't assume the "
            f"correct count (it just clusters into however many components "
            f"it's asked for)."
        )
    if init == "random":
        return None, None, None
    if init == "k-means":
        return _kmeans_init(X_train, data.Q_fit, n_components, rng)
    return _informed_init(data, n_skills, n_components, rng)


def fit_mola(
    data: SimulatedDataset, X_train: np.ndarray, rng: np.random.Generator
) -> tuple[MoLAWithInit, float]:
    """Fit MoLA, returning `(model, init_time_sec)`.

    `init_time_sec` is the wall-clock cost of `_initial_params` alone (e.g.
    k-means clustering under `initialization="k-means"`), timed separately
    from the EM loop that follows -- the two can differ by an order of
    magnitude and don't scale with problem size the same way (k-means's
    own `n_init`-restart convergence time is noisy and largely independent
    of `n_components`, unlike EM's per-iteration cost), so a caller wanting
    a clean per-EM-iteration reading needs `init_time_sec` isolated out of
    the total, not blended into it.
    """
    t0 = time.perf_counter()
    mu_init, theta_init, pi_init = _initial_params(data, X_train, rng)
    init_time_sec = time.perf_counter() - t0

    params = {
        "Q": data.Q_fit,
        "n_components": data.condition.resolved_n_components_fit(),
        "mu_prior": (2, 2),
        "theta_prior": (2, 2),
        "pi_prior": 2,
        "tol": 1e-5,
        "max_iter": data.condition.n_iter,
        "pseudo_likelihood": False,
        "random_state": int(rng.integers(0, 2**31 - 1)),
    }
    params.update(data.condition.fit_kwargs)
    model = MoLAWithInit(
        mu_init=mu_init, theta_init=theta_init, pi_init=pi_init, **params
    )
    model.fit(X_train)
    return model, init_time_sec


@dataclass
class SimulationResult:
    metrics: dict
    model: MoLAWithInit
    data: SimulatedDataset


def _score_fit(
    data: SimulatedDataset,
    model: MoLAWithInit,
    X_train: np.ndarray,
    hold_mask: np.ndarray,
    seed: int,
    train_time: float,
    init_time_sec: float,
) -> dict:
    """Recovery + predictive metrics for one fit of `model` against `data`.

    `train_time` is the total wall-clock cost (init + EM loop, what
    `run_condition` et al. measure around `fit_mola`); `init_time_sec` is
    `fit_mola`'s own split-out initialization-only timing (see its
    docstring). `fit_time_sec` -- the EM loop alone -- is the exact
    remainder, not an estimate, since nothing else happens between the two
    timed segments.
    """
    k = min(data.mu.shape[1], model.mu.shape[1])
    perm = align_components(data.mu[:, :k], model.mu[:, :k])
    post = model.predict_proba(X_train)

    y_true = data.X[hold_mask]
    y_prob = model.predict_item_proba(X_train)[hold_mask]

    return {
        **data.condition.to_dict(),
        "seed": seed,
        "mu_rmse": mu_rmse(data.mu, model.mu),
        "pi_mae": pi_mae(data.pi, model.pi, perm),
        "pi_rmse": pi_rmse(data.pi, model.pi, perm),
        "theta_rmse": theta_rmse(data.theta, model.theta),
        "assignment_ari": assignment_ari(data.z, post, perm),
        "holdout_auc": masked_auc(y_true, y_prob),
        "holdout_brier": brier_score(y_true, y_prob),
        "holdout_nll": masked_nll(y_true, y_prob),
        "n_em_iters": len(model.nll_trace),
        "final_nll": float(model.nll_trace[-1]) if model.nll_trace else float("nan"),
        "train_time_sec": train_time,
        "init_time_sec": init_time_sec,
        "fit_time_sec": train_time - init_time_sec,
        "n_observations": int((~np.isnan(data.X)).sum()),
    }


def run_condition(
    condition: SimulationCondition,
    seed: int | None = None,
    return_fit: bool = False,
) -> dict | SimulationResult:
    seed = condition.seed if seed is None else seed
    rng = np.random.default_rng(seed)
    data = generate_dataset(condition, seed=seed)
    X_train, hold_mask = split_holdout(data.X, condition.holdout_frac, rng)

    t0 = time.perf_counter()
    model, init_time_sec = fit_mola(data, X_train, rng)
    train_time = time.perf_counter() - t0

    metrics = _score_fit(data, model, X_train, hold_mask, seed, train_time, init_time_sec)
    if return_fit:
        return SimulationResult(metrics=metrics, model=model, data=data)
    return metrics


def run_init_repeats(
    condition: SimulationCondition,
    data_seed: int,
    n_repeats: int,
    base_init_seed: int = 0,
) -> list[SimulationResult]:
    """Fit ONE simulated dataset `n_repeats` times, varying only the EM
    initialization draw -- isolates robustness to initialization from sampling
    noise (see notebooks/simulation-study.ipynb, "Estimation Stability").

    `condition.initialization` should be "random": k-means/informed inits
    derive from the data/true parameters, so repeating them mostly re-measures
    a different RNG's seed, not MoLA's own sensitivity to where EM starts. The
    train/holdout split is also fixed once, so the only thing that changes
    between repeats is where EM starts.
    """
    data = generate_dataset(condition, seed=data_seed)
    split_rng = np.random.default_rng(data_seed)
    X_train, hold_mask = split_holdout(data.X, condition.holdout_frac, split_rng)

    results = []
    for i in range(n_repeats):
        init_seed = base_init_seed + i
        rng = np.random.default_rng(init_seed)
        t0 = time.perf_counter()
        model, init_time_sec = fit_mola(data, X_train, rng)
        train_time = time.perf_counter() - t0
        metrics = _score_fit(data, model, X_train, hold_mask, init_seed, train_time, init_time_sec)
        results.append(SimulationResult(metrics=metrics, model=model, data=data))
    return results


def run_sample_repeats(
    condition: SimulationCondition,
    population_seed: int,
    n_repeats: int,
    base_sample_seed: int = 0,
) -> list[SimulationResult]:
    """Fit MoLA on `n_repeats` independent samples drawn from ONE fixed
    data-generating population -- isolates sampling variability (a new cohort
    of learners, new realized responses) from EM initialization sensitivity
    (see notebooks/simulation-study.ipynb, "Estimation Stability").

    Unlike `run_init_repeats` (one dataset, varying EM's initialization draw),
    this generates the population once (`generate_dataset`) and re-draws only
    the sample each repeat (`resample_dataset`), refitting via
    `condition.initialization` every time -- meant for `initialization=
    "k-means"`, where EM's own initialization draw is no longer a meaningful
    source of variability on its own (see the notebook's initialization-scheme
    comparison), so what's left to characterize is genuine sampling
    variability at this population and sample size.
    """
    population = generate_dataset(condition, seed=population_seed)
    results = []
    for i in range(n_repeats):
        sample_seed = base_sample_seed + i
        data = resample_dataset(population, seed=sample_seed)
        split_rng = np.random.default_rng(sample_seed)
        X_train, hold_mask = split_holdout(data.X, condition.holdout_frac, split_rng)
        init_rng = np.random.default_rng(sample_seed)
        t0 = time.perf_counter()
        model, init_time_sec = fit_mola(data, X_train, init_rng)
        train_time = time.perf_counter() - t0
        metrics = _score_fit(data, model, X_train, hold_mask, sample_seed, train_time, init_time_sec)
        results.append(SimulationResult(metrics=metrics, model=model, data=data))
    return results


def run_m_sweep(
    condition: SimulationCondition,
    n_components_grid: list[int],
    seed: int = 0,
) -> list[SimulationResult]:
    """Fit the SAME simulated data with a range of `n_components_fit` values,
    to see how recovery/predictive quality and archetype redundancy respond to
    misspecifying the number of archetypes (see notebooks/simulation-study.ipynb,
    "Sensitivity to the Number of Archetypes").

    `generate_dataset` and `split_holdout` don't depend on `n_components_fit`,
    so passing the same `seed` for every grid point holds the true data and the
    train/holdout split fixed -- only the fitted `n_components` changes. Uses
    whatever `condition.initialization` already is at every grid point (default
    "k-means", per `SimulationCondition`) -- only `initialization="informed"`
    is incompatible with a misspecified grid point (see the validation guard in
    `_initial_params`); "k-means" clusters into however many components it's
    asked for and doesn't need this forced to "random" the way it used to.
    """
    results = []
    for m in n_components_grid:
        cond = replace(condition, n_components_fit=m)
        results.append(run_condition(cond, seed=seed, return_fit=True))
    return results


def run_m_sweep_with_repeats(
    condition: SimulationCondition,
    n_components_grid: list[int],
    data_seed: int = 0,
    n_repeats: int = 5,
    base_init_seed: int = 0,
) -> list[SimulationResult]:
    """`run_m_sweep`, but repeating `run_init_repeats` at every grid point
    instead of a single fit -- needed to measure parameter stability across
    EM initializations *as a function of* `n_components_fit` (see
    notebooks/simulation-study.ipynb, "Sensitivity to the Number of
    Archetypes", the M-selection-criteria evaluation).

    Whatever `condition.initialization` is set to applies at every grid
    point -- pass `initialization="random"` explicitly when the point is to
    measure EM-init spread itself (k-means would collapse that spread to
    ~0 at every `M`, which answers a different question, not this one; see
    `run_init_repeats`'s own docstring).
    """
    results = []
    for m in n_components_grid:
        cond = replace(condition, n_components_fit=m)
        results.extend(run_init_repeats(
            cond, data_seed=data_seed, n_repeats=n_repeats, base_init_seed=base_init_seed
        ))
    return results


def default_m_grid(true_m: int, n_points: int = 7) -> list[int]:
    """A grid of candidate `n_components_fit` values spanning under- and
    over-specification relative to `true_m` -- `[true_m // 2, 2 * true_m]`,
    comfortably past the true count in both directions (needed by the
    saturation-based separation/mass criteria in `pick_m_by_criterion`,
    which read a value at the largest `m` tested).
    """
    lo, hi = max(1, true_m // 2), max(true_m * 2, true_m + 1)
    grid = sorted(set(int(round(v)) for v in np.linspace(lo, hi, n_points)))
    return grid


def summarize_m_sweep(
    kmeans_results: list[SimulationResult],
    random_repeat_results: list[SimulationResult],
) -> pd.DataFrame:
    """One row per `n_components_fit` grid point, combining two different
    sweeps for two different purposes (see `run_m_selection_evaluation`):

    - `kmeans_results` (one fit per grid point, `initialization="k-means"`,
      from `run_m_sweep`): predictive (`holdout_auc`, `holdout_nll`) and
      redundancy (`effective_n_archetypes`, `effective_n_components`,
      `weight_min`, `closest_pair_distance`, from `MoLA.redundancy_report`)
      read directly off that one fit -- k-means is close enough to
      deterministic here (see the notebook's initialization-scheme
      comparison, Section 2a) that repeats aren't needed for these.
    - `random_repeat_results` (`n_repeats` fits per grid point,
      `initialization="random"`, from `run_m_sweep_with_repeats`):
      `mu_spread` = `repeat_alignment_spread` of that grid point's repeated
      `mu` estimates -- this one NEEDS EM-init variance to have a signal at
      all, which k-means wouldn't provide (see `run_m_sweep_with_repeats`'s
      docstring).

    Both sweeps must share the same `n_components_fit` grid (not checked).
    """
    grid = sorted({r.metrics["n_components_fit"] for r in kmeans_results})
    kmeans_by_m = {r.metrics["n_components_fit"]: r for r in kmeans_results}

    random_by_m: dict[int, list[SimulationResult]] = {}
    for r in random_repeat_results:
        random_by_m.setdefault(r.metrics["n_components_fit"], []).append(r)

    rows = []
    for m in grid:
        kr = kmeans_by_m[m]
        report = kr.model.redundancy_report()
        repeats = random_by_m.get(m, [])
        rows.append({
            "n_components_fit": m,
            "holdout_auc": kr.metrics["holdout_auc"],
            "holdout_nll": kr.metrics["holdout_nll"],
            "effective_n_archetypes": kr.model.effective_n_archetypes(),
            "effective_n_components": report["effective_n_components"],
            "weight_min": report["weight_min"],
            "closest_pair_distance": report["closest_pair_distance"],
            "mu_spread": (
                repeat_alignment_spread([r.model.mu for r in repeats])
                if len(repeats) >= 2 else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def pick_m_by_criterion(
    m_summary: pd.DataFrame,
    auc_eps: float = 0.002,
    drop_frac: float = 0.5,
) -> dict[str, int]:
    """Seven independent M_hat rules read off `summarize_m_sweep`'s output,
    one per candidate criterion -- deliberately NOT combined into a single
    answer (see notebooks/simulation-study.ipynb, "Sensitivity to the
    Number of Archetypes": the point of this evaluation is to characterize
    each criterion's own accuracy at recovering the true M, not to ship a
    combined selector).

    - `predictive_auc`: smallest `m` within `auc_eps` of the best mean
      holdout AUC (a slack-argmax, since AUC often plateaus rather than
      peaking sharply).
    - `predictive_nll`: argmin mean holdout NLL.
    - `stability`: argmin of `mu_spread` (EM-init spread). NOTE: `mu_spread`
      is trivially near 0 at the smallest `m` in the grid (nothing to
      disagree about with only 1-2 components), so this rule is
      structurally biased toward `min(grid)`. That bias is not a wiring
      bug -- if `stability` alone reliably lands on `min(grid)` regardless
      of the true M, that IS the finding this evaluation exists to surface.
    - `separation_saturation` / `mass_saturation`: `effective_n_archetypes`/
      `effective_n_components` are specifically documented
      (`MoLA.effective_n_archetypes`) to saturate near the true count even
      once over-specified, so a single read at the largest `m` tested
      (rounded) is a defensible rule without needing elbow-detection over
      the whole curve.
    - `separation_threshold` / `mass_threshold`: walking the grid ascending,
      the last `m` before `closest_pair_distance`/`weight_min` first drops
      below `drop_frac` of its own value at the smallest `m` -- relative,
      not absolute, since an absolute cutoff would be confounded with
      `archetype_separation` itself (a "low"-separation condition starts
      with smaller true pairwise distances even when correctly specified).
      Returns `max(grid)` if it never drops that much.
    """
    m_summary = m_summary.sort_values("n_components_fit").reset_index(drop=True)
    grid = m_summary["n_components_fit"].tolist()

    def _threshold_walk(col: str) -> int:
        values = m_summary[col].tolist()
        base = values[0]
        if not np.isfinite(base) or base <= 0:
            return grid[-1]
        for i, v in enumerate(values):
            if v < drop_frac * base:
                return grid[i - 1] if i > 0 else grid[0]
        return grid[-1]

    best_auc = m_summary["holdout_auc"].max()
    predictive_auc = next(
        m for m, auc in zip(grid, m_summary["holdout_auc"]) if auc >= best_auc - auc_eps
    )

    return {
        "predictive_auc": int(predictive_auc),
        "predictive_nll": int(grid[int(m_summary["holdout_nll"].idxmin())]),
        "stability": int(grid[int(m_summary["mu_spread"].idxmin())]),
        "separation_saturation": int(round(m_summary["effective_n_archetypes"].iloc[-1])),
        "separation_threshold": int(_threshold_walk("closest_pair_distance")),
        "mass_saturation": int(round(m_summary["effective_n_components"].iloc[-1])),
        "mass_threshold": int(_threshold_walk("weight_min")),
    }


def run_m_selection_evaluation(
    n_archetypes_grid: Sequence[int] = (2, 4, 8, 16),
    separation_grid: Sequence[str] = SEPARATION_LEVELS,
    seeds: Sequence[int] = (0, 1, 2),
    n_repeats: int = 5,
    n_grid_points: int = 7,
) -> pd.DataFrame:
    """Evaluate whether each of `pick_m_by_criterion`'s seven M_hat rules
    recovers the true `n_archetypes`, across a grid of true `n_archetypes`
    (`n_archetypes_grid`) x `archetype_separation` (`separation_grid`) x a
    few data seeds (see notebooks/simulation-study.ipynb, "Sensitivity to
    the Number of Archetypes").

    Each condition is scaled so a larger true M isn't data-starved:
    `n_skills = max(20, 2 * true_m)` (`archetype_means` in dgp.py needs
    `n_skills >= n_archetypes` to build one distinguishing skill group per
    archetype) and `n_learners = max(2_000, 300 * true_m)` (keeps a
    comparable expected learner count per archetype as `true_m` grows).

    Returns a long-form DataFrame, one row per (n_archetypes,
    archetype_separation, seed, criterion): `m_hat`, `error` (`m_hat -
    true_m`), `exact_match`. Compute cost scales as roughly
    `sum(len(default_m_grid(m)) for m in n_archetypes_grid) * (1 +
    n_repeats) * len(separation_grid) * len(seeds)` MoLA fits -- start with
    small grids/few seeds to smoke-test before widening (see the notebook).
    """
    rows = []
    for true_m in n_archetypes_grid:
        grid = default_m_grid(true_m, n_points=n_grid_points)
        n_skills = max(20, 2 * true_m)
        n_learners = max(2_000, 300 * true_m)
        for sep in separation_grid:
            base_condition = SimulationCondition(
                n_archetypes=true_m,
                n_skills=n_skills,
                n_learners=n_learners,
                archetype_separation=sep,
            )
            for seed in seeds:
                kmeans_results = run_m_sweep(
                    replace(base_condition, initialization="k-means"), grid, seed=seed
                )
                random_repeat_results = run_m_sweep_with_repeats(
                    replace(base_condition, initialization="random"),
                    grid, data_seed=seed, n_repeats=n_repeats,
                )
                m_summary = summarize_m_sweep(kmeans_results, random_repeat_results)
                m_hats = pick_m_by_criterion(m_summary)
                for criterion, m_hat in m_hats.items():
                    rows.append({
                        "n_archetypes": true_m,
                        "archetype_separation": sep,
                        "seed": seed,
                        "criterion": criterion,
                        "m_hat": m_hat,
                        "error": m_hat - true_m,
                        "exact_match": m_hat == true_m,
                    })
    return pd.DataFrame(rows)


def run_design(
    conditions: list[SimulationCondition] | None = None,
    n_replications: int = 5,
    base_seed: int = 0,
) -> pd.DataFrame:
    """Run each condition over Monte Carlo replications and return a results table."""
    if conditions is None:
        conditions = ofat_design()
    rows: list[dict] = []
    seeds = replicate_seeds(base_seed, n_replications)
    for condition in conditions:
        for seed in seeds:
            rows.append(run_condition(condition, seed=seed))
    return pd.DataFrame(rows)


def run_nested_init_repeats(
    condition: SimulationCondition,
    n_datasets: int = 3,
    n_fits: int = 3,
    base_seed: int = 0,
) -> list[tuple[int, SimulationResult]]:
    """Nested replication for ONE fixed condition: `n_datasets` independent
    datasets, each refit `n_fits` times varying only the EM init/fit seed
    (`run_init_repeats`'s own inner loop, reused as-is) -- the same nested
    structure as `run_nested_ofat_design`, but for a single condition
    rather than an OFAT sweep, and returning full `SimulationResult`s (so
    `.model.mu`/`.model.theta`/`.model.pi` are accessible) instead of a
    flat metrics DataFrame.

    Returns a list of `(dataset_idx, SimulationResult)` pairs, length
    `n_datasets * n_fits` -- group by `dataset_idx` to decompose variance
    (`within_between_variance`) on any per-repeat quantity, e.g. a
    recovered parameter array's own entries, the way
    `run_nested_ofat_design`'s caller does for scalar metrics.
    """
    rows: list[tuple[int, SimulationResult]] = []
    for dataset_idx in range(n_datasets):
        data_seed = base_seed + 10_007 * dataset_idx
        init_seed_base = base_seed + 10_009 * dataset_idx
        results = run_init_repeats(
            condition, data_seed=data_seed, n_repeats=n_fits, base_init_seed=init_seed_base,
        )
        for result in results:
            rows.append((dataset_idx, result))
    return rows


def run_nested_ofat_design(
    factors: Sequence[str] | None = None,
    baseline: SimulationCondition = BASELINE,
    n_datasets: int = 3,
    n_fits: int = 3,
    base_seed: int = 0,
) -> pd.DataFrame:
    """OFAT sweep with a nested dataset x fit replication structure, for
    decomposing each metric's total variance into within-dataset
    (estimation) and between-dataset (sampling) components -- see
    notebooks/simulation-study.ipynb, "Parameter Recovery, Stability, and
    Scalability", and `src.simulations.metrics.within_between_variance`.

    For each OFAT condition, draws `n_datasets` independent datasets
    (reusing `run_init_repeats`'s `data_seed`), and for each dataset
    fits `n_fits` times varying only the EM init/fit seed (exactly
    `run_init_repeats`'s own inner loop -- this function doesn't
    duplicate any data-generation/fitting logic, it just loops that
    existing primitive over an OFAT condition grid and a range of
    dataset seeds). `condition.initialization` stays whatever the OFAT
    design uses (k-means by default, matching the rest of the OFAT
    sweep elsewhere in this notebook) -- see `run_init_repeats`'s own
    docstring for why that means the within-dataset component is
    expected to come out near zero: k-means derives its start from the
    data itself, so repeating the fit on the same dataset mostly
    re-measures a different RNG seed, not genuine sensitivity to where
    EM starts.

    Returns one row per (condition, dataset_idx, fit_idx) with every
    `_score_fit` metric column (and every `SimulationCondition` field,
    via that dict's own `**data.condition.to_dict()`) plus `dataset_idx`
    and `fit_idx` -- the same shape `run_design` produces, with these
    two extra columns, so existing OFAT-panel grouping logic
    (`results[factor] == baseline_dict[factor]`) works unchanged; only
    a row that wants the variance decomposition needs to additionally
    group by `dataset_idx`.
    """
    conditions = ofat_design(baseline=baseline, factors=factors)
    rows: list[dict] = []
    for condition in conditions:
        for dataset_idx in range(n_datasets):
            data_seed = base_seed + 10_007 * dataset_idx
            init_seed_base = base_seed + 10_009 * dataset_idx
            results = run_init_repeats(
                condition, data_seed=data_seed, n_repeats=n_fits, base_init_seed=init_seed_base,
            )
            for fit_idx, result in enumerate(results):
                row = dict(result.metrics)
                row["dataset_idx"] = dataset_idx
                row["fit_idx"] = fit_idx
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_results(
    results: pd.DataFrame,
    by: str | list[str] = "condition_id",
    metrics: tuple[str, ...] = (
        "mu_rmse",
        "pi_mae",
        "pi_rmse",
        "theta_rmse",
        "assignment_ari",
        "holdout_auc",
        "holdout_brier",
        "holdout_nll",
        "train_time_sec",
    ),
) -> pd.DataFrame:
    grouped = results.groupby(by, as_index=False)
    frames = []
    for metric in metrics:
        if metric not in results.columns:
            continue
        stats = grouped[metric].agg(mean="mean", std="std", n="count")
        stats = stats.rename(columns={c: f"{metric}_{c}" for c in ("mean", "std", "n")})
        frames.append(stats)
    out = frames[0]
    for extra in frames[1:]:
        out = out.merge(extra, on=by)
    return out
