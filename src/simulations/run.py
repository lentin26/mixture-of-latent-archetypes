"""Fit MoLA under a simulation condition and score recovery."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.mola import MoLA
from src.simulations.dgp import SimulatedDataset, generate_dataset
from src.simulations.factors import (
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
    theta_rmse,
)


class MoLAWithInit(MoLA):
    """MoLA that can start from supplied (mu, theta, pi)."""

    def __init__(self, mu_init=None, theta_init=None, pi_init=None, **kwargs):
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
    item_mean = np.where(np.isnan(item_mean), 0.7, item_mean)
    theta = np.clip(item_mean, 0.15, 0.95)
    if not np.any(mask):
        theta[:] = 0.7
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
    n_components = data.condition.n_archetypes
    n_skills = data.Q_fit.shape[1]
    if init == "random":
        return None, None, None
    if init == "k-means":
        return _kmeans_init(X_train, data.Q_fit, n_components, rng)
    return _informed_init(data, n_skills, n_components, rng)


def fit_mola(
    data: SimulatedDataset, X_train: np.ndarray, rng: np.random.Generator
) -> MoLAWithInit:
    mu_init, theta_init, pi_init = _initial_params(data, X_train, rng)
    params = {
        "Q_matrix": data.Q_fit,
        "n_components": data.condition.n_archetypes,
        "mu_smooth": [2, 2],
        "theta_smooth": [2, 2],
        "pi_smooth": 2,
        "tol": 1e-5,
        "n_iter": data.condition.n_iter,
        "use_psuedo_likelihood": False,
        "random_seed": int(rng.integers(0, 2**31 - 1)),
    }
    params.update(data.condition.fit_kwargs)
    model = MoLAWithInit(
        mu_init=mu_init, theta_init=theta_init, pi_init=pi_init, **params
    )
    model.fit(X_train)
    return model


@dataclass
class SimulationResult:
    metrics: dict
    model: MoLAWithInit
    data: SimulatedDataset


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
    model = fit_mola(data, X_train, rng)
    train_time = time.perf_counter() - t0

    k = min(data.mu.shape[1], model.mu.shape[1])
    perm = align_components(data.mu[:, :k], model.mu[:, :k])
    post = model.get_posterior(model.convert_dense_to_sparse(X_train))

    y_true = data.X[hold_mask]
    y_prob = model.pred_item_probas(X_train)[hold_mask]

    metrics = {
        **condition.to_dict(),
        "seed": seed,
        "mu_rmse": mu_rmse(data.mu, model.mu),
        "pi_mae": pi_mae(data.pi, model.pi, perm),
        "theta_rmse": theta_rmse(data.theta, model.theta),
        "assignment_ari": assignment_ari(data.z, post, perm),
        "holdout_auc": masked_auc(y_true, y_prob),
        "holdout_brier": brier_score(y_true, y_prob),
        "holdout_nll": masked_nll(y_true, y_prob),
        "n_em_iters": len(model.nll_trace),
        "final_nll": float(model.nll_trace[-1]) if model.nll_trace else float("nan"),
        "train_time_sec": train_time,
        "n_observations": int((~np.isnan(data.X)).sum()),
    }
    if return_fit:
        return SimulationResult(metrics=metrics, model=model, data=data)
    return metrics


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


def summarize_results(
    results: pd.DataFrame,
    by: str | list[str] = "condition_id",
    metrics: tuple[str, ...] = (
        "mu_rmse",
        "pi_mae",
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
