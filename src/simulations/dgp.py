"""Simulate learner–item responses from a MoLA (or conjunctive) process."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from src.simulations.factors import (
    IMBALANCE_ZIPF,
    Q_FLIP_PROB,
    SEPARATION_DELTA,
    SimulationCondition,
)


@dataclass
class SimulatedDataset:
    X: np.ndarray
    Q: np.ndarray
    Q_fit: np.ndarray
    z: np.ndarray
    mu: np.ndarray
    pi: np.ndarray
    theta: np.ndarray
    condition: SimulationCondition

    @property
    def mask(self) -> np.ndarray:
        return ~np.isnan(self.X)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def mixture_weights(n_archetypes: int, imbalance: str, rng: np.random.Generator) -> np.ndarray:
    ranks = np.arange(1, n_archetypes + 1, dtype=float)
    exponent = IMBALANCE_ZIPF[imbalance]
    weights = 1.0 / np.power(ranks, exponent)
    weights = weights / weights.sum()
    rng.shuffle(weights)
    return weights


def archetype_means(
    n_archetypes: int,
    n_skills: int,
    separation: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Place archetypes in [0, 1]^K with controlled pairwise separation."""
    delta = SEPARATION_DELTA[separation]
    mu = np.full((n_archetypes, n_skills), 0.5)
    if n_skills >= n_archetypes:
        groups = np.array_split(np.arange(n_skills), n_archetypes)
        for m, skills in enumerate(groups):
            mu[m] = 0.5 - delta
            mu[m, skills] = 0.5 + delta
            if separation != "high":
                n_extra = max(1, n_skills // (4 * n_archetypes))
                extra = rng.choice(n_skills, size=min(n_extra, n_skills), replace=False)
                mu[m, extra] = 0.5 + 0.5 * delta
    else:
        raw = rng.normal(size=(n_archetypes, n_skills))
        high = raw > np.median(raw, axis=1, keepdims=True)
        mu = np.where(high, 0.5 + delta, 0.5 - delta).astype(float)
    noise = rng.normal(0.0, 0.02, size=mu.shape)
    return np.clip(mu + noise, 0.05, 0.95)


def item_easiness(n_items: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.beta(5.0, 2.0, size=(n_items, 1))
    return np.clip(theta, 0.15, 0.95)


def q_matrix(
    n_items: int,
    n_skills: int,
    coverage: str,
    rng: np.random.Generator,
    skills_per_item: tuple[int, int] = (1, 3),
) -> np.ndarray:
    Q = np.zeros((n_items, n_skills), dtype=float)
    lo, hi = skills_per_item
    if coverage == "balanced":
        skill_p = np.ones(n_skills) / n_skills
    else:
        skill_p = 1.0 / np.power(np.arange(1, n_skills + 1, dtype=float), 1.2)
        skill_p = skill_p / skill_p.sum()

    for j in range(n_items):
        n_q = int(rng.integers(lo, hi + 1))
        n_q = min(n_q, n_skills)
        skills = rng.choice(n_skills, size=n_q, replace=False, p=skill_p)
        Q[j, skills] = 1.0

    unused = np.where(Q.sum(axis=0) == 0)[0]
    for k in unused:
        Q[int(rng.integers(0, n_items)), k] = 1.0
    empty_items = np.where(Q.sum(axis=1) == 0)[0]
    for j in empty_items:
        Q[j, int(rng.integers(0, n_skills))] = 1.0
    return Q


def misspecify_q(Q: np.ndarray, rng: np.random.Generator, flip_prob: float = Q_FLIP_PROB) -> np.ndarray:
    """Perturb the Q-matrix and append one spurious skill column."""
    Q_fit = Q.copy()
    flips = rng.random(Q_fit.shape) < flip_prob
    Q_fit[flips] = 1.0 - Q_fit[flips]
    spurious = (rng.random(Q_fit.shape[0]) < 0.2).astype(float)[:, None]
    Q_fit = np.hstack([Q_fit, spurious])
    empty_items = np.where(Q_fit.sum(axis=1) == 0)[0]
    for j in empty_items:
        Q_fit[j, int(rng.integers(0, Q_fit.shape[1]))] = 1.0
    return Q_fit


def observation_mask(
    n_learners: int,
    n_items: int,
    responses_per_learner: int,
    sampling: str,
    coverage: str,
    rng: np.random.Generator,
) -> np.ndarray:
    n_obs = min(responses_per_learner, n_items)
    mask = np.zeros((n_learners, n_items), dtype=bool)

    if coverage == "balanced":
        item_p = np.ones(n_items) / n_items
    else:
        item_p = 1.0 / np.power(np.arange(1, n_items + 1, dtype=float), 1.1)
        item_p = item_p / item_p.sum()

    if sampling == "iid":
        for i in range(n_learners):
            items = rng.choice(n_items, size=n_obs, replace=False, p=item_p)
            mask[i, items] = True
        return mask

    n_blocks = min(8, n_items)
    block_id = np.array_split(rng.permutation(n_items), n_blocks)
    n_open = 2 if n_blocks > 2 else 1
    for i in range(n_learners):
        open_blocks = rng.choice(n_blocks, size=min(n_open, n_blocks), replace=False)
        eligible = np.concatenate([block_id[b] for b in open_blocks])
        p = item_p[eligible]
        p = p / p.sum()
        take = min(n_obs, eligible.size)
        chosen = rng.choice(eligible, size=take, replace=False, p=p)
        mask[i, chosen] = True
    return mask


def success_probability(
    mu_m: np.ndarray,
    theta_j: float,
    q_j: np.ndarray,
    response_function: str,
) -> float:
    active = q_j > 0
    if not np.any(active):
        return float(theta_j)
    mu_k = np.clip(mu_m[active], 1e-8, 1 - 1e-8)
    q_k = q_j[active]
    if response_function == "compensatory":
        q_norm = q_k / q_k.sum()
        log_p1 = np.log(theta_j) + q_norm @ np.log(mu_k)
        log_p2 = np.log(1.0 - theta_j) + q_norm @ np.log(1.0 - mu_k)
    else:
        log_p1 = np.log(theta_j) + q_k @ np.log(mu_k)
        log_p2 = np.log(1.0 - theta_j) + q_k @ np.log(1.0 - mu_k)
    return float(expit(log_p1 - log_p2))


def component_item_probs(
    mu: np.ndarray,
    theta: np.ndarray,
    Q: np.ndarray,
    response_function: str,
) -> np.ndarray:
    """Return P(X_j = 1 | z = m) with shape (M, J)."""
    n_archetypes, _ = mu.shape
    n_items = Q.shape[0]
    probs = np.empty((n_archetypes, n_items))
    theta = theta.ravel()
    for m in range(n_archetypes):
        for j in range(n_items):
            probs[m, j] = success_probability(mu[m], float(theta[j]), Q[j], response_function)
    return np.clip(probs, 1e-8, 1 - 1e-8)


def generate_dataset(condition: SimulationCondition, seed: int | None = None) -> SimulatedDataset:
    seed = condition.seed if seed is None else seed
    rng = _rng(seed)
    n_items = condition.resolved_n_items()

    pi = mixture_weights(condition.n_archetypes, condition.archetype_imbalance, rng)
    mu = archetype_means(
        condition.n_archetypes, condition.n_skills, condition.archetype_separation, rng
    )
    theta = item_easiness(n_items, rng)
    Q = q_matrix(n_items, condition.n_skills, condition.item_coverage, rng)
    z = rng.choice(condition.n_archetypes, size=condition.n_learners, p=pi)

    P = component_item_probs(mu, theta, Q, condition.response_function)
    draw = rng.random((condition.n_learners, n_items))
    X_full = (draw < P[z]).astype(float)

    mask = observation_mask(
        condition.n_learners,
        n_items,
        condition.responses_per_learner,
        condition.sampling,
        condition.item_coverage,
        rng,
    )
    X = np.where(mask, X_full, np.nan)

    if condition.model_specification == "misspecified":
        Q_fit = misspecify_q(Q, rng)
    else:
        Q_fit = Q.copy()

    return SimulatedDataset(
        X=X,
        Q=Q,
        Q_fit=Q_fit,
        z=z,
        mu=mu,
        pi=pi,
        theta=theta,
        condition=condition,
    )
