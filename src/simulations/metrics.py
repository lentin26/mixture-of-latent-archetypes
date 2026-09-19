"""Recovery and predictive metrics for simulated MoLA fits."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, roc_auc_score


def align_components(mu_true: np.ndarray, mu_est: np.ndarray) -> np.ndarray:
    """Hungarian match of estimated archetypes to true ones (by L2 on mu).

    Extra estimated components are dropped; extra true components are unmatched.
    Returns a permutation `perm` of estimated indices of length min(M_true, M_est)
    aligned to the first matches among true components.
    """
    cost = np.linalg.norm(mu_true[:, None, :] - mu_est[None, :, :], axis=2)
    true_ix, est_ix = linear_sum_assignment(cost)
    return est_ix


def permute_rows(arr: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return arr[perm]


def repeat_alignment_spread(mu_ests: Sequence[np.ndarray]) -> float:
    """Self-referential stability metric: spread of repeated `mu` estimates
    around EACH OTHER, with no ground truth involved.

    Unlike `mu_rmse`/`align_components`'s usual use (aligning an estimate to
    a KNOWN true `mu`), a real "is this fit stable" check can't peek at the
    truth -- so this treats `mu_ests[0]` as an arbitrary common reference
    frame, aligns every other estimate to it (`align_components` works on
    any two same-shaped matrices, not just true-vs-estimated), stacks the
    aligned estimates, and returns the same "mean per-skill std across
    repeats" formula `plot_component_recovery_distribution` uses (viz.py),
    just self-referential instead of vs.-truth. Requires at least 2
    estimates, all sharing the same `(n_components, n_skills)` shape --
    call it once per grid point of an M-sweep (e.g. `run_m_sweep_with_
    repeats`), not across different `n_components` values.
    """
    mu_ests = [np.asarray(m, dtype=float) for m in mu_ests]
    if len(mu_ests) < 2:
        raise ValueError("repeat_alignment_spread needs at least 2 estimates")
    reference = mu_ests[0]
    aligned = [reference]
    for mu_est in mu_ests[1:]:
        if mu_est.shape != reference.shape:
            raise ValueError(
                f"all mu_ests must share one shape; got {reference.shape} and {mu_est.shape}"
            )
        perm = align_components(reference, mu_est)
        aligned.append(mu_est[perm])
    stacked = np.stack(aligned)  # (n_repeats, n_components, n_skills)
    return float(stacked.std(axis=0).mean())


def mu_rmse(mu_true: np.ndarray, mu_est: np.ndarray) -> float:
    k = min(mu_true.shape[1], mu_est.shape[1])
    perm = align_components(mu_true[:, :k], mu_est[:, :k])
    m = min(mu_true.shape[0], perm.size)
    diff = mu_true[:m, :k] - mu_est[perm[:m], :k]
    return float(np.sqrt(np.mean(diff**2)))


def pi_mae(pi_true: np.ndarray, pi_est: np.ndarray, perm: np.ndarray) -> float:
    pi_true = pi_true.ravel()
    pi_est = pi_est.ravel()
    m = min(pi_true.size, perm.size)
    return float(np.mean(np.abs(pi_true[:m] - pi_est[perm[:m]])))


def pi_rmse(pi_true: np.ndarray, pi_est: np.ndarray, perm: np.ndarray) -> float:
    """RMSE version of `pi_mae`, for reporting alongside `mu_rmse`/`theta_rmse`
    on the same scale (RMSE throughout) rather than mixing MAE and RMSE."""
    pi_true = pi_true.ravel()
    pi_est = pi_est.ravel()
    m = min(pi_true.size, perm.size)
    diff = pi_true[:m] - pi_est[perm[:m]]
    return float(np.sqrt(np.mean(diff**2)))


def theta_rmse(theta_true: np.ndarray, theta_est: np.ndarray) -> float:
    a, b = theta_true.ravel(), theta_est.ravel()
    n = min(a.size, b.size)
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


def assignment_ari(z_true: np.ndarray, post: np.ndarray, perm: np.ndarray) -> float:
    z_hat = np.argmax(post, axis=1)
    remap = np.full(post.shape[1], fill_value=-1, dtype=int)
    for true_m, est_m in enumerate(perm):
        if est_m < remap.size:
            remap[est_m] = true_m
    z_aligned = remap[z_hat]
    valid = z_aligned >= 0
    if valid.sum() < 2:
        return float("nan")
    return float(adjusted_rand_score(z_true[valid], z_aligned[valid]))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_true - y_prob) ** 2))


def masked_nll(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-15
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def masked_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))
