"""Recovery and predictive metrics for simulated MoLA fits."""

from __future__ import annotations

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
