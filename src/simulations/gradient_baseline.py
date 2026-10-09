"""Autodiff (PyTorch) baseline: direct gradient-based MLE fitting of MoLA.

This module exists to answer one question empirically: is MoLA's nested
EM/MM scheme (`src.mola.MoLA`) actually doing something a generic
gradient-based optimizer couldn't do just as well on the same observed-data
negative log-likelihood? It implements that *exact* likelihood a second
time, independently, in PyTorch, so (a) it can be optimized directly with
autograd (Adam, L-BFGS) instead of via the hand-derived EM/MM updates, and
(b) its agreement with `MoLA.get_log_likelihood`/`get_nll` at a shared
parameter point is itself a cheap independent correctness check on the
NumPy implementation.

The comparison is restricted to the *unregularized* (flat-prior) objective
on both sides -- MoLA's EM/MM M-step is MAP under Beta/Dirichlet priors by
default (see `src.mola.train`), and its monotonic-ascent guarantee is a
property of the (possibly penalized) complete-data objective it actually
climbs, not of the plain observed-data NLL in general. Comparing EM/MM
against a gradient method that optimizes the plain NLL only makes sense if
EM/MM is *also* climbing the plain NLL, i.e. its priors are flattened to
Beta(1, 1)/Dirichlet(1) (which, at alpha=beta=1, contribute a zero offset
to MoLA's sufficient-statistic M-step and are exactly equivalent to no
prior at all).

Mirrors `MoLA.get_log_likelihood` (`pseudo_likelihood=False` branch) and
`MoLA.get_nll` in `src/mola/train.py` term for term:

    log P(correct | item j, archetype m) = log theta_j + Q_jk log mu_mk  (summed over k)
    log P(incorrect| item j, archetype m) = log(1-theta_j) + Q_jk log(1-mu_mk)
    normalized per (j, m) by log-sum-exp of the two unnormalized terms
    per-learner log-likelihood = logsumexp_m [ log pi_m + sum_{j observed} (...) ]
    NLL = -sum_n per-learner log-likelihood

`Q` here is assumed already row-normalized (as `MoLA.__init__` does via
`_normalize_q`), so its rows sum to 1 and the exponents above are exactly
`Q_jk`, matching the compensatory ("MoLA-id") response function.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------


def torch_nll(
    X: torch.Tensor,
    Q: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Observed-data NLL, summed over observed entries, matching
    `MoLA.get_nll()` exactly (same quantity MoLA reports in `nll_trace`).

    X : (N, J) with 1/0 for correct/incorrect and NaN for unobserved.
    Q : (J, K) *raw* (not yet row-normalized) Q-matrix -- normalized here,
        matching `MoLA.__init__`'s `_normalize_q`, so callers can pass the
        same `Q` they'd hand to `MoLA(Q=...)` directly.
    mu : (M, K) in (0, 1).
    theta : (J, 1) or (J,) in (0, 1).
    pi : (M,) or (M, 1), sums to 1.
    """
    Q = Q / Q.sum(dim=1, keepdim=True)
    theta = theta.reshape(-1, 1)  # (J, 1)
    pi = pi.reshape(-1)  # (M,)

    log_mu = torch.log(mu.T)  # (K, M)
    log_1m_mu = torch.log(1 - mu.T)  # (K, M)

    log_P1 = torch.log(theta) + Q @ log_mu  # (J, M)
    log_P2 = torch.log(1 - theta) + Q @ log_1m_mu  # (J, M)
    log_P_norm = torch.logaddexp(log_P1, log_P2)  # (J, M)

    log_P1 = log_P1 - log_P_norm
    log_P2 = log_P2 - log_P_norm

    mask = ~torch.isnan(X)
    X0 = torch.nan_to_num(X, nan=0.0)
    X1 = torch.where(mask & (X0 == 1), torch.ones_like(X0), torch.zeros_like(X0))
    X2 = torch.where(mask & (X0 == 0), torch.ones_like(X0), torch.zeros_like(X0))

    # (N, J) @ (J, M) -> (N, M)
    log_likelihood = X1 @ log_P1 + X2 @ log_P2

    log_pi = torch.log(pi + eps)
    log_weighted = log_likelihood + log_pi  # (N, M) + (M,) broadcasts
    logsum = torch.logsumexp(log_weighted, dim=1)  # (N,)
    return -logsum.sum()


def unconstrained_to_params(
    mu_raw: torch.Tensor, theta_raw: torch.Tensor, pi_raw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """sigmoid/softmax reparameterization so the optimizer runs over all of
    R^(...) while mu, theta stay in (0, 1) and pi stays on the simplex --
    exactly the constraints MoLA's EM/MM updates enforce by construction."""
    mu = torch.sigmoid(mu_raw)
    theta = torch.sigmoid(theta_raw)
    pi = torch.softmax(pi_raw, dim=0)
    return mu, theta, pi


def params_to_unconstrained(
    mu: np.ndarray, theta: np.ndarray, pi: np.ndarray, clip: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse map (logit / log), so a gradient run can start from the exact
    same point as an EM/MM run (e.g. the same k-means init)."""
    mu = np.clip(mu, clip, 1 - clip)
    theta = np.clip(theta, clip, 1 - clip)
    pi = np.clip(pi.reshape(-1), clip, None)
    pi = pi / pi.sum()

    mu_raw = torch.from_numpy(np.log(mu / (1 - mu))).double()
    theta_raw = torch.from_numpy(np.log(theta / (1 - theta))).double()
    pi_raw = torch.from_numpy(np.log(pi)).double()  # softmax is shift-invariant
    return mu_raw, theta_raw, pi_raw


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass
class GradientFitResult:
    mu: np.ndarray
    theta: np.ndarray
    pi: np.ndarray
    nll_trace: list[float] = field(default_factory=list)
    time_trace: list[float] = field(default_factory=list)  # wall-clock seconds, cumulative
    n_steps: int = 0
    converged: bool = False
    optimizer: str = ""
    lr: float | None = None


def fit_gradient(
    X: np.ndarray,
    Q: np.ndarray,
    mu_init: np.ndarray,
    theta_init: np.ndarray,
    pi_init: np.ndarray,
    optimizer: str = "adam",
    lr: float = 0.1,
    max_iter: int = 2000,
    tol: float = 1e-5,
    dtype: torch.dtype = torch.float64,
) -> GradientFitResult:
    """Fit MoLA's parameters by direct autodiff gradient descent on the
    plain (unregularized) observed-data NLL, starting from `mu_init`,
    `theta_init`, `pi_init` -- pass MoLA/MoLAWithInit's own k-means init
    here for an apples-to-apples comparison against EM/MM.

    `tol` mirrors `MoLA`'s own stopping rule: stop once the relative NLL
    change between successive steps drops below `tol`.
    """
    X_t = torch.from_numpy(X).to(dtype)
    Q_t = torch.from_numpy(Q).to(dtype)

    mu_raw, theta_raw, pi_raw = params_to_unconstrained(mu_init, theta_init, pi_init)
    mu_raw = mu_raw.to(dtype).requires_grad_(True)
    theta_raw = theta_raw.to(dtype).requires_grad_(True)
    pi_raw = pi_raw.to(dtype).requires_grad_(True)

    params = [mu_raw, theta_raw, pi_raw]

    if optimizer == "adam":
        opt = torch.optim.Adam(params, lr=lr)
    elif optimizer == "lbfgs":
        opt = torch.optim.LBFGS(params, lr=lr, max_iter=1, line_search_fn="strong_wolfe")
    elif optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=lr)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")

    nll_trace: list[float] = []
    time_trace: list[float] = []
    t0 = time.perf_counter()
    converged = False
    step = 0

    def closure():
        opt.zero_grad()
        mu, theta, pi = unconstrained_to_params(mu_raw, theta_raw, pi_raw)
        loss = torch_nll(X_t, Q_t, mu, theta, pi)
        loss.backward()
        return loss

    for step in range(1, max_iter + 1):
        if optimizer == "lbfgs":
            loss = opt.step(closure)
        else:
            opt.zero_grad()
            mu, theta, pi = unconstrained_to_params(mu_raw, theta_raw, pi_raw)
            loss = torch_nll(X_t, Q_t, mu, theta, pi)
            loss.backward()
            opt.step()

        nll = float(loss.detach())
        nll_trace.append(nll)
        time_trace.append(time.perf_counter() - t0)

        if len(nll_trace) > 1:
            a, b = nll_trace[-1], nll_trace[-2]
            if abs(b - a) / max(abs(a), 1e-12) < tol:
                converged = True
                break

    with torch.no_grad():
        mu, theta, pi = unconstrained_to_params(mu_raw, theta_raw, pi_raw)

    return GradientFitResult(
        mu=mu.numpy(),
        theta=theta.numpy(),
        pi=pi.numpy(),
        nll_trace=nll_trace,
        time_trace=time_trace,
        n_steps=step,
        converged=converged,
        optimizer=optimizer,
        lr=lr,
    )
