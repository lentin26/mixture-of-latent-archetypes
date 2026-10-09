"""Driver: EM/MM vs. direct autodiff gradient descent on MoLA's plain NLL.

Fits the BASELINE simulation condition (M=4, K=20, N=2,000, 30
responses/learner, k-means init) three ways from the *same* k-means start:

  1. MoLA's own EM/MM loop (`src.mola.MoLA`), flat Beta(1,1)/Dirichlet(1)
     priors so it is climbing the same, unregularized plain NLL as (2)/(3).
  2. Adam (`src.simulations.gradient_baseline.fit_gradient`), swept over a
     few learning rates.
  3. L-BFGS, the standard quasi-Newton baseline for smooth likelihoods.

Records, for each: the NLL trace, its wall-clock timestamps, and the
number of iterations in which NLL *increased* (a direct violation of
monotonic ascent -- EM/MM is guaranteed zero by construction; gradient
methods are not guaranteed anything).
"""

from __future__ import annotations

import time

import numpy as np

from src.simulations.dgp import generate_dataset, resample_dataset
from src.simulations.factors import BASELINE, SimulationCondition
from src.simulations.run import MoLAWithInit, _kmeans_init
from src.simulations.gradient_baseline import fit_gradient


class TimedMoLA(MoLAWithInit):
    """`MoLAWithInit` (EM/MM that can start from a supplied `mu`/`theta`/`pi`,
    e.g. a shared k-means init -- see `src.simulations.run`) with a
    wall-clock timestamp recorded alongside every `nll_trace` entry, via the
    same `get_nll` call site the base fit loop already uses once per
    iteration (`_MoLABase.fit`) -- no change to the EM/MM algorithm itself,
    just an added timestamp."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_trace: list[float] = []
        self._t0: float | None = None

    def get_nll(self):
        if self._t0 is None:
            self._t0 = time.perf_counter()
        nll = super().get_nll()
        self.time_trace.append(time.perf_counter() - self._t0)
        return nll


def run(
    condition: SimulationCondition = BASELINE,
    seed: int = 0,
    adam_lrs: tuple[float, ...] = (0.003, 0.01, 0.03, 0.1, 0.3),
    em_max_iter: int = 200,
    gradient_max_iter: int = 3000,
    lbfgs_max_iter: int = 500,
    verbose: bool = True,
) -> dict:
    """Fit `condition` three ways from the same k-means init: EM/MM (flat
    priors, i.e. plain MLE -- see module docstring), an Adam learning-rate
    sweep, and L-BFGS. Returns a dict keyed by method name
    (`"em_mm"`, `"adam_lr<lr>"`, `"lbfgs"`), each holding `nll_trace`,
    `time_trace` (wall-clock seconds, cumulative from that method's own
    start), `n_steps`, `violations` (count of NLL *increases*, i.e.
    monotonicity violations), `final_nll`, `total_time`, and that method's
    final `mu`/`theta`/`pi` estimates (for recovery-vs-ground-truth or
    cross-method agreement checks -- see `src.simulations.metrics`'s
    `mu_rmse`/`theta_rmse`/`pi_rmse`/`align_components`).
    """
    rng = np.random.default_rng(seed)
    data = generate_dataset(condition, seed=seed)
    X = data.X
    Q = data.Q_fit
    n_components = data.condition.resolved_n_components_fit()

    mu_init, theta_init, pi_init = _kmeans_init(X, Q, n_components, rng)

    results = {}

    # 1) EM/MM, flat priors (pure MLE)
    model = TimedMoLA(
        mu_init=mu_init,
        theta_init=theta_init,
        pi_init=pi_init,
        Q=Q,
        n_components=n_components,
        mu_prior=(1, 1),
        theta_prior=(1, 1),
        pi_prior=1,
        max_iter=em_max_iter,
        tol=1e-6,
    )
    t0 = time.perf_counter()
    model.fit(X)
    fit_time = time.perf_counter() - t0
    em_nll = np.array(model.nll_trace)
    em_time = np.array(model.time_trace)
    em_viol = int((np.diff(em_nll) > 1e-6).sum())
    results["em_mm"] = dict(
        nll_trace=em_nll, time_trace=em_time, n_steps=len(em_nll),
        violations=em_viol, final_nll=float(em_nll[-1]), total_time=fit_time,
        converged=len(em_nll) < em_max_iter,
        mu=model.mu.copy(), theta=model.theta.copy(), pi=model.pi.copy(),
    )
    if verbose:
        print(f"EM/MM      : {len(em_nll):4d} iters, final NLL={em_nll[-1]:.4f}, "
              f"monotonicity violations={em_viol}, wall-clock={fit_time:.3f}s")

    # 2) Adam, learning-rate sweep
    for lr in adam_lrs:
        res = fit_gradient(
            X, Q, mu_init, theta_init, pi_init,
            optimizer="adam", lr=lr, max_iter=gradient_max_iter, tol=1e-6,
        )
        nll = np.array(res.nll_trace)
        viol = int((np.diff(nll) > 1e-6).sum())
        key = f"adam_lr{lr}"
        results[key] = dict(
            nll_trace=nll, time_trace=np.array(res.time_trace), n_steps=res.n_steps,
            violations=viol, final_nll=float(nll[-1]), total_time=res.time_trace[-1],
            converged=res.converged, mu=res.mu, theta=res.theta, pi=res.pi,
        )
        if verbose:
            print(f"Adam lr={lr:<6}: {res.n_steps:4d} iters, final NLL={nll[-1]:.4f}, "
                  f"monotonicity violations={viol}, wall-clock={res.time_trace[-1]:.3f}s, "
                  f"converged={res.converged}")

    # 3) L-BFGS
    res = fit_gradient(
        X, Q, mu_init, theta_init, pi_init,
        optimizer="lbfgs", lr=1.0, max_iter=lbfgs_max_iter, tol=1e-8,
    )
    nll = np.array(res.nll_trace)
    viol = int((np.diff(nll) > 1e-6).sum())
    results["lbfgs"] = dict(
        nll_trace=nll, time_trace=np.array(res.time_trace), n_steps=res.n_steps,
        violations=viol, final_nll=float(nll[-1]), total_time=res.time_trace[-1],
        converged=res.converged, mu=res.mu, theta=res.theta, pi=res.pi,
    )
    if verbose:
        print(f"L-BFGS     : {res.n_steps:4d} iters, final NLL={nll[-1]:.4f}, "
              f"monotonicity violations={viol}, wall-clock={res.time_trace[-1]:.3f}s, "
              f"converged={res.converged}")

    return results


def stability_repeats(
    condition: SimulationCondition = BASELINE,
    population_seed: int = 0,
    n_repeats: int = 5,
    base_sample_seed: int = 0,
    adam_lr: float = 0.03,
    em_max_iter: int = 200,
    gradient_max_iter: int = 3000,
    lbfgs_max_iter: int = 500,
) -> dict[str, list[np.ndarray]]:
    """Sampling-stability check across EM/MM, Adam, and L-BFGS.

    Mirrors `src.simulations.run.run_sample_repeats`'s own methodology
    exactly (same `generate_dataset` + `resample_dataset` split): draw ONE
    population (`generate_dataset(condition, seed=population_seed)`), then
    redraw just the sample (a new cohort of learners, new realized
    responses) `n_repeats` times. Each repeat gets its own FRESH k-means
    init computed on that repeat's own resampled data -- not one init
    shared across repeats -- exactly matching how `run_sample_repeats`
    isolates genuine sampling variability from EM's own initialization
    draw (meaningless to vary separately once `initialization="k-means"`,
    per that function's docstring).

    Returns `{"em_mm": [mu_0, ..., mu_{n_repeats-1}], "adam": [...],
    "lbfgs": [...]}` -- feed any of these lists to
    `src.simulations.metrics.repeat_alignment_spread` to compare how much
    each method's recovered archetypes move around just from resampling.
    """
    population = generate_dataset(condition, seed=population_seed)
    out: dict[str, list[np.ndarray]] = {"em_mm": [], "adam": [], "lbfgs": []}

    for i in range(n_repeats):
        sample_seed = base_sample_seed + i
        data = resample_dataset(population, seed=sample_seed)
        X, Q = data.X, data.Q_fit
        n_components = data.condition.resolved_n_components_fit()
        rng = np.random.default_rng(sample_seed)
        mu_init, theta_init, pi_init = _kmeans_init(X, Q, n_components, rng)

        model = MoLAWithInit(
            mu_init=mu_init, theta_init=theta_init, pi_init=pi_init,
            Q=Q, n_components=n_components,
            mu_prior=(1, 1), theta_prior=(1, 1), pi_prior=1,
            max_iter=em_max_iter, tol=1e-6,
        )
        model.fit(X)
        out["em_mm"].append(model.mu.copy())

        res = fit_gradient(
            X, Q, mu_init, theta_init, pi_init,
            optimizer="adam", lr=adam_lr, max_iter=gradient_max_iter, tol=1e-6,
        )
        out["adam"].append(res.mu)

        res = fit_gradient(
            X, Q, mu_init, theta_init, pi_init,
            optimizer="lbfgs", lr=1.0, max_iter=lbfgs_max_iter, tol=1e-8,
        )
        out["lbfgs"].append(res.mu)

    return out


if __name__ == "__main__":
    run()
