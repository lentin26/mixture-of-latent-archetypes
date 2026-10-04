"""Recovery and predictive metrics for simulated MoLA fits."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats
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


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0, ci: float = 0.95
) -> dict:
    """Paired bootstrap CI for `mean(a) - mean(b)` over matched samples.

    `a`/`b` must be the same metric for the same units under two
    conditions -- e.g. the same evaluation learners' `mola_auc` and
    `neural_cdm_auc` -- not two independent samples. Resamples *indices*
    with replacement (jointly, so each resample keeps `a[i]`/`b[i]`
    paired) `n_boot` times, computing `mean(a) - mean(b)` on each
    resample, then reports the percentile CI of that distribution.

    Rows where either `a` or `b` is NaN (e.g. an evaluation learner whose
    held-out responses are all one class, so `masked_auc` is undefined)
    are dropped before resampling, since they can't be paired.

    A CI that excludes zero supports a claim that the two conditions
    differ significantly. A CI that includes zero supports only "no
    significant difference detected at this sample size" -- NOT a claim
    that the two conditions are equivalent (failing to reject a null of
    no difference is not evidence the true difference is zero; a genuine
    equivalence claim needs a pre-specified equivalence margin and a
    dedicated test such as TOST, which this does not attempt).

    Returns `{"n", "point_diff", "ci_lo", "ci_hi", "excludes_zero",
    "p_value"}`. `p_value` is the two-sided bootstrap p-value
    `2 * min(P(diff <= 0), P(diff >= 0))` (estimated from the same
    resampled distribution, capped at 1.0) -- the CI-based `excludes_zero`
    and this `p_value` agree by construction (`p_value <= 1 - ci` iff the
    CI excludes zero), but a p-value is what a multiple-comparisons
    correction such as Holm-Bonferroni (see `holm_bonferroni`) needs.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    n = a.size

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)

    alpha = 1.0 - ci
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_value = min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    return {
        "n": int(n),
        "point_diff": float(a.mean() - b.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "p_value": float(p_value),
    }


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm's step-down multiple-comparisons correction.

    Controls the family-wise error rate (probability of *any* false
    positive across the whole family of tests) with less power loss than
    a plain Bonferroni correction, which uses the same `alpha / m`
    threshold for every test regardless of how small its p-value is.

    Sorts `p_values` ascending; the `i`-th smallest (1-indexed) is
    rejected only if it's `<= alpha / (m - i + 1)` *and* every smaller
    p-value was also rejected -- the step-down stops at the first
    non-rejection, so a large p-value earlier in the sorted order can
    prevent a later, larger p-value from being called significant even
    if it would pass its own threshold in isolation.

    Returns significance flags in the *original* input order (not sorted
    order), so callers can zip the result directly with their own
    comparison labels.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = p_values.size
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    significant_sorted = np.zeros(m, dtype=bool)
    for i, p in enumerate(sorted_p):
        threshold = alpha / (m - i)
        if p <= threshold:
            significant_sorted[i] = True
        else:
            break  # step-down: stop at the first non-rejection

    significant = np.zeros(m, dtype=bool)
    significant[order] = significant_sorted
    return significant.tolist()


def nadeau_bengio_test(
    a: np.ndarray, b: np.ndarray, n_train: int, n_test: int, ci: float = 0.95
) -> dict:
    """Corrected paired t-test for repeated random train/test resampling
    (Nadeau & Bengio, 2003) -- e.g. sklearn's `ShuffleSplit` run many
    times over the same underlying dataset, as opposed to genuinely
    independent evaluation units.

    `a`/`b` must be the same metric for the same model comparison across
    `n` such repeated splits. Unlike `paired_bootstrap_diff`, that `n` is
    *not* a count of independent observations here: every split's
    training set overlaps heavily with every other split's (and, for
    `ShuffleSplit` specifically, test sets overlap across splits too,
    since the same individual can land in many splits' test sets), so
    the resulting per-split metric values are correlated. Both a plain
    paired t-test and a plain paired bootstrap (`paired_bootstrap_diff`)
    treat the `n` splits as independent and so *underestimate* the
    variance of the mean difference, inflating the false-positive rate --
    confirmed directly on this codebase's low-dim experiment, where
    several comparisons that were "significant" under
    `paired_bootstrap_diff` were not once corrected.

    This replaces the naive variance-of-the-mean term `1/n` with
    `1/n + n_test/n_train`, which accounts for the overlap-induced
    correlation and is always larger than the naive term (equal only in
    the degenerate case `n_test = 0`). The test statistic is then
    `mean(diff) / sqrt(corrected_var)`, referred to a `t` distribution
    with `n - 1` degrees of freedom, exactly as an ordinary paired
    t-test would use `mean(diff) / sqrt(sample_var / n)`.

    Returns the same shape as `paired_bootstrap_diff`:
    `{"n", "point_diff", "ci_lo", "ci_hi", "excludes_zero", "p_value"}`,
    so it's a drop-in replacement wherever that function's output feeds
    `holm_bonferroni` or a results table.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[valid]
    n = d.size

    mean_d = float(d.mean())
    sample_var = float(d.var(ddof=1))
    corrected_var = (1.0 / n + n_test / n_train) * sample_var
    se = float(np.sqrt(corrected_var))
    df = n - 1

    t_stat = mean_d / se if se > 0 else 0.0
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df)))

    alpha = 1.0 - ci
    margin = float(stats.t.ppf(1 - alpha / 2, df=df) * se)
    ci_lo, ci_hi = mean_d - margin, mean_d + margin

    return {
        "n": int(n),
        "point_diff": mean_d,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        "p_value": p_value,
    }
