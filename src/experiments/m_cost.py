"""Predictive cost of fixing the number of archetypes below the selected value.

Run from the repository root so that ``import src...`` resolves:

    python -m src.experiments.m_cost

    # a different fixed value, or fewer splits
    python -m src.experiments.m_cost --m-fixed 4 --n-splits 20

The qualitative analysis caps ``M`` at 5 on pedagogical grounds -- roughly the
number of groups an instructor can differentiate across -- while the
low-dimensional protocol selects ``M`` for predictive performance and lands
higher. Those are different questions, but the smaller archetype set is not
free, and this quantifies what it costs.

For each of ``n_splits`` learner-level 80/20 partitions, ``M`` is selected on a
slice of the training learners by held-out predictive log-likelihood (the
protocol of ``low_dim.select_mola_m``), then MoLA is refit on the full training
set twice -- once at the selected ``M``, once at the fixed one -- and both are
scored on the same masked held-out responses of that split's test learners. The
reported quantity is the paired difference in test NLL, positive when the fixed
value predicts worse.

This builds its own splits from ``SPLIT_SEED`` rather than the ones the
low-dimensional results use, which are constructed in the analysis notebook, so
the number here is comparable across runs of this module but is not a row of
the sensitivity table in the paper.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from src.experiments.low_dim import get_y_proba_and_true, select_mola_m
from src.mola import MoLA
from src.simulations.metrics import masked_nll
from src.utils.datasets import download_frac_subtr_to_pandas

SPLIT_SEED = 0
TRAIN_FRAC = 0.8
M_GRID = (2, 4, 6, 8, 10, 12, 14)

# Matches low_dim.fit_mola's configuration, so the fits differ from the
# low-dimensional results only in which splits they run on.
MOLA_KWARGS = {
    "mu_prior": (2, 2),
    "theta_prior": (2, 2),
    "pi_prior": 2,
    "tol": 1e-5,
    "max_iter": 50,
    "pseudo_likelihood": False,
}


def _load():
    X = download_frac_subtr_to_pandas("fraction.subtraction.data").to_numpy()
    Q = download_frac_subtr_to_pandas("fraction.subtraction.qmatrix").to_numpy()
    return np.where(X == -1, 0, X), Q


def _test_nll(X_train, X_test, Q, n_components):
    model = MoLA(Q=Q, n_components=n_components, **MOLA_KWARGS)
    model.fit(X_train)
    y_proba, y_true = get_y_proba_and_true(model, X_test)
    return masked_nll(y_true, y_proba)


def run(n_splits: int = 100, m_fixed: int = 5, m_grid=M_GRID) -> dict:
    X, Q = _load()
    rng = np.random.default_rng(SPLIT_SEED)
    n_learners = X.shape[0]
    n_train = int(round(TRAIN_FRAC * n_learners))

    selected, nll_selected, nll_fixed = [], [], []
    for split_i in range(n_splits):
        perm = rng.permutation(n_learners)
        X_train, X_test = X[perm[:n_train]], X[perm[n_train:]]

        best_m, _ = select_mola_m(X_train, Q, list(m_grid), seed=split_i)
        selected.append(best_m)
        nll_selected.append(_test_nll(X_train, X_test, Q, best_m))
        nll_fixed.append(_test_nll(X_train, X_test, Q, m_fixed))

    diff = np.asarray(nll_fixed) - np.asarray(nll_selected)
    return {
        "m_fixed": m_fixed,
        "n_splits": n_splits,
        "selected_counts": {int(m): int(c) for m, c in zip(*np.unique(selected, return_counts=True))},
        "mean_diff": float(diff.mean()),
        "sd_diff": float(diff.std(ddof=1)),
        "worse_in": int((diff > 0).sum()),
        "mean_nll_selected": float(np.mean(nll_selected)),
        "mean_nll_fixed": float(np.mean(nll_fixed)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-splits", type=int, default=100)
    parser.add_argument("--m-fixed", type=int, default=5)
    args = parser.parse_args(argv)

    out = run(n_splits=args.n_splits, m_fixed=args.m_fixed)
    print(f"\nFraction subtraction, {out['n_splits']} splits, fixed M={out['m_fixed']}")
    print(f"  selected M counts:   {out['selected_counts']}")
    print(f"  mean test NLL, selected M: {out['mean_nll_selected']:.4f}")
    print(f"  mean test NLL, fixed M:    {out['mean_nll_fixed']:.4f}")
    print(
        f"  paired difference (fixed - selected): {out['mean_diff']:+.4f} "
        f"(SD {out['sd_diff']:.4f}), worse in {out['worse_in']}/{out['n_splits']} splits"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
