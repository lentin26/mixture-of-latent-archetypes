"""What the one-standard-error rule selects on fraction subtraction.

Run from the repository root so that ``import src...`` resolves:

    python -m src.experiments.m_rule_frac_subtr

The low-dimensional protocol selects the number of archetypes by minimizing
held-out negative log-likelihood on a slice of each split's training learners.
That rule takes the single best value and ignores how precisely it is
estimated, which on a dataset this small is not very precisely at all. The
one-standard-error rule instead takes the smallest candidate whose held-out
NLL is within one standard error of the best, the standard error being that of
the mean over held-out responses.

This matters here because the qualitative analysis reads five archetypes, a
number chosen for what an instructor can act on rather than for predictive
fit. Whether the data tolerate a value that small is a question the selection
rule should answer, and the two rules answer it differently.

Mirrors ``low_dim.select_mola_m``: the same configuration, the same
train/select split proportions, and the same masked scoring.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np

from src.experiments.low_dim import get_y_proba_and_true
from src.mola import MoLA
from src.utils.datasets import download_frac_subtr_to_pandas

M_GRID = (2, 4, 6, 8, 10, 12, 14)
TRAIN_FRAC = 0.8
SELECT_FRAC = 0.2
SPLIT_SEED = 0

MOLA_KWARGS = {
    "mu_prior": (2, 2),
    "theta_prior": (2, 2),
    "pi_prior": 2,
    "tol": 1e-5,
    # Above MoLA's default of 50, which binds on this dataset; see
    # src.experiments.m_cost for the convergence profile.
    "max_iter": 500,
    "pseudo_likelihood": False,
}


def _load():
    X = download_frac_subtr_to_pandas("fraction.subtraction.data").to_numpy()
    Q = download_frac_subtr_to_pandas("fraction.subtraction.qmatrix").to_numpy()
    return np.where(X == -1, 0, X), Q


def _nll_terms(model, X_select):
    y_prob, y_true = get_y_proba_and_true(model, X_select)
    p = np.clip(y_prob, 1e-12, 1.0 - 1e-12)
    return -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))


def run(n_splits: int = 100, m_grid=M_GRID) -> dict:
    X, Q = _load()
    n_learners = X.shape[0]
    n_train = int(round(TRAIN_FRAC * n_learners))
    rng = np.random.default_rng(SPLIT_SEED)

    argmin, one_se, ses = [], [], []
    for split in range(n_splits):
        train = X[rng.permutation(n_learners)[:n_train]]
        inner = np.random.default_rng(split).permutation(n_train)
        n_select = int(round(SELECT_FRAC * n_train))
        X_fit, X_select = train[inner[n_select:]], train[inner[:n_select]]

        curve = []
        for m in m_grid:
            model = MoLA(Q=Q, n_components=m, **MOLA_KWARGS)
            model.fit(X_fit)
            terms = _nll_terms(model, X_select)
            curve.append((m, float(terms.mean()), float(terms.std(ddof=1) / np.sqrt(terms.size))))

        best_ix = int(np.argmin([c[1] for c in curve]))
        best, se = curve[best_ix][1], curve[best_ix][2]
        argmin.append(curve[best_ix][0])
        one_se.append(next(m for m, v, _ in curve if v <= best + se))
        ses.append(se)

    return {
        "n_splits": n_splits,
        "mean_se": float(np.mean(ses)),
        "argmin_counts": dict(sorted(Counter(argmin).items())),
        "one_se_counts": dict(sorted(Counter(one_se).items())),
        "one_se_median": int(np.median(one_se)),
        "one_se_at_most_6": float(np.mean(np.asarray(one_se) <= 6)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-splits", type=int, default=100)
    args = parser.parse_args(argv)

    out = run(n_splits=args.n_splits)
    print(f"\nFraction subtraction, {out['n_splits']} splits, grid {M_GRID}")
    print(f"  mean SE of the held-out NLL : {out['mean_se']:.4f}")
    print(f"  argmin selects              : {out['argmin_counts']}")
    print(f"  one-SE selects              : {out['one_se_counts']}")
    print(f"  one-SE median               : {out['one_se_median']}")
    print(f"  one-SE at most 6            : {out['one_se_at_most_6']:.0%} of splits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
