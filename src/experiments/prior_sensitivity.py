"""Whether the qualitative analysis's within-archetype readings depend on the prior.

Run from the repository root so that ``import src...`` resolves:

    python -m src.experiments.prior_sensitivity

The response likelihood is invariant to a translation of the archetype log-odds
against the item difficulties, so an unanchored MAP fit has its scale set by the
Beta priors rather than by the data: the priors select one translate from the
equivalence class. The fraction-subtraction archetype table is read on the
scale a ``Beta(2, 2)`` prior selects, and its between-archetype contrasts are
invariant to that choice while its within-archetype attribute comparisons are
not. This refits the same model under other priors and measures how much the
latter actually move.

Two groups of settings, because changing a Beta prior changes two things at
once:

* ``Beta(2,2)``, ``Beta(4,4)``, ``Beta(8,8)`` share a mean of 0.5 and differ
  only in strength, so they isolate shrinkage.
* ``Beta(5,2)``, ``Beta(2,5)``, ``Beta(6,3)`` have means away from 0.5, so the
  translate they prefer moves. These are the settings that bear on the
  translation indeterminacy.

All settings keep ``alpha, beta > 1``, which the M-step's root selection
requires at the domain boundary, and hold the item-easiness prior at the
qualitative analysis's ``Beta(2, 2)``. Note that the item prior also
contributes to selecting the translate, so holding it fixed isolates the
archetype prior's share rather than the whole effect.

Reported per setting, after Hungarian alignment to the reference fit:

``within_spearman``
    Rank correlation between each archetype's attribute ordering and the same
    archetype's ordering under the reference prior, averaged over archetypes.
    This is the comparison the identifiability remark says is not determined by
    the response distribution; a value well below 1 means the archetype table's
    attribute ranking is an artifact of the prior.
``between_r``
    Correlation of ``eta_mk - eta_m'k`` against the reference, over all
    archetype pairs and attributes -- predicted to stay near 1.
``top_attr_stable`` / ``bottom_attr_stable``
    Whether each archetype's strongest and weakest attribute are the same
    attribute as under the reference prior. The paper's report rests on exactly
    this for the headline pair.
``ab_corr``
    Correlation between the two profiles of the headline archetype pair, which
    the paper quotes as ``-0.740``.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.mola import MoLA
from src.simulations.metrics import align_components
from src.utils.datasets import download_frac_subtr_to_pandas

N_COMPONENTS = 5
REFERENCE_PRIOR = (2, 2)
MU_PRIORS = ((2, 2), (4, 4), (8, 8), (5, 2), (2, 5), (6, 3))

BASE_KWARGS = {
    "theta_prior": (2, 2),
    "pi_prior": 1,
    "tol": 1e-5,
    # Above MoLA's default of 50, which binds here: at M=5 the EM loop needs
    # 96 iterations to meet `tol`, and a cap of 50 leaves a materially
    # different fit -- the archetype table of the qualitative analysis is only
    # reproduced once the fit is allowed to converge.
    "max_iter": 500,
    "pseudo_likelihood": False,
    "init": "k-means",
}


def _logit(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _fit(X, Q, mu_prior):
    model = MoLA(Q=Q, n_components=N_COMPONENTS, mu_prior=mu_prior, **BASE_KWARGS)
    model.fit(X)
    return model.mu


def _between_contrasts(mu):
    eta = _logit(mu)
    return np.concatenate(
        [eta[i] - eta[j] for i in range(len(eta)) for j in range(i + 1, len(eta))]
    )


def headline_pair(mu):
    """The two archetypes most strongly anticorrelated across attributes.

    The qualitative analysis identifies its pair this way -- the ``r = -0.740``
    inversion -- so the pair is located by the same criterion here rather than
    by index, which alignment could otherwise shuffle.
    """
    best, pair = 1.0, (0, 1)
    for i in range(len(mu)):
        for j in range(i + 1, len(mu)):
            r = float(pearsonr(mu[i], mu[j])[0])
            if r < best:
                best, pair = r, (i, j)
    return pair, best


def run(priors=MU_PRIORS) -> pd.DataFrame:
    X = download_frac_subtr_to_pandas("fraction.subtraction.data").to_numpy()
    Q = download_frac_subtr_to_pandas("fraction.subtraction.qmatrix").to_numpy()
    X = np.where(X == -1, 0, X)

    mu_ref = _fit(X, Q, REFERENCE_PRIOR)
    ref_between = _between_contrasts(mu_ref)
    ref_top = mu_ref.argmax(axis=1)
    ref_bottom = mu_ref.argmin(axis=1)

    rows = []
    for prior in priors:
        mu = mu_ref if prior == REFERENCE_PRIOR else _fit(X, Q, prior)
        perm = align_components(mu_ref, mu)
        mu_aligned = mu[perm]

        within = [
            float(spearmanr(mu_ref[m], mu_aligned[m]).statistic)
            for m in range(len(mu_ref))
        ]
        between = _between_contrasts(mu_aligned)
        (ia, ib), ab_corr = headline_pair(mu_aligned)

        rows.append(
            {
                "mu_prior": f"Beta{prior}",
                "prior_mean": prior[0] / sum(prior),
                "within_spearman": float(np.mean(within)),
                "within_spearman_min": float(np.min(within)),
                "between_r": float(pearsonr(ref_between, between)[0]),
                "top_attr_stable": int((mu_aligned.argmax(axis=1) == ref_top).sum()),
                "bottom_attr_stable": int((mu_aligned.argmin(axis=1) == ref_bottom).sum()),
                "ab_corr": ab_corr,
                "ab_gap": float(
                    np.abs(mu_aligned[ia] - mu_aligned[ib]).max()
                ),
            }
        )
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="optional CSV path")
    args = parser.parse_args(argv)

    df = run()
    print(f"\nFraction subtraction, M={N_COMPONENTS}, reference prior Beta{REFERENCE_PRIOR}")
    print(f"'stable' counts are out of {N_COMPONENTS} archetypes\n")
    print(df.round(4).to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
