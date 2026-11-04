"""Reproduces the rank-condition verification reported in the identifiability
remark of the paper.

Run from the repository root so that ``import src...`` resolves:

    python -m src.experiments.rank_condition

    # skip EdNet (needs notebooks/data/ednet/*.zip)
    python -m src.experiments.rank_condition --skip-ednet

The mixture-identifiability proposition assumes the items admit a partition into
three groups within which the ``M`` archetype-conditional response distributions
are linearly independent. The accompanying remark declines to obtain that from
generic-identifiability results, because MoLA's response probabilities occupy a
constrained subset of the unrestricted latent-class parameter space, and a
property holding off a measure-zero subset of the ambient space need not hold
off a measure-zero subset of that subset. The condition is therefore checked
directly, at each parameter value of interest:

* the baseline simulation condition, at its generating parameters;
* the fraction-subtraction fit, at its fitted parameters;
* the EdNet-KT1 fits at the smallest and largest population-size scale points,
  at their fitted parameters.

See ``src.mola.identifiability`` for why the factor matrices are never formed
and why the minors are equilibrated before their singular values are taken.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.mola import MoLA
from src.mola.identifiability import archetype_item_probs, rank_certificate

# M as selected by held-out predictive log-likelihood in each experiment; see
# the model-selection subsections and the scale-point table in the paper.
FRAC_SUBTR_M = 5
EDNET_SCALE_POINTS = {200: 16, 2000: 16}


def _report(label: str, P: np.ndarray, seed: int = 0) -> dict:
    result = rank_certificate(P, seed=seed)
    worst = max(g["cond"] for g in result["per_group"])
    print(f"\n=== {label} ===")
    print(f"  M={result['M']}  J={result['J']}")
    for i, group in enumerate(result["per_group"], start=1):
        print(
            f"  group {i}: {group['n_items']:>5d} items  "
            f"sigma_min={group['sigma_min']:.3e}  cond={group['cond']:.3e}  "
            f"rank={group['numerical_rank']:>3d}  nonsingular={group['nonsingular']}"
        )
    print(
        f"  kappa={result['kappa']}  sum={result['kappa_sum']}  "
        f"bound=2M+2={result['kruskal_bound']}  satisfied={result['kruskal_satisfied']}"
    )
    return {
        "fit": label,
        "M": result["M"],
        "J": result["J"],
        "kappa_sum": result["kappa_sum"],
        "bound": result["kruskal_bound"],
        "satisfied": result["kruskal_satisfied"],
        "worst_cond": worst,
    }


def check_baseline_simulation() -> dict:
    """Baseline simulation condition, at its generating parameters."""
    from src.simulations.dgp import component_item_probs, generate_dataset
    from src.simulations.factors import BASELINE

    data = generate_dataset(BASELINE)
    P = component_item_probs(
        data.mu, data.theta, data.Q, data.condition.response_function
    )
    return _report("Baseline simulation (generating parameters)", P)


def check_fraction_subtraction() -> dict:
    """Fraction-subtraction fit, at the M used for the qualitative analysis."""
    from src.utils.datasets import download_frac_subtr_to_pandas

    X = download_frac_subtr_to_pandas("fraction.subtraction.data").to_numpy()
    Q = download_frac_subtr_to_pandas("fraction.subtraction.qmatrix").to_numpy()

    # Same configuration as the qualitative study: k-means init, Beta(2,2) priors.
    model = MoLA(
        Q=Q,
        n_components=FRAC_SUBTR_M,
        mu_prior=(2, 2),
        theta_prior=(2, 2),
        pi_prior=1,
        tol=1e-5,
        max_iter=50,
        pseudo_likelihood=False,
        init="k-means",
    )
    model.fit(X)
    return _report("Fraction subtraction (fitted parameters)", archetype_item_probs(model))


def check_ednet(n_learners: int, selected_m: int) -> dict:
    """EdNet-KT1 fit at one population-size scale point, at fitted parameters.

    Mirrors ``high_dim.run_evaluation``: the same learner split, with the
    population parameters fit on ``param_fit + m_select`` once ``M`` is fixed.
    ``M`` is taken from the paper's reported selection rather than re-run, since
    re-selecting it would refit the whole grid for a diagnostic that does not
    depend on how ``M`` was chosen.
    """
    from src.experiments.high_dim import (
        _build_response_matrix,
        build_item_index,
        split_learners,
    )
    from src.utils.ednet import sample_eligible_learners

    responses_df, Q, questions_df = sample_eligible_learners(
        n_learners=n_learners, min_responses=30, min_item_responses=30, seed=0
    )
    item_id_to_row = build_item_index(questions_df)
    param_fit_ids, m_select_ids, _ = split_learners(
        responses_df, param_frac=0.7, m_select_frac=0.2, seed=0
    )
    population_df = pd.concat(
        [
            responses_df[responses_df["user_id"].isin(param_fit_ids)],
            responses_df[responses_df["user_id"].isin(m_select_ids)],
        ],
        ignore_index=True,
    )
    X_param, _ = _build_response_matrix(
        population_df, population_df["user_id"].unique(), item_id_to_row, Q.shape[0]
    )

    model = MoLA(Q=Q, n_components=selected_m, max_iter=50)
    model.fit(X_param)
    return _report(
        f"EdNet-KT1, {n_learners:,} learners (fitted parameters)",
        archetype_item_probs(model),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-ednet", action="store_true", help="skip the EdNet fits")
    parser.add_argument(
        "--skip-frac-subtr", action="store_true", help="skip fraction subtraction (needs network)"
    )
    args = parser.parse_args(argv)

    rows = [check_baseline_simulation()]
    if not args.skip_frac_subtr:
        rows.append(check_fraction_subtraction())
    if not args.skip_ednet:
        for n_learners, selected_m in EDNET_SCALE_POINTS.items():
            rows.append(check_ednet(n_learners, selected_m))

    summary = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print(summary.to_string(index=False))

    if not summary["satisfied"].all():
        print("\nWARNING: the rank condition was not certified for every fit above.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
