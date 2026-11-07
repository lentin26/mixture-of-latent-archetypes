"""Quantifies what anchor items buy, for the quantities the translation
indeterminacy actually costs.

Run from the repository root so that ``import src...`` resolves:

    python -m src.experiments.anchoring

    # fewer datasets, for a quick check
    python -m src.experiments.anchoring --n-datasets 3

Remark "Translation indeterminacy" in the paper establishes that
``eta_m -> eta_m + delta``, ``b -> b + Q* delta`` leaves every response
probability unchanged, and the companion remark sorts the consequences into
three cases: comparisons *between* archetypes on a fixed attribute are
invariant, comparisons *between attributes* within one archetype are not, and
the posterior responsibilities -- hence the partition of learners into
archetypes -- are invariant because they depend on the parameters only through
the response probabilities.

Those are claims about the population. This experiment checks that the fitted
model behaves as they predict, and that fixing anchor-item difficulties
recovers what the indeterminacy costs:

* ``none``    -- ordinary fit; the scale is set only by the Beta priors.
* ``partial`` -- difficulties fixed for pure items covering half the skills,
                 so ``rank(Q*_A) = K/2``. The anchor proposition predicts a
                 residual ``(K - r)``-dimensional family of translations, i.e.
                 partial recovery.
* ``full``    -- difficulties fixed for one pure item per skill, giving
                 ``Q*_A = I_K``. The proposition predicts the indeterminacy is
                 eliminated outright.

The baseline data-generating process embeds a ``K x K`` identity submatrix
among the items, so a pure item exists for every skill; this is the one
setting in the paper where the single-attribute anchor recipe is available.
On a real Q-matrix it generally is not -- the fraction-subtraction Q-matrix,
for instance, has three single-skill items covering two distinct skills, and
spanning ``R^K`` there requires multi-attribute anchors.

Metrics, chosen to separate the three cases above rather than to report one
aggregate:

``mu_rmse`` / ``theta_rmse``
    Parameter recovery, for comparison against the oracle correction that
    removes the fitted translation by least squares (reported as the
    ``ls_corrected`` row). Anchoring should reach comparable accuracy without
    needing the ground truth the oracle correction consumes.
``within_auc``
    Per archetype, how well the recovered profile separates the skills in that
    archetype's own elevated-proficiency group from the rest, as an AUC over
    the ``K`` attributes, averaged across archetypes. This is the
    within-archetype, across-attribute comparison the remark says is not
    identified, in the form a practitioner would use it: which skills is this
    archetype strong on?
``between_r`` / ``between_sign``
    Across all archetype pairs and attributes, the correlation and sign
    agreement between recovered and true ``eta_mk - eta_m'k``. The remark
    predicts these are already high without anchors.
``ari`` / ``mean_max_gamma``
    Adjusted Rand index of the induced hard partition against the true
    archetype labels, and the mean of ``max_m gamma_nm``. The remark predicts
    the partition is unaffected by anchoring.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from src.mola import MoLA
from src.simulations.dgp import generate_dataset
from src.simulations.factors import BASELINE
from src.simulations.metrics import align_components, assignment_ari, mu_rmse, theta_rmse

ANCHOR_LEVELS = ("none", "partial", "full")


def anchor_items_for(Q: np.ndarray, level: str, rng: np.random.Generator) -> np.ndarray | None:
    """Pure items giving ``rank(Q*_A) = K`` (``full``) or ``K // 2`` (``partial``).

    Returns ``None`` for ``level == "none"``. Raises if the Q-matrix lacks a
    pure item for some skill, since the levels above are defined in terms of
    them; see the module docstring on why that holds for the baseline
    condition and not in general.
    """
    if level == "none":
        return None
    n_skills = Q.shape[1]
    single = np.flatnonzero(Q.sum(axis=1) == 1)
    by_skill = {}
    for j in single:
        by_skill.setdefault(int(Q[j].argmax()), j)
    missing = sorted(set(range(n_skills)) - set(by_skill))
    if missing:
        raise ValueError(f"no pure item for skills {missing}; cannot build a {level} anchor set")

    skills = np.array(sorted(by_skill))
    if level == "partial":
        skills = rng.choice(skills, size=n_skills // 2, replace=False)
    elif level != "full":
        raise ValueError(f"unknown anchor level {level!r}")
    return np.array([by_skill[int(k)] for k in sorted(skills)], dtype=int)


def _logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def remove_translation(mu_est, theta_est, mu_true, theta_true, Q_star):
    """Least-squares removal of the fitted translation, as reported in the paper.

    Solves the stacked problem matching ``eta_hat - eta`` to ``delta`` over all
    ``(m, k)`` and ``b_hat - b`` to ``(Q* delta)_j`` over all ``j``, then
    returns the corrected ``(mu, theta)``. This consumes the ground truth, so
    it is an oracle diagnostic, not an estimator -- it is the reference the
    anchored fit is meant to match without that information.
    """
    n_components, n_skills = mu_est.shape
    eta_resid = (_logit(mu_est) - _logit(mu_true)).ravel()
    # theta is easiness, so b = -logit(theta); the difficulty residual is
    # b_hat - b = logit(theta) - logit(theta_hat).
    b_resid = (_logit(theta_true).ravel() - _logit(theta_est).ravel())

    design_eta = np.tile(np.eye(n_skills), (n_components, 1))
    design = np.vstack([design_eta, Q_star])
    target = np.concatenate([eta_resid, b_resid])
    delta, *_ = np.linalg.lstsq(design, target, rcond=None)

    mu_corr = 1.0 / (1.0 + np.exp(-(_logit(mu_est) - delta[None, :])))
    # b = -logit(theta), so the corrected difficulty b_hat - Q* delta is the
    # corrected easiness sigma(logit(theta_hat) + Q* delta).
    theta_corr = 1.0 / (1.0 + np.exp(-(_logit(theta_est).ravel() + Q_star @ delta)))
    return mu_corr, theta_corr.reshape(-1, 1), delta


def within_archetype_auc(mu_true: np.ndarray, mu_est: np.ndarray, perm: np.ndarray) -> float:
    """Mean AUC for separating each archetype's own elevated skills, from the
    recovered profile alone.

    The generating profiles place each archetype's own skill group at
    ``0.5 + delta`` and the rest at ``0.5 - delta``, so "own group" is read off
    the true profile by thresholding at its midpoint. An archetype whose
    recovered profile ranks its own skills above the others scores 1; one whose
    attribute ordering has been scrambled by an attribute-specific offset
    scores near 0.5.
    """
    scores = []
    for m_true, m_est in enumerate(perm):
        truth = mu_true[m_true]
        label = (truth > 0.5 * (truth.min() + truth.max())).astype(int)
        if label.min() == label.max():
            continue
        scores.append(roc_auc_score(label, mu_est[m_est]))
    return float(np.mean(scores)) if scores else float("nan")


def between_archetype_agreement(mu_true, mu_est, perm) -> tuple[float, float]:
    """Correlation and sign agreement of ``eta_mk - eta_m'k``, over all
    archetype pairs and attributes."""
    eta_true, eta_est = _logit(mu_true), _logit(mu_est)
    true_d, est_d = [], []
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            true_d.append(eta_true[i] - eta_true[j])
            est_d.append(eta_est[perm[i]] - eta_est[perm[j]])
    true_d = np.concatenate(true_d)
    est_d = np.concatenate(est_d)
    r = float(pearsonr(true_d, est_d)[0])
    sign = float(np.mean(np.sign(true_d) == np.sign(est_d)))
    return r, sign


def _evaluate(mu_true, theta_true, mu_est, theta_est, post, z_true):
    perm = align_components(mu_true, mu_est)
    r, sign = between_archetype_agreement(mu_true, mu_est, perm)
    return {
        "mu_rmse": mu_rmse(mu_true, mu_est),
        "theta_rmse": theta_rmse(theta_true, theta_est),
        "within_auc": within_archetype_auc(mu_true, mu_est, perm),
        "between_r": r,
        "between_sign": sign,
        "ari": assignment_ari(z_true, post, perm),
        "mean_max_gamma": float(np.mean(post.max(axis=1))),
    }


def translation_spread(mu_est, theta_est, mu_true, theta_true, Q_star) -> float:
    """Range of the fitted per-attribute offsets, in log-odds.

    Reported alongside ``within_auc`` because the two together say whether the
    indeterminacy is large enough to matter: an attribute-specific offset
    scrambles an archetype's attribute ordering only to the extent that its
    spread is comparable to the archetype's own attribute contrast, which the
    generating profiles fix at ``2 delta`` in probability.
    """
    _, _, delta = remove_translation(mu_est, theta_est, mu_true, theta_true, Q_star)
    return float(delta.max() - delta.min())


def run_one(seed: int, condition=BASELINE) -> list[dict]:
    """Fit every anchor level to one simulated dataset and score each."""
    data = generate_dataset(condition, seed=seed)
    Q_star = data.Q_fit / data.Q_fit.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    rows = []

    for level in ANCHOR_LEVELS:
        items = anchor_items_for(data.Q_fit, level, rng)
        # The generating easiness is on the same scale as the model's own
        # ``theta``, so the true values pass straight through. Using the true
        # values is what makes this anchoring rather than a reparameterization:
        # they are known independently of the responses being fit.
        easiness = None if items is None else data.theta.ravel()[items]
        model = MoLA(
            Q=data.Q_fit,
            n_components=condition.n_archetypes,
            max_iter=condition.n_iter,
            init=condition.initialization,
            anchor_items=items,
            anchor_easiness=easiness,
        )
        model.fit(data.X)
        post = np.asarray(model.predict_proba(data.X))

        row = {
            "separation": condition.archetype_separation,
            "seed": seed,
            "anchors": level,
            "n_anchors": 0 if items is None else items.size,
            "delta_spread": translation_spread(
                model.mu, model.theta, data.mu, data.theta, Q_star
            ),
        }
        row.update(_evaluate(data.mu, data.theta, model.mu, model.theta, post, data.z))
        rows.append(row)

        if level == "none":
            mu_c, theta_c, delta = remove_translation(
                model.mu, model.theta, data.mu, data.theta, Q_star
            )
            row_c = {
                "separation": condition.archetype_separation,
                "seed": seed,
                "anchors": "ls_corrected",
                "n_anchors": 0,
                "delta_spread": float(delta.max() - delta.min()),
            }
            row_c.update(_evaluate(data.mu, data.theta, mu_c, theta_c, post, data.z))
            rows.append(row_c)

    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-datasets", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument(
        "--separations",
        default="low,medium,high",
        help="comma-separated archetype-separation levels to sweep (baseline: medium)",
    )
    parser.add_argument("--out", default=None, help="optional CSV path for the per-fit rows")
    args = parser.parse_args(argv)

    separations = [s.strip() for s in args.separations.split(",") if s.strip()]
    rows = []
    for separation in separations:
        condition = replace(BASELINE, archetype_separation=separation)
        for i in range(args.n_datasets):
            seed = args.seed0 + i
            print(f"[{separation}] dataset {i + 1}/{args.n_datasets} (seed {seed})...", flush=True)
            rows.extend(run_one(seed, condition))

    df = pd.DataFrame(rows)
    order = ["none", "ls_corrected", "partial", "full"]
    cols = [
        "mu_rmse",
        "theta_rmse",
        "within_auc",
        "between_r",
        "between_sign",
        "ari",
        "mean_max_gamma",
        "delta_spread",
    ]
    summary = (
        df.groupby(["separation", "anchors"])[cols]
        .mean()
        .reindex(pd.MultiIndex.from_product([separations, order], names=["separation", "anchors"]))
        .round(4)
    )

    print(f"\n{'=' * 100}")
    print(
        f"Anchoring comparison, baseline condition, {args.n_datasets} datasets per separation level"
    )
    print(f"{'=' * 100}")
    print(summary.to_string())
    print("\nAnchors per level:", df.groupby("anchors")["n_anchors"].max().reindex(order).to_dict())

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nPer-fit rows written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
