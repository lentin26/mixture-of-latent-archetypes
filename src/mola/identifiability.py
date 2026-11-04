"""
Diagnostics for the linear-independence hypothesis of the mixture-identifiability
proposition.

The proposition requires the items to admit a partition into three groups
$S_1, S_2, S_3$ such that, within each group, the $M$ archetype-conditional
response distributions

    (x) tensor_{j in S_i} Bernoulli(P_mj),      m = 1, ..., M

are linearly independent. Equivalently, each factor matrix M_i (shape
``(M, 2 ** |S_i|)``, rows indexed by archetype) has full row rank, i.e. Kruskal
rank ``kappa_i = M``. The hypothesis is certified at a given parameter value by
exhibiting one nonsingular ``M x M`` minor of each M_i.

Two things make a naive check useless here:

* ``M_i`` cannot be formed. It has ``2 ** |S_i|`` columns -- about a million for
  a 20-item group, and astronomically more for the high-dimensional fits. Only
  ``M`` of its columns are needed, so we select a well-conditioned subset
  instead of materializing the matrix.
* Its entries underflow. Each is a product of ``|S_i|`` probabilities, so a
  determinant computed on the raw values collapses to zero and the minor would
  be reported singular regardless of its actual rank. Entries are therefore
  accumulated in log space, and the minor is equilibrated by row and column
  scaling before its singular values are taken. Rank is invariant under
  positive diagonal scaling, so this changes the conditioning we can measure
  without changing the property being tested.
"""

import numpy as np
from scipy.linalg import qr, svdvals

__all__ = [
    "archetype_item_probs",
    "roundrobin_partition",
    "group_minor",
    "rank_certificate",
]


def archetype_item_probs(model, eps: float = 1e-12) -> np.ndarray:
    """
    ``(M, J)`` matrix of archetype-conditional success probabilities ``P_mj``.

    Taken from the model's own inference artifacts so that the diagnostic sees
    exactly the response probabilities the fitted model would use to score a
    learner, rather than a re-derivation of the item response function.
    """
    if getattr(model, "a", None) is None or getattr(model, "b", None) is None:
        model.update_ab_params(use_norm=True)
    P = 1.0 / (1.0 + np.exp(model.b.T - model.a.T))
    return np.clip(P, eps, 1.0 - eps)


def roundrobin_partition(n_items: int, n_groups: int = 3) -> list:
    """
    Assign item ``j`` to group ``j % n_groups``.

    Interleaving rather than slicing contiguous blocks keeps each group's skill
    coverage close to that of the full item set; contiguous blocks can strand
    every item loading on some skill in a single group, leaving the other groups
    uninformative about it.
    """
    idx = np.arange(n_items)
    return [idx[idx % n_groups == g] for g in range(n_groups)]


def _candidate_patterns(P_group: np.ndarray, n_random: int, rng) -> np.ndarray:
    """
    Pool of candidate response patterns for one group, as a ``(C, |S|)`` array.

    Seeded with each archetype's modal pattern -- the single most probable
    response vector under that archetype -- which is the natural candidate for
    separating it from the others, then padded with random patterns so the
    column selection has slack when modal patterns coincide across archetypes.
    """
    modal = (P_group > 0.5).astype(np.float64)
    pool = [modal]
    if n_random > 0:
        pool.append(rng.integers(0, 2, size=(n_random, P_group.shape[1])).astype(np.float64))
    return np.unique(np.vstack(pool), axis=0)


def group_minor(P_group: np.ndarray, n_random: int = 512, seed: int = 0) -> dict:
    """
    Select a well-conditioned ``M x M`` minor of one group's factor matrix.

    ``P_group`` is ``(M, |S_i|)``. Returns the minor's singular values together
    with the chosen response patterns.
    """
    M = P_group.shape[0]
    rng = np.random.default_rng(seed)
    patterns = _candidate_patterns(P_group, n_random, rng)

    # Column c of the factor matrix, in log space: the log-probability of
    # pattern c under each archetype.
    logP = np.log(P_group)
    log1mP = np.log1p(-P_group)
    L = (patterns @ logP.T + (1.0 - patterns) @ log1mP.T).T  # (M, C)

    # Row scaling in log space, so every row peaks at 1 after exponentiating.
    L = L - L.max(axis=1, keepdims=True)
    A = np.exp(L)

    # Column-pivoted QR orders columns by how much they extend the span, so the
    # leading M pivots are a well-conditioned subset -- the standard greedy
    # maximal-volume heuristic.
    _, _, piv = qr(A, mode="economic", pivoting=True)
    cols = piv[:M]
    minor = A[:, cols]

    # Column scaling, likewise rank-preserving, to finish equilibrating.
    minor = minor / np.abs(minor).max(axis=0, keepdims=True)

    sv = svdvals(minor)
    tol = sv[0] * M * np.finfo(float).eps
    return {
        "n_items": P_group.shape[1],
        "n_patterns_tried": patterns.shape[0],
        "sigma_min": float(sv[-1]),
        "sigma_max": float(sv[0]),
        "cond": float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf,
        "numerical_rank": int((sv > tol).sum()),
        "tol": float(tol),
        "nonsingular": bool(sv[-1] > tol),
        "patterns": patterns[cols],
    }


def rank_certificate(
    P: np.ndarray,
    groups=None,
    n_groups: int = 3,
    n_random: int = 512,
    seed: int = 0,
) -> dict:
    """
    Certify the independence hypothesis for a fitted parameter value.

    ``P`` is the ``(M, J)`` matrix of archetype-conditional success
    probabilities. Returns per-group diagnostics, the implied Kruskal ranks, and
    whether Kruskal's condition ``sum kappa_i >= 2M + 2`` is met.

    A group reported as ``nonsingular`` certifies ``kappa_i = M`` at this
    parameter value. A group reported singular is *not* proof that the
    hypothesis fails: it means no nonsingular minor was found among the patterns
    searched.
    """
    M, J = P.shape
    if groups is None:
        groups = roundrobin_partition(J, n_groups)

    per_group = [
        group_minor(P[:, g], n_random=n_random, seed=seed + i)
        for i, g in enumerate(groups)
    ]
    certified = [g["nonsingular"] for g in per_group]
    kappa = [M if c else per_group[i]["numerical_rank"] for i, c in enumerate(certified)]

    return {
        "M": M,
        "J": J,
        "groups": [np.asarray(g) for g in groups],
        "per_group": per_group,
        "kappa": kappa,
        "kappa_sum": int(sum(kappa)),
        "kruskal_bound": 2 * M + 2,
        "all_certified": bool(all(certified)),
        "kruskal_satisfied": bool(sum(kappa) >= 2 * M + 2),
    }
