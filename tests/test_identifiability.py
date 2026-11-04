"""Tests for the identifiability diagnostics (``src/mola/identifiability``).

Run from the repo root:

    pytest tests/test_identifiability.py -q

The point of these tests is that the certificate has to be *falsifiable*: a
check that reports "nonsingular" for every input would certify the
mixture-identifiability hypothesis vacuously. The degenerate and near-degenerate
cases below are therefore the load-bearing ones -- they pin down that collapsing
two archetypes is detected as rank deficiency, and that merely bringing them
close is reported as ill-conditioning rather than failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.mola import MoLA
from src.mola.identifiability import (
    archetype_item_probs,
    group_minor,
    rank_certificate,
    roundrobin_partition,
)

M = 4
J = 60


@pytest.fixture
def probs() -> np.ndarray:
    """``(M, J)`` archetype-conditional success probabilities, well separated."""
    rng = np.random.default_rng(0)
    return rng.uniform(0.05, 0.95, size=(M, J))


# --------------------------------------------------------------------------- #
# partitioning
# --------------------------------------------------------------------------- #
def test_roundrobin_partition_covers_every_item_once():
    groups = roundrobin_partition(J, n_groups=3)
    assert len(groups) == 3
    assert sorted(np.concatenate(groups).tolist()) == list(range(J))
    assert all(len(g) == 20 for g in groups)


def test_roundrobin_partition_handles_indivisible_item_counts():
    groups = roundrobin_partition(20, n_groups=3)
    assert sorted(len(g) for g in groups) == [6, 7, 7]
    assert sorted(np.concatenate(groups).tolist()) == list(range(20))


# --------------------------------------------------------------------------- #
# the certificate on separated archetypes
# --------------------------------------------------------------------------- #
def test_separated_archetypes_are_certified(probs):
    result = rank_certificate(probs, seed=0)
    assert result["kappa"] == [M, M, M]
    assert result["kappa_sum"] == 3 * M
    assert result["kruskal_bound"] == 2 * M + 2
    assert result["kruskal_satisfied"]
    assert result["all_certified"]


def test_minor_is_square_and_rank_M(probs):
    out = group_minor(probs[:, :20], seed=0)
    assert out["numerical_rank"] == M
    assert out["nonsingular"]
    assert out["sigma_min"] > 0.0


# --------------------------------------------------------------------------- #
# falsifiability: the checks that make the certificate mean something
# --------------------------------------------------------------------------- #
def test_duplicate_archetype_is_detected_as_rank_deficient(probs):
    """Two archetypes sharing a profile give M_i two identical rows."""
    degenerate = probs.copy()
    degenerate[1] = degenerate[0]

    result = rank_certificate(degenerate, seed=0)
    assert not result["all_certified"]
    assert not result["kruskal_satisfied"]
    assert result["kappa"] == [M - 1, M - 1, M - 1]
    for group in result["per_group"]:
        assert not group["nonsingular"]
        assert group["numerical_rank"] == M - 1


def test_near_duplicate_archetypes_are_ill_conditioned_not_singular(probs):
    """Separation shrinking toward zero should show up in the condition number."""
    conds = []
    for gap in (1e-2, 1e-4, 1e-6):
        near = probs.copy()
        near[1] = np.clip(near[0] + gap, 1e-6, 1 - 1e-6)
        result = rank_certificate(near, seed=0)
        assert result["all_certified"], f"gap={gap} should still be nonsingular"
        conds.append(max(g["cond"] for g in result["per_group"]))

    # Conditioning degrades monotonically as the archetypes converge.
    assert conds[0] < conds[1] < conds[2]


# --------------------------------------------------------------------------- #
# integration with a fitted model
# --------------------------------------------------------------------------- #
def test_archetype_item_probs_matches_model_inference_path():
    """``archetype_item_probs`` must agree with what the model scores with."""
    rng = np.random.default_rng(0)
    Q = (rng.random((15, 4)) < 0.4).astype(float)
    Q[Q.sum(axis=1) == 0, 0] = 1.0
    X = (rng.random((200, 15)) < 0.6).astype(float)

    model = MoLA(Q=Q, n_components=3, max_iter=5)
    model.fit(X)

    P = archetype_item_probs(model)
    assert P.shape == (3, 15)
    assert np.all((P > 0.0) & (P < 1.0))

    # Same quantity the predictive path builds, up to the clipping applied here.
    expected = 1.0 / (1.0 + np.exp(model.b.T - model.a.T))
    np.testing.assert_allclose(P, expected, atol=1e-10)


def test_certificate_runs_on_a_fitted_model():
    rng = np.random.default_rng(1)
    Q = (rng.random((30, 5)) < 0.4).astype(float)
    Q[Q.sum(axis=1) == 0, 0] = 1.0
    X = (rng.random((300, 30)) < 0.6).astype(float)

    model = MoLA(Q=Q, n_components=3, max_iter=5)
    model.fit(X)

    result = rank_certificate(archetype_item_probs(model), seed=0)
    assert result["M"] == 3 and result["J"] == 30
    assert len(result["per_group"]) == 3
    assert result["kappa_sum"] <= 3 * 3
