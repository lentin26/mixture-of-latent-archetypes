"""Tests for the anchoring experiment (``src/experiments/anchoring``).

Run from the repo root:

    pytest tests/test_anchoring.py -q

Two things here are load-bearing. First, ``AnchoredMoLA`` has to actually hold
the anchored difficulties fixed -- a clamp that silently stopped working would
leave every "anchored" result identical to the unanchored one, and the
experiment would report that anchoring does nothing. Second, the oracle
translation removal has to recover a *planted* translation, since the sign
conventions relating the model's item easiness to the paper's difficulty
``b = -logit(theta)`` are easy to get backwards and a flipped sign still
produces plausible-looking numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.experiments.anchoring import (
    AnchoredMoLA,
    anchor_items_for,
    between_archetype_agreement,
    remove_translation,
    within_archetype_auc,
)
from src.simulations.dgp import generate_dataset
from src.simulations.factors import BASELINE


@pytest.fixture(scope="module")
def data():
    return generate_dataset(BASELINE, seed=0)


def _q_star(Q):
    return Q / Q.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# anchor set construction
# --------------------------------------------------------------------------- #

def test_full_anchor_set_spans_the_skill_space(data):
    items = anchor_items_for(data.Q_fit, "full", np.random.default_rng(0))
    Q_A = _q_star(data.Q_fit)[items]
    assert items.size == data.Q_fit.shape[1]
    assert np.linalg.matrix_rank(Q_A) == data.Q_fit.shape[1]


def test_partial_anchor_set_is_rank_deficient(data):
    n_skills = data.Q_fit.shape[1]
    items = anchor_items_for(data.Q_fit, "partial", np.random.default_rng(0))
    rank = np.linalg.matrix_rank(_q_star(data.Q_fit)[items])
    assert rank == n_skills // 2 < n_skills


def test_no_anchors_for_the_none_level(data):
    assert anchor_items_for(data.Q_fit, "none", np.random.default_rng(0)) is None


def test_raises_when_a_skill_has_no_pure_item():
    # Every item taps both skills, so no single-attribute anchor exists.
    Q = np.array([[1, 1], [1, 1], [1, 1]], dtype=float)
    with pytest.raises(ValueError, match="no pure item"):
        anchor_items_for(Q, "full", np.random.default_rng(0))


def test_unknown_level_is_rejected(data):
    with pytest.raises(ValueError, match="unknown anchor level"):
        anchor_items_for(data.Q_fit, "sideways", np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# the clamp
# --------------------------------------------------------------------------- #

def test_anchored_difficulties_are_held_exactly_fixed(data):
    items = anchor_items_for(data.Q_fit, "full", np.random.default_rng(0))
    target = data.theta.ravel()[items]
    model = AnchoredMoLA(
        Q=data.Q_fit,
        n_components=BASELINE.n_archetypes,
        max_iter=10,
        anchor_items=items,
        anchor_easiness=target,
    )
    model.fit(data.X)
    np.testing.assert_allclose(model.theta.ravel()[items], target, rtol=0, atol=1e-12)


def test_the_clamp_pins_only_the_anchors(data):
    """The clamp must pin the anchors and nothing else -- freezing the whole
    difficulty vector would make the comparison meaningless, and it would look
    like anchoring works perfectly."""
    items = anchor_items_for(data.Q_fit, "partial", np.random.default_rng(0))
    model = AnchoredMoLA(
        Q=data.Q_fit,
        n_components=BASELINE.n_archetypes,
        max_iter=10,
        anchor_items=items,
        anchor_easiness=data.theta.ravel()[items],
    )
    model.fit(data.X)
    # Only an anchored item can land exactly on its generating value.
    pinned = np.flatnonzero(np.isclose(model.theta.ravel(), data.theta.ravel(), atol=1e-12))
    np.testing.assert_array_equal(np.sort(pinned), np.sort(items))


def test_anchoring_reduces_the_fitted_translation(data):
    """The substantive check: a spanning anchor set should collapse the
    translation, which is what the anchor proposition claims."""
    Q_star = _q_star(data.Q_fit)
    spreads = {}
    for level in ("none", "full"):
        items = anchor_items_for(data.Q_fit, level, np.random.default_rng(0))
        model = AnchoredMoLA(
            Q=data.Q_fit,
            n_components=BASELINE.n_archetypes,
            max_iter=BASELINE.n_iter,
            anchor_items=items,
            anchor_easiness=None if items is None else data.theta.ravel()[items],
        )
        model.fit(data.X)
        _, _, delta = remove_translation(
            model.mu, model.theta, data.mu, data.theta, Q_star
        )
        spreads[level] = delta.max() - delta.min()
    assert spreads["full"] < 0.5 * spreads["none"]


def test_anchor_arguments_must_be_paired(data):
    with pytest.raises(ValueError, match="together"):
        AnchoredMoLA(Q=data.Q_fit, n_components=2, anchor_items=[0, 1])


def test_unanchored_subclass_matches_the_base_model(data):
    """With no anchors, the subclass must be a no-op relative to MoLA itself."""
    from src.mola import MoLA

    kwargs = dict(Q=data.Q_fit, n_components=BASELINE.n_archetypes, max_iter=5)
    base = MoLA(**kwargs)
    base.fit(data.X)
    sub = AnchoredMoLA(**kwargs)
    sub.fit(data.X)
    np.testing.assert_allclose(sub.mu, base.mu, rtol=1e-10)
    np.testing.assert_allclose(sub.theta, base.theta, rtol=1e-10)


# --------------------------------------------------------------------------- #
# oracle translation removal
# --------------------------------------------------------------------------- #

def test_planted_translation_is_recovered(data):
    """Apply a known translation to the true parameters, then check the
    least-squares fit recovers it and undoes it.

    This is the test that catches a sign flip between the model's item
    easiness and the paper's difficulty ``b = -logit(theta)``: a flipped sign
    leaves ``mu`` correct but drives the difficulty error up rather than down.
    """
    rng = np.random.default_rng(1)
    Q_star = _q_star(data.Q_fit)
    n_skills = data.mu.shape[1]
    delta_true = rng.normal(0.0, 0.4, size=n_skills)

    eta = np.log(data.mu / (1 - data.mu))
    mu_shift = 1.0 / (1.0 + np.exp(-(eta + delta_true[None, :])))
    # b -> b + Q* delta, i.e. logit(theta) -> logit(theta) - Q* delta.
    logit_theta = np.log(data.theta.ravel() / (1 - data.theta.ravel()))
    theta_shift = (1.0 / (1.0 + np.exp(-(logit_theta - Q_star @ delta_true)))).reshape(-1, 1)

    mu_c, theta_c, delta_hat = remove_translation(
        mu_shift, theta_shift, data.mu, data.theta, Q_star
    )
    np.testing.assert_allclose(delta_hat, delta_true, atol=1e-8)
    np.testing.assert_allclose(mu_c, data.mu, atol=1e-8)
    np.testing.assert_allclose(theta_c.ravel(), data.theta.ravel(), atol=1e-8)


def test_translation_removal_is_a_no_op_at_the_truth(data):
    Q_star = _q_star(data.Q_fit)
    mu_c, theta_c, delta = remove_translation(
        data.mu, data.theta, data.mu, data.theta, Q_star
    )
    np.testing.assert_allclose(delta, 0.0, atol=1e-8)
    np.testing.assert_allclose(mu_c, data.mu, atol=1e-8)


# --------------------------------------------------------------------------- #
# the invariance metrics
# --------------------------------------------------------------------------- #

def test_within_auc_is_perfect_at_the_truth(data):
    perm = np.arange(data.mu.shape[0])
    assert within_archetype_auc(data.mu, data.mu, perm) == pytest.approx(1.0)


def test_within_auc_degrades_under_a_large_attribute_offset(data):
    """A per-attribute translation large relative to the profile's own
    attribute contrast should scramble the within-archetype ordering -- this is
    the quantity the translation indeterminacy costs."""
    rng = np.random.default_rng(2)
    perm = np.arange(data.mu.shape[0])
    eta = np.log(data.mu / (1 - data.mu))
    big = rng.normal(0.0, 4.0, size=data.mu.shape[1])
    scrambled = 1.0 / (1.0 + np.exp(-(eta + big[None, :])))
    assert within_archetype_auc(data.mu, scrambled, perm) < 0.9


def test_between_archetype_agreement_is_invariant_to_translation(data):
    """The complement: the same offset must leave between-archetype contrasts
    untouched, exactly as the interpretation remark claims."""
    rng = np.random.default_rng(3)
    perm = np.arange(data.mu.shape[0])
    eta = np.log(data.mu / (1 - data.mu))
    shifted = 1.0 / (1.0 + np.exp(-(eta + rng.normal(0.0, 4.0, size=data.mu.shape[1])[None, :])))
    r, sign = between_archetype_agreement(data.mu, shifted, perm)
    assert r == pytest.approx(1.0, abs=1e-10)
    assert sign == pytest.approx(1.0)
