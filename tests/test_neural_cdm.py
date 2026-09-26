"""Tests for the NeuralCDM (EduCDM NCDM) wrapper (``src/cdm/neural_cdm.py``).

Run from the repo root:

    pytest tests/test_neural_cdm.py -q

Requires ``torch``/``EduCDM`` (optional dependencies -- see
requirements.txt); skipped automatically if unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("EduCDM")

from src.cdm.neural_cdm import NeuralCDM  # noqa: E402


def _tiny_setup(seed=0):
    rng = np.random.default_rng(seed)
    n_skills, n_items, n_learners = 3, 8, 5
    item2knowledge = (rng.random((n_items, n_skills)) < 0.5).astype(float)
    item2knowledge[item2knowledge.sum(1) == 0, 0] = 1.0

    rows = []
    for u in range(n_learners):
        for it in range(n_items):
            rows.append({"user_row": u, "item_row": it, "correct": float(rng.random() < 0.5)})
    train_df = pd.DataFrame(rows)
    return n_skills, n_items, n_learners, item2knowledge, train_df


def test_fit_returns_self_and_records_train_time():
    n_skills, n_items, n_learners, item2knowledge, train_df = _tiny_setup()
    model = NeuralCDM(n_skills, n_items, n_learners, epoch=1, batch_size=8)

    result = model.fit(train_df, item2knowledge)

    assert result is model
    assert model.train_time_sec is not None
    assert model.train_time_sec >= 0


def test_fit_freezes_every_population_parameter():
    n_skills, n_items, n_learners, item2knowledge, train_df = _tiny_setup()
    model = NeuralCDM(n_skills, n_items, n_learners, epoch=1, batch_size=8)
    model.fit(train_df, item2knowledge)

    for param in model.model.ncdm_net.parameters():
        assert not param.requires_grad


def test_profile_learner_does_not_change_any_population_parameter():
    n_skills, n_items, n_learners, item2knowledge, train_df = _tiny_setup()
    model = NeuralCDM(n_skills, n_items, n_learners, epoch=1, profile_epochs=5, batch_size=8)
    model.fit(train_df, item2knowledge)

    net = model.model.ncdm_net
    before = {name: p.clone() for name, p in net.named_parameters()}

    profile_df = pd.DataFrame({"item_row": [0, 1, 2, 3], "correct": [1.0, 0.0, 1.0, 1.0]})
    model.profile_learner(profile_df, item2knowledge)

    for name, param in net.named_parameters():
        assert torch.equal(before[name], param), f"{name} changed during profile_learner"
        assert param.grad is None or torch.all(param.grad == 0), f"{name} accumulated a nonzero gradient"


def test_profile_learner_returns_shape_and_predict_item_proba_is_valid_probability():
    n_skills, n_items, n_learners, item2knowledge, train_df = _tiny_setup()
    model = NeuralCDM(n_skills, n_items, n_learners, epoch=1, profile_epochs=5, batch_size=8)
    model.fit(train_df, item2knowledge)

    profile_df = pd.DataFrame({"item_row": [0, 1, 2], "correct": [1.0, 0.0, 1.0]})
    stu_emb = model.profile_learner(profile_df, item2knowledge)
    assert stu_emb.shape == (1, n_skills)
    assert not stu_emb.requires_grad  # detached

    preds = model.predict_item_proba(stu_emb, np.array([4, 5, 6]), item2knowledge)
    assert preds.shape == (3,)
    assert np.all((preds >= 0) & (preds <= 1))


def test_population_learner_embedding_matches_frozen_table_row():
    n_skills, n_items, n_learners, item2knowledge, train_df = _tiny_setup()
    model = NeuralCDM(n_skills, n_items, n_learners, epoch=1, batch_size=8)
    model.fit(train_df, item2knowledge)

    emb = model.population_learner_embedding(2)
    expected = model.model.ncdm_net.student_emb.weight[2:3]
    assert torch.equal(emb, expected)


def test_random_state_makes_fit_deterministic():
    # Without a seed, weight init/shuffle order depend on torch's global
    # RNG state, which in turn depends on whatever ran earlier in the
    # process -- not reproducible from this call's arguments alone.
    # random_state fixes that: two fresh models built and fit the same
    # way, with the same random_state, must end up bitwise identical.
    n_skills, n_items, n_learners, item2knowledge, train_df = _tiny_setup()

    a = NeuralCDM(n_skills, n_items, n_learners, epoch=2, batch_size=4, random_state=0)
    a.fit(train_df, item2knowledge)
    b = NeuralCDM(n_skills, n_items, n_learners, epoch=2, batch_size=4, random_state=0)
    b.fit(train_df, item2knowledge)

    for pa, pb in zip(a.model.ncdm_net.parameters(), b.model.ncdm_net.parameters()):
        assert torch.equal(pa, pb)


def test_random_state_makes_profile_learner_deterministic():
    n_skills, n_items, n_learners, item2knowledge, train_df = _tiny_setup()
    model = NeuralCDM(n_skills, n_items, n_learners, epoch=1, profile_epochs=5, batch_size=8, random_state=0)
    model.fit(train_df, item2knowledge)
    profile_df = pd.DataFrame({"item_row": [0, 1, 2], "correct": [1.0, 0.0, 1.0]})

    emb_a = model.profile_learner(profile_df, item2knowledge, seed=1)
    emb_b = model.profile_learner(profile_df, item2knowledge, seed=1)

    assert torch.equal(emb_a, emb_b)
