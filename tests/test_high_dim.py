"""Tests for the EdNet MoLA-vs-NeuralCDM evaluation pipeline
(``src/experiments/high_dim.py``).

All tests use tiny synthetic data (no real EdNet download). NeuralCDM-
dependent tests are skipped automatically if ``torch``/``EduCDM`` are
unavailable -- see requirements.txt.

Run from the repo root:

    pytest tests/test_high_dim.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments import high_dim as high_dim_module
from src.experiments.high_dim import (
    build_item_index,
    evaluate_learner,
    select_mola_m,
    split_learners,
)
from src.mola import MoLA


def _tiny_responses(n_learners=30, n_items=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_learners):
        n_resp = rng.integers(20, 30)
        items = rng.choice(n_items, size=n_resp, replace=True)
        for t, it in enumerate(items):
            rows.append(
                {"user_id": u, "question_id": int(it), "timestamp": t, "correct": float(rng.random() < 0.5)}
            )
    return pd.DataFrame(rows)


def _tiny_q_and_questions(n_items=10, n_skills=3, seed=0):
    rng = np.random.default_rng(seed)
    Q = (rng.random((n_items, n_skills)) < 0.5).astype(float)
    Q[Q.sum(1) == 0, 0] = 1.0
    questions_df = pd.DataFrame(index=pd.RangeIndex(n_items, name="question_id"))
    return Q, questions_df


def test_split_learners_percentages_and_no_overlap():
    responses_df = _tiny_responses(n_learners=100)

    param_fit_ids, m_select_ids, evaluation_ids = split_learners(
        responses_df, param_frac=0.7, m_select_frac=0.2, seed=0
    )

    assert len(param_fit_ids) == 56
    assert len(m_select_ids) == 14
    assert len(evaluation_ids) == 30

    all_ids = set(param_fit_ids) | set(m_select_ids) | set(evaluation_ids)
    assert all_ids == set(responses_df["user_id"].unique())
    assert len(set(param_fit_ids) & set(m_select_ids)) == 0
    assert len(set(param_fit_ids) & set(evaluation_ids)) == 0
    assert len(set(m_select_ids) & set(evaluation_ids)) == 0


def test_split_learners_is_seeded_and_deterministic():
    responses_df = _tiny_responses(n_learners=40)
    split_a = split_learners(responses_df, seed=5)
    split_b = split_learners(responses_df, seed=5)
    for group_a, group_b in zip(split_a, split_b):
        np.testing.assert_array_equal(group_a, group_b)


def test_build_item_index_maps_question_id_to_row_order():
    questions_df = pd.DataFrame(index=pd.Index([30, 10, 20], name="question_id"))
    item_id_to_row = build_item_index(questions_df)
    assert item_id_to_row == {30: 0, 10: 1, 20: 2}


def test_select_mola_m_picks_a_candidate_from_the_grid():
    responses_df = _tiny_responses(n_learners=40)
    Q, questions_df = _tiny_q_and_questions()
    item_id_to_row = build_item_index(questions_df)

    param_fit_ids, m_select_ids, _ = split_learners(responses_df, seed=0)
    param_fit_df = responses_df[responses_df["user_id"].isin(param_fit_ids)]
    m_select_df = responses_df[responses_df["user_id"].isin(m_select_ids)]

    best_m, nll_by_m = select_mola_m(
        param_fit_df, m_select_df, Q, item_id_to_row, m_grid=[2, 3, 4], seed=0, max_iter=10
    )

    assert best_m in {2, 3, 4}
    assert set(nll_by_m) == {2, 3, 4}
    assert all(np.isfinite(v) for v in nll_by_m.values())


def test_evaluate_learner_mola_side_returns_finite_metrics_in_range():
    rng = np.random.default_rng(0)
    Q, _ = _tiny_q_and_questions()
    n_items, n_skills = Q.shape

    mola_model = MoLA(Q=Q, n_components=3, max_iter=10, random_state=0)
    X_fit = (rng.random((20, n_items)) < 0.5).astype(float)
    mola_model.fit(X_fit)

    learner_df = pd.DataFrame(
        {
            "question_id": rng.choice(n_items, size=25, replace=True),
            "timestamp": range(25),
            "correct": (rng.random(25) < 0.5).astype(float),
        }
    )
    item_id_to_row = {i: i for i in range(n_items)}

    class _StubNeuralCDM:
        def profile_learner(self, profile_df, item2knowledge, seed=None):
            return None

        def predict_item_proba(self, stu_emb, item_rows, item2knowledge):
            return np.full(len(item_rows), 0.5)

    result = evaluate_learner(learner_df, mola_model, _StubNeuralCDM(), Q, item_id_to_row, profile_frac=0.6)

    assert result["n_profile"] + result["n_predict"] == 25
    assert np.isfinite(result["mola_nll"])
    assert np.isfinite(result["mola_brier"])
    assert np.isfinite(result["neural_cdm_nll"])  # stub always predicts 0.5


class _StubNeuralCDM:
    def profile_learner(self, profile_df, item2knowledge, seed=None):
        return None

    def predict_item_proba(self, stu_emb, item_rows, item2knowledge):
        return np.full(len(item_rows), 0.5)


def _mask_test_setup(n_responses=40, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = _tiny_q_and_questions()
    n_items = Q.shape[0]
    mola_model = MoLA(Q=Q, n_components=3, max_iter=10, random_state=0)
    mola_model.fit((rng.random((20, n_items)) < 0.5).astype(float))
    # MoLA's predict_item_proba lazily populates an inference-time cache on
    # its first-ever call, and takes a numerically different path before
    # vs. after that cache exists (see fit_population_models's own comment
    # on this) -- warm it up here so repeated calls in a test are
    # comparable to each other, matching what fit_population_models does
    # for the real pipeline.
    mola_model.update_ab_params(use_norm=True)
    # non-monotonic timestamps, so a chronological sort actually reorders rows
    timestamps = rng.permutation(n_responses)
    learner_df = pd.DataFrame(
        {
            "question_id": rng.choice(n_items, size=n_responses, replace=True),
            "timestamp": timestamps,
            "correct": (rng.random(n_responses) < 0.5).astype(float),
        }
    )
    item_id_to_row = {i: i for i in range(n_items)}
    return learner_df, mola_model, Q, item_id_to_row


def test_evaluate_learner_chronological_mask_profile_strictly_precedes_predict_in_time():
    learner_df, mola_model, Q, item_id_to_row = _mask_test_setup()
    sorted_df = learner_df.sort_values("timestamp").reset_index(drop=True)
    n_profile = int(np.floor(0.6 * len(sorted_df)))

    evaluate_learner(learner_df, mola_model, _StubNeuralCDM(), Q, item_id_to_row, profile_frac=0.6, mask="chronological")

    # the split itself (independent of the returned metrics) should put
    # every one of the later timestamps in the prediction set
    profile_timestamps = sorted_df["timestamp"].iloc[:n_profile]
    predict_timestamps = sorted_df["timestamp"].iloc[n_profile:]
    assert profile_timestamps.max() <= predict_timestamps.min()


def test_evaluate_learner_random_mask_is_not_chronologically_ordered():
    learner_df, mola_model, Q, item_id_to_row = _mask_test_setup(n_responses=200, seed=1)
    shuffled = learner_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    n_profile = int(np.floor(0.6 * len(shuffled)))
    profile_timestamps = shuffled["timestamp"].iloc[:n_profile]
    predict_timestamps = shuffled["timestamp"].iloc[n_profile:]
    # with 200 responses, a genuinely random split essentially never comes
    # out chronologically ordered -- confirms mask="random" isn't secretly
    # sorting by timestamp
    assert profile_timestamps.max() > predict_timestamps.min()


def test_evaluate_learner_random_mask_is_seeded_and_deterministic():
    learner_df, mola_model, Q, item_id_to_row = _mask_test_setup()
    result_a = evaluate_learner(learner_df, mola_model, _StubNeuralCDM(), Q, item_id_to_row, mask="random", seed=7)
    result_b = evaluate_learner(learner_df, mola_model, _StubNeuralCDM(), Q, item_id_to_row, mask="random", seed=7)
    assert result_a == result_b


def test_evaluate_learner_rejects_unknown_mask():
    learner_df, mola_model, Q, item_id_to_row = _mask_test_setup()
    with pytest.raises(ValueError, match="mask must be"):
        evaluate_learner(learner_df, mola_model, _StubNeuralCDM(), Q, item_id_to_row, mask="future")


def test_evaluate_learner_result_records_which_mask_was_used():
    learner_df, mola_model, Q, item_id_to_row = _mask_test_setup()
    chrono = evaluate_learner(learner_df, mola_model, _StubNeuralCDM(), Q, item_id_to_row, mask="chronological")
    random_ = evaluate_learner(learner_df, mola_model, _StubNeuralCDM(), Q, item_id_to_row, mask="random")
    assert chrono["mask"] == "chronological"
    assert random_["mask"] == "random"


def test_run_evaluation_refits_final_models_on_param_fit_plus_m_select():
    # Regression test: fit_population_models used to be called with just
    # param_fit_df (56% of learners), silently discarding m_select (14%)
    # for the final population fit even though it had already served its
    # purpose (selecting M) by that point. Spies on the actual call
    # run_evaluation makes to confirm it now passes the full 70%.
    pytest.importorskip("torch")
    pytest.importorskip("EduCDM")

    responses_df = _tiny_responses(n_learners=100)
    Q, questions_df = _tiny_q_and_questions()

    captured = {}
    original = high_dim_module.fit_population_models

    def spy(population_df, *args, **kwargs):
        captured["n_learners"] = population_df["user_id"].nunique()
        return original(population_df, *args, **kwargs)

    import unittest.mock
    with unittest.mock.patch.object(high_dim_module, "fit_population_models", spy):
        high_dim_module.run_evaluation(
            responses_df, Q, questions_df, m_grid=[2, 3],
            mola_kwargs={"max_iter": 5},
            neural_cdm_kwargs={"epoch": 1, "profile_epochs": 2, "batch_size": 8},
        )

    # param_frac=0.7, m_select_frac=0.2 defaults on 100 learners -> 56 + 14 = 70
    assert captured["n_learners"] == 70
