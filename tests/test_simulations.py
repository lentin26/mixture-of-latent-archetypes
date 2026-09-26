"""Tests for the MoLA simulation study (``src/simulations``).

Run from the repo root:

    pytest tests/test_simulations.py -q
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.simulations import __main__ as sim_cli
from src.simulations.dgp import generate_dataset, q_matrix, resample_dataset
from src.simulations.factors import (
    BASELINE,
    FACTOR_LEVELS,
    SimulationCondition,
    ofat_design,
    replicate_seeds,
)
from src.simulations.metrics import paired_bootstrap_diff, pi_rmse, repeat_alignment_spread
from src.simulations.run import (
    default_m_grid,
    fit_mola,
    pick_m_by_criterion,
    run_condition,
    run_design,
    run_init_repeats,
    run_m_selection_evaluation,
    run_m_sweep,
    run_m_sweep_with_repeats,
    run_sample_repeats,
    split_holdout,
    summarize_m_sweep,
    summarize_results,
)

# A deliberately tiny condition so the full fit path runs in a fraction of a second.
SMALL = SimulationCondition(
    n_archetypes=2,
    n_learners=150,
    responses_per_learner=8,
    n_skills=4,
    n_iter=5,
    holdout_frac=0.2,
)

METRIC_KEYS = {
    "mu_rmse",
    "pi_mae",
    "pi_rmse",
    "theta_rmse",
    "assignment_ari",
    "holdout_auc",
    "holdout_brier",
    "holdout_nll",
    "n_em_iters",
    "final_nll",
    "train_time_sec",
    "init_time_sec",
    "fit_time_sec",
    "n_observations",
    "seed",
    "condition_id",
}


# --------------------------------------------------------------------------- #
# factors.py
# --------------------------------------------------------------------------- #
def test_ofat_design_includes_baseline_exactly_once():
    conditions = ofat_design()
    assert all(isinstance(c, SimulationCondition) for c in conditions)
    baseline_id = BASELINE.condition_id()
    assert sum(c.condition_id() == baseline_id for c in conditions) == 1
    # condition ids are unique (the design de-duplicates)
    ids = [c.condition_id() for c in conditions]
    assert len(ids) == len(set(ids))


def test_ofat_design_can_restrict_factors():
    conditions = ofat_design(factors=["n_archetypes"])
    # baseline + one condition per non-baseline level of n_archetypes
    varied = {c.n_archetypes for c in conditions}
    assert set(FACTOR_LEVELS["n_archetypes"]).issubset(varied)
    # nothing else moved off its baseline value
    assert {c.n_learners for c in conditions} == {BASELINE.n_learners}


def test_ofat_design_rejects_unknown_factor():
    with pytest.raises(KeyError):
        ofat_design(factors=["not_a_factor"])


def test_replicate_seeds_are_distinct_and_deterministic():
    seeds = replicate_seeds(0, 5)
    assert seeds[0] == 0
    assert len(seeds) == len(set(seeds)) == 5
    assert replicate_seeds(0, 5) == seeds
    assert replicate_seeds(7, 3)[0] == 7


def test_condition_id_unaffected_by_correctly_specified_n_components_fit():
    explicit = SimulationCondition(**{**SMALL.__dict__, "n_components_fit": SMALL.n_archetypes})
    assert explicit.condition_id() == SMALL.condition_id()
    assert explicit.resolved_n_components_fit() == SMALL.n_archetypes


def test_condition_id_reflects_misspecified_n_components_fit():
    misspecified = SimulationCondition(**{**SMALL.__dict__, "n_components_fit": SMALL.n_archetypes + 2})
    assert misspecified.condition_id() != SMALL.condition_id()
    assert misspecified.resolved_n_components_fit() == SMALL.n_archetypes + 2


# --------------------------------------------------------------------------- #
# dgp.py
# --------------------------------------------------------------------------- #
def test_generate_dataset_shapes_and_invariants():
    data = generate_dataset(SMALL, seed=0)
    n_items = SMALL.resolved_n_items()

    assert data.X.shape == (SMALL.n_learners, n_items)
    assert data.mu.shape == (SMALL.n_archetypes, SMALL.n_skills)
    assert data.pi.shape == (SMALL.n_archetypes,)
    assert data.theta.shape == (n_items, 1)
    assert data.z.shape == (SMALL.n_learners,)

    assert math.isclose(data.pi.sum(), 1.0, rel_tol=1e-9)
    assert set(np.unique(data.z)).issubset(set(range(SMALL.n_archetypes)))

    observed = np.isfinite(data.X)
    # every learner sees exactly responses_per_learner items
    assert np.all(observed.sum(axis=1) == SMALL.responses_per_learner)
    # observed responses are binary
    assert set(np.unique(data.X[observed])).issubset({0.0, 1.0})


def test_generate_dataset_is_seed_deterministic():
    a = generate_dataset(SMALL, seed=123)
    b = generate_dataset(SMALL, seed=123)
    c = generate_dataset(SMALL, seed=124)
    np.testing.assert_array_equal(np.nan_to_num(a.X, nan=-1), np.nan_to_num(b.X, nan=-1))
    assert not np.array_equal(
        np.nan_to_num(a.X, nan=-1), np.nan_to_num(c.X, nan=-1)
    )


def test_resample_dataset_keeps_population_and_varies_sample():
    population = generate_dataset(SMALL, seed=0)
    a = resample_dataset(population, seed=1)
    b = resample_dataset(population, seed=2)

    # the population is untouched...
    np.testing.assert_array_equal(a.mu, population.mu)
    np.testing.assert_array_equal(a.theta, population.theta)
    np.testing.assert_array_equal(a.pi, population.pi)
    np.testing.assert_array_equal(a.Q, population.Q)
    # ...but the sample (learners, responses) differs between resamples.
    assert not np.array_equal(a.z, b.z)
    assert not np.array_equal(np.nan_to_num(a.X, nan=-1), np.nan_to_num(b.X, nan=-1))


def test_resample_dataset_is_seed_deterministic():
    population = generate_dataset(SMALL, seed=0)
    a = resample_dataset(population, seed=5)
    b = resample_dataset(population, seed=5)
    np.testing.assert_array_equal(a.z, b.z)
    np.testing.assert_array_equal(np.nan_to_num(a.X, nan=-1), np.nan_to_num(b.X, nan=-1))


def test_misspecified_q_adds_a_column():
    correct = generate_dataset(SMALL, seed=0)
    mis = generate_dataset(
        SimulationCondition(**{**SMALL.__dict__, "model_specification": "misspecified"}),
        seed=0,
    )
    assert correct.Q_fit.shape[1] == SMALL.n_skills
    assert mis.Q_fit.shape[1] == SMALL.n_skills + 1


def test_q_matrix_columns_are_always_distinct():
    # a handful of seeds and both coverage modes -- a property test, not just one example
    for seed in range(5):
        for coverage in ("balanced", "uneven"):
            rng = np.random.default_rng(seed)
            Q = q_matrix(n_items=40, n_skills=SMALL.n_skills, coverage=coverage, rng=rng)
            assert np.unique(Q, axis=1).shape[1] == SMALL.n_skills


def test_q_matrix_balances_coverage_far_better_than_unweighted_sampling():
    rng_balanced = np.random.default_rng(0)
    Q_balanced = q_matrix(n_items=60, n_skills=20, coverage="balanced", rng=rng_balanced)

    rng_naive = np.random.default_rng(0)
    Q_naive = q_matrix(n_items=60, n_skills=20, coverage="balanced", rng=rng_naive, balance_power=0.0)

    cv_balanced = Q_balanced.sum(axis=0).std() / Q_balanced.sum(axis=0).mean()
    cv_naive = Q_naive.sum(axis=0).std() / Q_naive.sum(axis=0).mean()
    assert cv_balanced < cv_naive / 2  # substantially tighter, not just marginally


def test_q_matrix_raises_when_distinct_columns_are_structurally_impossible():
    # skills_per_item=(2, 2) with n_skills=2 forces every item to carry both
    # skills, so every column is identical no matter how many times we resample.
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="distinct skill columns"):
        q_matrix(n_items=2, n_skills=2, coverage="balanced", rng=rng, skills_per_item=(2, 2), max_resample=5)


# --------------------------------------------------------------------------- #
# metrics.py
# --------------------------------------------------------------------------- #
def test_pi_rmse_matches_rmse_formula_and_differs_from_mae():
    pi_true = np.array([0.5, 0.3, 0.2])
    pi_est = np.array([0.4, 0.35, 0.25])
    perm = np.array([0, 1, 2])
    expected_rmse = np.sqrt(np.mean((pi_true - pi_est) ** 2))
    expected_mae = np.mean(np.abs(pi_true - pi_est))
    assert np.isclose(pi_rmse(pi_true, pi_est, perm), expected_rmse)
    assert not np.isclose(expected_rmse, expected_mae)  # sanity: not accidentally equal here


def test_paired_bootstrap_diff_excludes_zero_for_a_clear_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.05, size=200)
    b = a - 0.5 + rng.normal(0.0, 0.01, size=200)  # clearly different, still paired to a

    result = paired_bootstrap_diff(a, b, n_boot=2000, seed=0)

    assert result["n"] == 200
    assert np.isclose(result["point_diff"], np.mean(a) - np.mean(b), atol=1e-9)
    assert result["excludes_zero"] is True
    assert result["ci_lo"] > 0  # a > b throughout, so the CI should sit above zero


def test_paired_bootstrap_diff_includes_zero_when_there_is_no_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.2, size=200)
    b = a.copy()  # identical -- the true difference is exactly zero

    result = paired_bootstrap_diff(a, b, n_boot=2000, seed=0)

    assert result["point_diff"] == 0.0
    assert result["excludes_zero"] is False


def test_paired_bootstrap_diff_drops_nan_rows_pairwise():
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    b = np.array([1.0, 2.0, 3.0, np.nan, 5.0])

    result = paired_bootstrap_diff(a, b, n_boot=100, seed=0)

    # rows 2 and 3 (0-indexed) each have a NaN in one array -- both dropped,
    # leaving only the 3 fully-paired rows.
    assert result["n"] == 3
    assert result["point_diff"] == 0.0


# --------------------------------------------------------------------------- #
# run.py
# --------------------------------------------------------------------------- #
def test_fit_mola_returns_model_and_nonnegative_init_time():
    data = generate_dataset(SMALL, seed=0)
    rng = np.random.default_rng(0)
    X_train, _hold_mask = split_holdout(data.X, SMALL.holdout_frac, rng)

    model, init_time_sec = fit_mola(data, X_train, rng)

    assert init_time_sec >= 0.0
    assert model.mu.shape == (SMALL.n_archetypes, SMALL.n_skills)
    assert len(model.nll_trace) > 0


def test_score_fit_splits_train_time_into_init_and_fit():
    metrics = run_condition(SMALL, seed=0)

    assert metrics["init_time_sec"] >= 0.0
    assert metrics["fit_time_sec"] >= 0.0
    # exact by construction (a subtraction, not an estimate)
    assert metrics["init_time_sec"] + metrics["fit_time_sec"] == metrics["train_time_sec"]


def test_run_condition_returns_expected_metrics():
    metrics = run_condition(SMALL, seed=0)

    assert METRIC_KEYS.issubset(metrics.keys())
    assert metrics["seed"] == 0
    assert metrics["condition_id"] == SMALL.condition_id()

    assert metrics["mu_rmse"] >= 0.0
    assert metrics["theta_rmse"] >= 0.0
    assert 0.0 <= metrics["holdout_brier"] <= 1.0
    assert 1 <= metrics["n_em_iters"] <= SMALL.n_iter

    ari = metrics["assignment_ari"]
    assert math.isnan(ari) or -1.0 <= ari <= 1.0
    auc = metrics["holdout_auc"]
    assert math.isnan(auc) or 0.0 <= auc <= 1.0


def test_run_condition_is_seed_deterministic():
    m1 = run_condition(SMALL, seed=42)
    m2 = run_condition(SMALL, seed=42)
    for key in ("mu_rmse", "theta_rmse", "final_nll", "holdout_brier"):
        assert m1[key] == pytest.approx(m2[key])


def test_run_condition_return_fit_exposes_model_and_data():
    result = run_condition(SMALL, seed=0, return_fit=True)
    assert result.metrics["condition_id"] == SMALL.condition_id()
    assert result.model.mu.shape == (SMALL.n_archetypes, SMALL.n_skills)
    assert result.data.X.shape == (SMALL.n_learners, SMALL.resolved_n_items())


def test_run_design_row_count_and_columns():
    conditions = [SMALL, SimulationCondition(**{**SMALL.__dict__, "n_archetypes": 3})]
    results = run_design(conditions=conditions, n_replications=2, base_seed=0)

    assert isinstance(results, pd.DataFrame)
    assert len(results) == len(conditions) * 2
    assert "condition_id" in results.columns
    assert results["seed"].nunique() == 2
    assert set(results["condition_id"]) == {c.condition_id() for c in conditions}


def test_summarize_results_aggregates_by_condition():
    results = run_design(conditions=[SMALL], n_replications=3, base_seed=0)
    summary = summarize_results(results)

    assert len(summary) == 1  # one condition_id
    for metric in ("mu_rmse", "holdout_brier"):
        assert f"{metric}_mean" in summary.columns
        assert f"{metric}_std" in summary.columns
        assert summary[f"{metric}_n"].iloc[0] == 3


def test_informed_init_rejects_misspecified_n_components_fit():
    misspecified = SimulationCondition(
        **{**SMALL.__dict__, "initialization": "informed", "n_components_fit": SMALL.n_archetypes + 1}
    )
    with pytest.raises(ValueError, match="initialization='informed'"):
        run_condition(misspecified, seed=0)


def test_kmeans_init_works_with_misspecified_n_components_fit():
    """Unlike "informed", k-means doesn't assume the correct component count
    -- it just clusters into however many it's asked for."""
    misspecified = SimulationCondition(
        **{**SMALL.__dict__, "initialization": "k-means", "n_components_fit": SMALL.n_archetypes + 1}
    )
    result = run_condition(misspecified, seed=0, return_fit=True)
    assert result.model.mu.shape[0] == SMALL.n_archetypes + 1


def test_run_init_repeats_holds_data_fixed_and_varies_only_init():
    random_only = SimulationCondition(**{**SMALL.__dict__, "initialization": "random"})
    results = run_init_repeats(random_only, data_seed=0, n_repeats=4)

    assert len(results) == 4
    # every repeat shares the exact same simulated dataset...
    assert all(r.data is results[0].data for r in results)
    # ...and the same condition, so the same condition_id.
    assert {r.metrics["condition_id"] for r in results} == {random_only.condition_id()}
    for r in results:
        assert r.model.mu.shape == (random_only.n_archetypes, random_only.n_skills)


def test_run_sample_repeats_holds_population_fixed_and_varies_the_sample():
    kmeans_cond = SimulationCondition(**{**SMALL.__dict__, "initialization": "k-means"})
    results = run_sample_repeats(kmeans_cond, population_seed=0, n_repeats=4)

    assert len(results) == 4
    # every repeat shares the exact same population (true mu/theta/pi/Q)...
    for r in results[1:]:
        np.testing.assert_array_equal(r.data.mu, results[0].data.mu)
        np.testing.assert_array_equal(r.data.theta, results[0].data.theta)
        np.testing.assert_array_equal(r.data.Q, results[0].data.Q)
    # ...but not the same drawn sample.
    assert not all(
        np.array_equal(np.nan_to_num(r.data.X, nan=-1), np.nan_to_num(results[0].data.X, nan=-1))
        for r in results[1:]
    )
    assert {r.metrics["condition_id"] for r in results} == {kmeans_cond.condition_id()}
    for r in results:
        assert r.model.mu.shape == (kmeans_cond.n_archetypes, kmeans_cond.n_skills)


def test_run_m_sweep_varies_fitted_m_holding_data_fixed():
    grid = [1, 2, 3]
    results = run_m_sweep(SMALL, n_components_grid=grid, seed=0)

    assert [r.model.mu.shape[0] for r in results] == grid
    assert [r.metrics["n_components_fit"] for r in results] == grid
    # the underlying simulated data shouldn't change with n_components_fit
    x0 = np.nan_to_num(results[0].data.X, nan=-1.0)
    for r in results[1:]:
        np.testing.assert_array_equal(np.nan_to_num(r.data.X, nan=-1.0), x0)
    # a correctly-specified grid point matches the plain condition_id --
    # run_m_sweep no longer overrides initialization (k-means now tolerates
    # a misspecified n_components_fit, so nothing needs forcing to "random").
    baseline_ix = grid.index(SMALL.n_archetypes)
    assert results[baseline_ix].metrics["condition_id"] == SMALL.condition_id()


# --------------------------------------------------------------------------- #
# choosing M: repeat_alignment_spread, run_m_sweep_with_repeats,
# default_m_grid, summarize_m_sweep, pick_m_by_criterion,
# run_m_selection_evaluation
# --------------------------------------------------------------------------- #
def test_repeat_alignment_spread_is_near_zero_for_identical_or_permuted_repeats():
    mu = np.array([[0.2, 0.8, 0.5], [0.7, 0.3, 0.4]])
    assert repeat_alignment_spread([mu, mu, mu]) < 1e-8
    # a row-permuted repeat is still "the same answer" once re-aligned
    assert repeat_alignment_spread([mu, mu[::-1]]) < 1e-8


def test_repeat_alignment_spread_is_nonzero_for_noisy_repeats():
    rng = np.random.default_rng(0)
    mu = np.array([[0.2, 0.8, 0.5], [0.7, 0.3, 0.4]])
    noisy = [mu + rng.normal(0.0, 0.05, mu.shape) for _ in range(5)]
    assert repeat_alignment_spread(noisy) > 0.01


def test_repeat_alignment_spread_requires_at_least_two_estimates():
    with pytest.raises(ValueError, match="at least 2"):
        repeat_alignment_spread([np.zeros((2, 3))])


def test_run_m_sweep_with_repeats_holds_data_fixed_across_m_and_repeats():
    grid = [1, 2, 3]
    n_repeats = 3
    results = run_m_sweep_with_repeats(SMALL, grid, data_seed=0, n_repeats=n_repeats)

    assert len(results) == len(grid) * n_repeats
    assert [r.metrics["n_components_fit"] for r in results] == [
        m for m in grid for _ in range(n_repeats)
    ]
    x0 = np.nan_to_num(results[0].data.X, nan=-1.0)
    for r in results[1:]:
        np.testing.assert_array_equal(np.nan_to_num(r.data.X, nan=-1.0), x0)


def test_default_m_grid_spans_below_and_above_true_m():
    grid = default_m_grid(4)
    assert min(grid) < 4 < max(grid)

    grid16 = default_m_grid(16)
    assert max(grid16) >= 16
    assert min(grid16) < 16

    # doesn't fall over at the smallest possible true M
    assert default_m_grid(1) == sorted(set(default_m_grid(1)))
    assert min(default_m_grid(1)) >= 1


def test_summarize_m_sweep_one_row_per_grid_point():
    grid = [1, 2, 3]
    kmeans_cond = SimulationCondition(**{**SMALL.__dict__, "initialization": "k-means"})
    random_cond = SimulationCondition(**{**SMALL.__dict__, "initialization": "random"})
    kmeans_results = run_m_sweep(kmeans_cond, grid, seed=0)
    random_repeat_results = run_m_sweep_with_repeats(random_cond, grid, data_seed=0, n_repeats=2)

    m_summary = summarize_m_sweep(kmeans_results, random_repeat_results)

    assert len(m_summary) == len(grid)
    assert m_summary["n_components_fit"].tolist() == grid
    for col in (
        "holdout_auc", "holdout_nll", "effective_n_archetypes",
        "effective_n_components", "weight_min", "closest_pair_distance", "mu_spread",
    ):
        assert col in m_summary.columns


def test_pick_m_by_criterion_returns_all_expected_keys():
    grid = [1, 2, 3]
    kmeans_cond = SimulationCondition(**{**SMALL.__dict__, "initialization": "k-means"})
    random_cond = SimulationCondition(**{**SMALL.__dict__, "initialization": "random"})
    m_summary = summarize_m_sweep(
        run_m_sweep(kmeans_cond, grid, seed=0),
        run_m_sweep_with_repeats(random_cond, grid, data_seed=0, n_repeats=2),
    )

    m_hats = pick_m_by_criterion(m_summary)

    expected_keys = {
        "predictive_auc", "predictive_nll", "stability",
        "separation_saturation", "separation_threshold",
        "mass_saturation", "mass_threshold",
    }
    assert set(m_hats) == expected_keys
    for m_hat in m_hats.values():
        assert isinstance(m_hat, int)


def test_run_m_selection_evaluation_row_count_and_columns():
    evaluation = run_m_selection_evaluation(
        n_archetypes_grid=(2, 3),
        separation_grid=("low",),
        seeds=(0,),
        n_repeats=2,
        n_grid_points=3,
    )

    n_criteria = 7
    assert len(evaluation) == 2 * 1 * 1 * n_criteria
    for col in (
        "n_archetypes", "archetype_separation", "seed",
        "criterion", "m_hat", "error", "exact_match",
    ):
        assert col in evaluation.columns
    assert set(evaluation["n_archetypes"]) == {2, 3}
    np.testing.assert_array_equal(
        evaluation["error"], evaluation["m_hat"] - evaluation["n_archetypes"]
    )


# --------------------------------------------------------------------------- #
# __main__.py  (CLI wiring — stubs the heavy fit)
# --------------------------------------------------------------------------- #
def test_cli_parser_reads_options():
    args = sim_cli._build_parser().parse_args(
        ["--design", "ofat", "--factors", "n_archetypes", "n_skills",
         "--replications", "3", "--limit", "4", "--summarize"]
    )
    assert args.design == "ofat"
    assert args.factors == ["n_archetypes", "n_skills"]
    assert args.replications == 3
    assert args.limit == 4
    assert args.summarize is True


def test_cli_main_writes_csv_and_respects_limit(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_run_design(conditions, n_replications, base_seed):
        seen["n_conditions"] = len(conditions)
        seen["n_replications"] = n_replications
        return pd.DataFrame(
            {"condition_id": ["c0"], "seed": [0], "mu_rmse": [0.1], "holdout_auc": [0.9]}
        )

    monkeypatch.setattr(sim_cli, "run_design", fake_run_design)

    out = tmp_path / "sim.csv"
    rc = sim_cli.main(
        ["--design", "ofat", "--limit", "2", "--replications", "1", "--out", str(out)]
    )

    assert rc == 0
    assert seen["n_conditions"] == 2
    assert seen["n_replications"] == 1
    assert out.exists()
    assert len(pd.read_csv(out)) == 1

    # the aggregated table is written alongside the per-run table by default
    summary_out = tmp_path / "sim_summary.csv"
    assert summary_out.exists()
    summary = pd.read_csv(summary_out)
    assert "condition_id" in summary.columns
    assert "mu_rmse_mean" in summary.columns


def test_cli_main_respects_explicit_summary_out(tmp_path, monkeypatch):
    def fake_run_design(conditions, n_replications, base_seed):
        return pd.DataFrame(
            {"condition_id": ["c0", "c0"], "seed": [0, 1], "mu_rmse": [0.1, 0.3]}
        )

    monkeypatch.setattr(sim_cli, "run_design", fake_run_design)

    summary_out = tmp_path / "agg" / "summary.csv"
    rc = sim_cli.main(["--design", "baseline", "--summary-out", str(summary_out)])

    assert rc == 0
    assert summary_out.exists()
    summary = pd.read_csv(summary_out)
    assert len(summary) == 1
    assert summary["mu_rmse_mean"].iloc[0] == pytest.approx(0.2)
    assert summary["mu_rmse_n"].iloc[0] == 2


def test_cli_main_baseline_design_runs_single_condition(tmp_path, monkeypatch):
    captured = {}

    def fake_run_design(conditions, n_replications, base_seed):
        captured["ids"] = [c.condition_id() for c in conditions]
        return pd.DataFrame({"condition_id": ["c0"], "seed": [0], "mu_rmse": [0.1]})

    monkeypatch.setattr(sim_cli, "run_design", fake_run_design)
    sim_cli.main(["--design", "baseline", "--replications", "1"])
    assert captured["ids"] == [BASELINE.condition_id()]
