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
from src.simulations.dgp import generate_dataset
from src.simulations.factors import (
    BASELINE,
    FACTOR_LEVELS,
    SimulationCondition,
    ofat_design,
    replicate_seeds,
)
from src.simulations.run import run_condition, run_design, summarize_results

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
    "theta_rmse",
    "assignment_ari",
    "holdout_auc",
    "holdout_brier",
    "holdout_nll",
    "n_em_iters",
    "final_nll",
    "train_time_sec",
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


def test_misspecified_q_adds_a_column():
    correct = generate_dataset(SMALL, seed=0)
    mis = generate_dataset(
        SimulationCondition(**{**SMALL.__dict__, "model_specification": "misspecified"}),
        seed=0,
    )
    assert correct.Q_fit.shape[1] == SMALL.n_skills
    assert mis.Q_fit.shape[1] == SMALL.n_skills + 1


# --------------------------------------------------------------------------- #
# run.py
# --------------------------------------------------------------------------- #
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
