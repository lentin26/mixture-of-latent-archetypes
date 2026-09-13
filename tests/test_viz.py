"""Tests for src/simulations/viz.py (assumed vs. recovered component plots).

Run from the repo root:

    pytest tests/test_viz.py -q
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; no display required

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.simulations.dgp import generate_dataset
from src.simulations.factors import SimulationCondition, ofat_design
from src.simulations.run import run_condition, run_design, run_init_repeats, run_m_sweep
from src.simulations.viz import (
    METRIC_LABELS,
    align_recovered_components,
    plot_component_recovery,
    plot_component_recovery_distribution,
    plot_components,
    plot_m_sensitivity,
    plot_ofat_sensitivity,
    plot_recovery_scatter,
    plot_setup_diagnostics,
    plot_theta_recovery,
)

RNG = np.random.default_rng(0)
MU_TRUE = np.clip(RNG.uniform(0.1, 0.9, size=(3, 5)), 0.05, 0.95)
# a lightly perturbed, row-shuffled copy stands in for a "recovered" mu
MU_EST = np.clip(MU_TRUE[[2, 0, 1]] + RNG.normal(0, 0.03, size=(3, 5)), 0.01, 0.99)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #
def test_align_recovered_components_recovers_row_permutation():
    _, _, perm, n_matched = align_recovered_components(MU_TRUE, MU_EST)
    assert n_matched == 3
    # MU_EST was built as MU_TRUE[[2, 0, 1]], so the match should invert that
    np.testing.assert_array_equal(perm, [1, 2, 0])


def test_align_recovered_components_handles_extra_estimated_component():
    mu_est = np.vstack([MU_EST, RNG.uniform(0, 1, size=(1, 5))])
    mu_t, mu_e, perm, n_matched = align_recovered_components(MU_TRUE, mu_est)
    assert n_matched == 3  # min(3 true, 4 est)
    assert perm.size == 3


def test_align_recovered_components_handles_extra_skill_column():
    mu_est = np.hstack([MU_EST, RNG.uniform(0, 1, size=(3, 1))])  # misspecified Q
    _, _, perm, n_matched = align_recovered_components(MU_TRUE, mu_est)
    assert n_matched == 3
    assert perm.size == 3


# --------------------------------------------------------------------------- #
# plot_component_recovery
# --------------------------------------------------------------------------- #
def test_recovery_grid_from_arrays_has_one_panel_per_component():
    fig = plot_component_recovery(mu_true=MU_TRUE, mu_est=MU_EST)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 3
    # each panel draws the assumed and the recovered profile
    assert len(visible[0].get_lines()) >= 2
    ax0 = visible[0]
    assert ax0.get_xlabel() == "skill index"
    assert ax0.get_ylim()[0] < 0.05 and ax0.get_ylim()[1] > 0.95


def test_recovery_overlay_uses_single_axes():
    fig = plot_component_recovery(mu_true=MU_TRUE, mu_est=MU_EST, layout="overlay")
    assert len(fig.axes) == 1
    assert len(fig.axes[0].get_lines()) >= 6  # 3 components x (assumed + recovered)


def test_recovery_requires_arrays_or_result():
    with pytest.raises(ValueError):
        plot_component_recovery()


def test_recovery_rejects_unknown_layout():
    with pytest.raises(ValueError):
        plot_component_recovery(mu_true=MU_TRUE, mu_est=MU_EST, layout="spiral")


def test_recovery_tolerates_extra_estimated_skill_column():
    mu_est = np.hstack([MU_EST, RNG.uniform(0, 1, size=(3, 1))])
    fig = plot_component_recovery(mu_true=MU_TRUE, mu_est=mu_est)
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 3


def test_recovery_saves_file(tmp_path):
    out = tmp_path / "rec.png"
    plot_component_recovery(mu_true=MU_TRUE, mu_est=MU_EST, save=out)
    assert out.exists() and out.stat().st_size > 0


# --------------------------------------------------------------------------- #
# plot_components / plot_recovery_scatter
# --------------------------------------------------------------------------- #
def test_plot_components_returns_axes_with_a_line_per_component():
    ax = plot_components(MU_TRUE, weights=[0.5, 0.3, 0.2], skill_labels=list("ABCDE"))
    assert isinstance(ax, plt.Axes)
    assert len(ax.get_lines()) == 3
    assert ax.get_ylabel().startswith("proficiency")


def test_plot_recovery_scatter_returns_axes():
    ax = plot_recovery_scatter(mu_true=MU_TRUE, mu_est=MU_EST)
    assert isinstance(ax, plt.Axes)
    assert len(ax.collections) == 3  # one scatter per component


# --------------------------------------------------------------------------- #
# end-to-end from a real fit
# --------------------------------------------------------------------------- #
def test_recovery_from_simulation_result():
    cond = SimulationCondition(
        n_archetypes=2, n_learners=150, responses_per_learner=8, n_skills=4, n_iter=5
    )
    result = run_condition(cond, seed=0, return_fit=True)
    fig = plot_component_recovery(result)
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 2
    # titles carry the pi_true -> pi_est annotation when a result is passed
    assert "pi" in [ax for ax in fig.axes if ax.get_visible()][0].get_title()


# --------------------------------------------------------------------------- #
# plot_component_recovery_distribution (Estimation Stability)
# --------------------------------------------------------------------------- #
SMALL_VIZ = SimulationCondition(
    n_archetypes=2, n_learners=150, responses_per_learner=8, n_skills=4, n_iter=5,
    initialization="random",
)


def test_recovery_distribution_from_arrays_has_one_panel_per_component():
    # 5 repeats: MU_TRUE with independent noise + row shuffles, standing in for
    # 5 different EM initializations recovering the same true components.
    mu_ests = [
        np.clip(MU_TRUE[RNG.permutation(3)] + RNG.normal(0, 0.03, size=MU_TRUE.shape), 0.01, 0.99)
        for _ in range(5)
    ]
    fig = plot_component_recovery_distribution(mu_true=MU_TRUE, mu_ests=mu_ests)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 3
    # 5 thin repeat lines + 1 bold median + 1 assumed line, per panel
    assert len(visible[0].get_lines()) == 7


def test_recovery_distribution_requires_arrays_or_results():
    with pytest.raises(ValueError):
        plot_component_recovery_distribution()


def test_recovery_distribution_from_run_init_repeats():
    results = run_init_repeats(SMALL_VIZ, data_seed=0, n_repeats=4)
    fig = plot_component_recovery_distribution(results)
    assert len([ax for ax in fig.axes if ax.get_visible()]) == SMALL_VIZ.n_archetypes


# --------------------------------------------------------------------------- #
# plot_ofat_sensitivity (Robustness to Data Sparsity / Computational Scalability)
# --------------------------------------------------------------------------- #
def test_ofat_sensitivity_one_panel_per_factor_numeric_and_categorical():
    conditions = ofat_design(
        baseline=SMALL_VIZ, factors=["n_learners", "archetype_separation"]
    )
    results = run_design(conditions, n_replications=2, base_seed=0)

    fig = plot_ofat_sensitivity(
        results,
        factors=["n_learners", "archetype_separation"],
        metric="mu_rmse",
        baseline=SMALL_VIZ,
        log_x=True,
    )
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 2


def test_ofat_sensitivity_can_plot_train_time_for_scalability():
    conditions = ofat_design(baseline=SMALL_VIZ, factors=["n_learners"])
    results = run_design(conditions, n_replications=2, base_seed=0)

    fig = plot_ofat_sensitivity(
        results, factors=["n_learners"], metric="train_time_sec", baseline=SMALL_VIZ,
    )
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 1


def test_ofat_sensitivity_log_y_sets_log_yscale():
    conditions = ofat_design(baseline=SMALL_VIZ, factors=["n_learners"])
    results = run_design(conditions, n_replications=2, base_seed=0)

    fig = plot_ofat_sensitivity(
        results, factors=["n_learners"], metric="train_time_sec", baseline=SMALL_VIZ,
        log_x=True, log_y=True,
    )
    ax = [a for a in fig.axes if a.get_visible()][0]
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"


def test_ofat_sensitivity_uses_nice_labels_by_default():
    conditions = ofat_design(baseline=SMALL_VIZ, factors=["n_learners"])
    results = run_design(conditions, n_replications=2, base_seed=0)

    fig = plot_ofat_sensitivity(
        results, factors=["n_learners"], metric="mu_rmse", baseline=SMALL_VIZ,
    )
    ax = [a for a in fig.axes if a.get_visible()][0]
    assert ax.get_title() == METRIC_LABELS["n_learners"]
    assert ax.get_ylabel() == METRIC_LABELS["mu_rmse"]
    assert METRIC_LABELS["mu_rmse"] in fig._suptitle.get_text()
    # a raw snake_case name should never leak into the figure
    assert "n_learners" not in ax.get_title()


def test_ofat_sensitivity_labels_override_the_default():
    conditions = ofat_design(baseline=SMALL_VIZ, factors=["n_learners"])
    results = run_design(conditions, n_replications=2, base_seed=0)

    fig = plot_ofat_sensitivity(
        results, factors=["n_learners"], metric="mu_rmse", baseline=SMALL_VIZ,
        labels={"n_learners": "Custom Label"},
    )
    ax = [a for a in fig.axes if a.get_visible()][0]
    assert ax.get_title() == "Custom Label"


# --------------------------------------------------------------------------- #
# plot_m_sensitivity (Sensitivity to the Number of Archetypes)
# --------------------------------------------------------------------------- #
def test_m_sensitivity_one_panel_per_metric():
    results = run_m_sweep(SMALL_VIZ, n_components_grid=[1, 2, 3], seed=0)
    fig = plot_m_sensitivity(
        results,
        true_n_archetypes=SMALL_VIZ.n_archetypes,
        metrics=("mu_rmse", "effective_n_archetypes"),
    )
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 2


def test_m_sensitivity_uses_nice_labels_by_default():
    results = run_m_sweep(SMALL_VIZ, n_components_grid=[1, 2, 3], seed=0)
    fig = plot_m_sensitivity(
        results, true_n_archetypes=SMALL_VIZ.n_archetypes, metrics=("mu_rmse",),
    )
    ax = [a for a in fig.axes if a.get_visible()][0]
    assert ax.get_title() == METRIC_LABELS["mu_rmse"]
    assert ax.get_ylabel() == METRIC_LABELS["mu_rmse"]
    assert ax.get_xlabel() == METRIC_LABELS["n_components_fit"]


def test_m_sensitivity_labels_override_the_default():
    results = run_m_sweep(SMALL_VIZ, n_components_grid=[1, 2, 3], seed=0)
    fig = plot_m_sensitivity(
        results, true_n_archetypes=SMALL_VIZ.n_archetypes, metrics=("mu_rmse",),
        labels={"mu_rmse": "Custom Metric Name"},
    )
    ax = [a for a in fig.axes if a.get_visible()][0]
    assert ax.get_title() == "Custom Metric Name"


# --------------------------------------------------------------------------- #
# plot_setup_diagnostics / plot_theta_recovery
# --------------------------------------------------------------------------- #
def test_setup_diagnostics_has_six_panels_from_real_data():
    data = generate_dataset(SMALL_VIZ, seed=0)
    fig = plot_setup_diagnostics(data)
    assert len(fig.axes) == 6


def test_setup_diagnostics_users_per_archetype_matches_realized_counts():
    # z is a plain (not stratified) draw from pi -- real cohorts sample with
    # ordinary noise around the population mixture, so the plot should show
    # the actual realized bincount, not an idealized exact match to pi.
    data = generate_dataset(SMALL_VIZ, seed=0)
    fig = plot_setup_diagnostics(data)
    bar_heights = [p.get_height() for p in fig.axes[0].patches]
    np.testing.assert_allclose(sorted(bar_heights), sorted(np.bincount(data.z)))


def test_theta_recovery_from_result():
    cond = SimulationCondition(
        n_archetypes=2, n_learners=150, responses_per_learner=8, n_skills=4, n_iter=5
    )
    result = run_condition(cond, seed=0, return_fit=True)
    ax = plot_theta_recovery(result)
    n_items = result.data.theta.shape[0]
    assert len(ax.collections[0].get_offsets()) == n_items


def test_theta_recovery_from_arrays():
    theta_true = np.array([0.2, 0.5, 0.8])
    theta_est = np.array([0.25, 0.45, 0.75])
    ax = plot_theta_recovery(theta_true=theta_true, theta_est=theta_est)
    assert len(ax.collections[0].get_offsets()) == 3


def test_theta_recovery_requires_result_or_arrays():
    with pytest.raises(ValueError):
        plot_theta_recovery()
