"""Tests for src/simulations/viz.py (assumed vs. recovered component plots).

Run from the repo root:

    pytest tests/test_viz.py -q
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; no display required

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.simulations.dgp import generate_dataset
from src.simulations.factors import SimulationCondition, ofat_design
from src.simulations.run import run_condition, run_design, run_init_repeats, run_m_sweep
from src.simulations.viz import (
    CRITERION_LABELS,
    METRIC_LABELS,
    align_recovered_components,
    plot_component_recovery,
    plot_component_recovery_distribution,
    plot_components,
    plot_m_selection_recovery,
    plot_m_sensitivity,
    plot_ofat_sensitivity,
    plot_parameter_recovery,
    plot_recovery_scatter,
    plot_setup_diagnostics,
    plot_theta_pi_recovery_distribution,
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


def test_recovery_distribution_panel_titles_omit_stats():
    # sample size / spread are reported in text/a table now, not baked
    # into the panel title.
    results = run_init_repeats(SMALL_VIZ, data_seed=0, n_repeats=4)
    fig = plot_component_recovery_distribution(results)
    ax = [a for a in fig.axes if a.get_visible()][0]
    assert ax.get_title() == "component 0"


def test_recovery_distribution_xlabel_is_figure_level_not_per_panel():
    results = run_init_repeats(SMALL_VIZ, data_seed=0, n_repeats=4)
    fig = plot_component_recovery_distribution(results)
    visible = [a for a in fig.axes if a.get_visible()]
    assert fig.get_supxlabel() == "skill index"
    assert all(ax.get_xlabel() == "" for ax in visible)


def test_recovery_distribution_groups_overlays_in_shared_panels():
    from dataclasses import replace
    random_results = run_init_repeats(replace(SMALL_VIZ, initialization="random"), data_seed=0, n_repeats=3)
    kmeans_results = run_init_repeats(replace(SMALL_VIZ, initialization="k-means"), data_seed=0, n_repeats=3)

    fig = plot_component_recovery_distribution(
        groups={"Random": random_results, "K-means": kmeans_results},
        group_colors={"Random": "#d62728", "K-means": "#1f77b4"},
    )
    visible = [a for a in fig.axes if a.get_visible()]
    assert len(visible) == SMALL_VIZ.n_archetypes
    legend_labels = {t.get_text() for t in visible[0].get_legend().get_texts()}
    assert legend_labels == {"Random", "K-means", "assumed"}
    # 3 thin lines + 1 median per group (2 groups) + 1 assumed line
    assert len(visible[0].get_lines()) == (3 + 1) * 2 + 1


def test_recovery_distribution_groups_rejects_mismatched_true_mu():
    from dataclasses import replace
    group_a = run_init_repeats(SMALL_VIZ, data_seed=0, n_repeats=2)
    group_b = run_init_repeats(SMALL_VIZ, data_seed=1, n_repeats=2)  # different dataset -> different true mu
    with pytest.raises(ValueError, match="does not share the same true mu"):
        plot_component_recovery_distribution(groups={"A": group_a, "B": group_b})


# --------------------------------------------------------------------------- #
# plot_theta_pi_recovery_distribution
# --------------------------------------------------------------------------- #
def test_theta_pi_recovery_distribution_has_two_panels_with_legends():
    from dataclasses import replace
    random_results = run_init_repeats(replace(SMALL_VIZ, initialization="random"), data_seed=0, n_repeats=3)
    kmeans_results = run_init_repeats(replace(SMALL_VIZ, initialization="k-means"), data_seed=0, n_repeats=3)

    fig = plot_theta_pi_recovery_distribution(
        groups={"Random": random_results, "K-means": kmeans_results},
    )
    assert len(fig.axes) == 2
    for ax in fig.axes:
        legend_labels = {t.get_text() for t in ax.get_legend().get_texts()}
        assert legend_labels == {"Random", "K-means"}


def test_theta_pi_recovery_distribution_plots_difficulty_by_default():
    from dataclasses import replace
    random_results = run_init_repeats(replace(SMALL_VIZ, initialization="random"), data_seed=0, n_repeats=2)
    fig = plot_theta_pi_recovery_distribution(groups={"Random": random_results})
    theta_ax = fig.axes[0]
    assert "difficulty" in theta_ax.get_xlabel()
    offsets = np.asarray(theta_ax.collections[0].get_offsets())
    n_items = random_results[0].data.theta.shape[0]
    expected_x = np.concatenate([1.0 - r.data.theta.ravel() for r in random_results])
    assert len(offsets) == n_items * len(random_results)
    np.testing.assert_allclose(sorted(offsets[:, 0]), sorted(expected_x))


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


def test_ofat_sensitivity_sharey_puts_every_panel_on_one_axis():
    conditions = ofat_design(baseline=SMALL_VIZ, factors=["n_learners", "n_skills"])
    results = run_design(conditions, n_replications=2, base_seed=0)

    fig = plot_ofat_sensitivity(
        results, factors=["n_learners", "n_skills"], metric="train_time_sec",
        baseline=SMALL_VIZ, sharey=True,
    )
    axes = [a for a in fig.axes if a.get_visible()]
    assert axes[0].get_shared_y_axes().joined(axes[0], axes[1])


def test_ofat_sensitivity_not_shared_by_default():
    conditions = ofat_design(baseline=SMALL_VIZ, factors=["n_learners", "n_skills"])
    results = run_design(conditions, n_replications=2, base_seed=0)

    fig = plot_ofat_sensitivity(
        results, factors=["n_learners", "n_skills"], metric="train_time_sec",
        baseline=SMALL_VIZ,
    )
    axes = [a for a in fig.axes if a.get_visible()]
    assert not axes[0].get_shared_y_axes().joined(axes[0], axes[1])


def test_ofat_sensitivity_reference_line_draws_p1_power_law():
    conditions = ofat_design(baseline=SMALL_VIZ, factors=["n_learners"])
    results = run_design(conditions, n_replications=2, base_seed=0)

    fig = plot_ofat_sensitivity(
        results, factors=["n_learners"], metric="train_time_sec", baseline=SMALL_VIZ,
        log_x=True, log_y=True, reference_line="linear",
    )
    ax = [a for a in fig.axes if a.get_visible()][0]
    # errorbar (data) + one dashed reference line
    dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
    assert len(dashed) == 1
    x = dashed[0].get_xdata()
    y = dashed[0].get_ydata()
    # y = y[0] * (x / x[0]): a p=1 power law anchored at the first point
    np.testing.assert_allclose(y, y[0] * (np.asarray(x) / x[0]))
    assert ax.get_legend() is not None


def test_ofat_sensitivity_rejects_unknown_reference_line():
    conditions = ofat_design(baseline=SMALL_VIZ, factors=["n_learners"])
    results = run_design(conditions, n_replications=2, base_seed=0)
    with pytest.raises(ValueError, match="reference_line"):
        plot_ofat_sensitivity(
            results, factors=["n_learners"], metric="train_time_sec",
            baseline=SMALL_VIZ, reference_line="quadratic",
        )


# --------------------------------------------------------------------------- #
# plot_ofat_sensitivity: multi-metric (one row per metric)
# --------------------------------------------------------------------------- #
def _scalability_results():
    conditions = ofat_design(baseline=SMALL_VIZ, factors=["n_learners", "n_skills"])
    results = run_design(conditions, n_replications=2, base_seed=0)
    results = results.copy()
    results["time_per_iter_sec"] = results["train_time_sec"] / results["n_em_iters"]
    return results


def test_ofat_sensitivity_multi_metric_grid_shape_and_titles():
    results = _scalability_results()
    fig = plot_ofat_sensitivity(
        results, factors=["n_learners", "n_skills"],
        metric=["time_per_iter_sec", "n_em_iters"], baseline=SMALL_VIZ,
    )
    assert len(fig.axes) == 4  # 2 metrics x 2 factors
    # titles (factor names) only on the top row
    assert fig.axes[0].get_title() == METRIC_LABELS["n_learners"]
    assert fig.axes[1].get_title() == METRIC_LABELS["n_skills"]
    assert fig.axes[2].get_title() == ""
    assert fig.axes[3].get_title() == ""
    # y-labels (metric names) only on the left column, one per row
    assert fig.axes[0].get_ylabel() == METRIC_LABELS["time_per_iter_sec"]
    assert fig.axes[1].get_ylabel() == ""
    assert fig.axes[2].get_ylabel() == METRIC_LABELS["n_em_iters"]
    assert fig.axes[3].get_ylabel() == ""


def test_ofat_sensitivity_multi_metric_per_row_log_y_and_reference_line():
    results = _scalability_results()
    fig = plot_ofat_sensitivity(
        results, factors=["n_learners", "n_skills"],
        metric=["time_per_iter_sec", "n_em_iters"], baseline=SMALL_VIZ,
        log_x=True, log_y=[True, False], reference_line=["linear", None],
    )
    top_row, bottom_row = fig.axes[:2], fig.axes[2:]
    for ax in top_row:
        assert ax.get_yscale() == "log"
        assert any(ln.get_linestyle() == "--" for ln in ax.get_lines())
    for ax in bottom_row:
        assert ax.get_yscale() == "linear"
        assert not any(ln.get_linestyle() == "--" for ln in ax.get_lines())
    # legend only where a reference line was actually drawn (leftmost of the top row)
    assert fig.axes[0].get_legend() is not None
    assert fig.axes[2].get_legend() is None


def test_ofat_sensitivity_multi_metric_sharey_scoped_to_row_only():
    results = _scalability_results()
    fig = plot_ofat_sensitivity(
        results, factors=["n_learners", "n_skills"],
        metric=["time_per_iter_sec", "n_em_iters"], baseline=SMALL_VIZ, sharey=True,
    )
    top_left, top_right, bottom_left, _ = fig.axes
    assert top_left.get_shared_y_axes().joined(top_left, top_right)
    assert not top_left.get_shared_y_axes().joined(top_left, bottom_left)


def test_ofat_sensitivity_multi_metric_rejects_mismatched_sequence_lengths():
    results = _scalability_results()
    with pytest.raises(ValueError, match="log_y"):
        plot_ofat_sensitivity(
            results, factors=["n_learners"], metric=["time_per_iter_sec", "n_em_iters"],
            baseline=SMALL_VIZ, log_y=[True],
        )
    with pytest.raises(ValueError, match="reference_line"):
        plot_ofat_sensitivity(
            results, factors=["n_learners"], metric=["time_per_iter_sec", "n_em_iters"],
            baseline=SMALL_VIZ, reference_line=["linear", "linear", None],
        )


def test_ofat_sensitivity_single_element_metric_list_matches_string():
    results = _scalability_results()
    fig_str = plot_ofat_sensitivity(
        results, factors=["n_learners"], metric="n_em_iters", baseline=SMALL_VIZ,
    )
    fig_list = plot_ofat_sensitivity(
        results, factors=["n_learners"], metric=["n_em_iters"], baseline=SMALL_VIZ,
    )
    assert len(fig_str.axes) == len(fig_list.axes) == 1
    assert fig_str.axes[0].get_title() == fig_list.axes[0].get_title()


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
# plot_m_selection_recovery (choosing M: criteria evaluation)
# --------------------------------------------------------------------------- #
def _tiny_m_selection_evaluation() -> pd.DataFrame:
    """A small synthetic evaluation table -- bypasses the heavy
    run_m_selection_evaluation driver, matching how other viz tests build
    data directly rather than running a full pipeline."""
    rows = []
    rng = np.random.default_rng(0)
    for sep in ("low", "high"):
        for true_m in (2, 4):
            for seed in (0, 1):
                for criterion in ("predictive_auc", "stability"):
                    rows.append({
                        "n_archetypes": true_m,
                        "archetype_separation": sep,
                        "seed": seed,
                        "criterion": criterion,
                        "m_hat": true_m + int(rng.integers(-1, 2)),
                    })
    df = pd.DataFrame(rows)
    df["error"] = df["m_hat"] - df["n_archetypes"]
    df["exact_match"] = df["error"] == 0
    return df


def test_m_selection_recovery_one_panel_per_separation_level():
    evaluation = _tiny_m_selection_evaluation()
    fig = plot_m_selection_recovery(evaluation)
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 2


def test_m_selection_recovery_uses_nice_labels_by_default():
    evaluation = _tiny_m_selection_evaluation()
    fig = plot_m_selection_recovery(evaluation)
    ax = fig.axes[0]
    assert ax.get_ylabel() == METRIC_LABELS["m_hat"]
    assert ax.get_xlabel() == METRIC_LABELS["n_archetypes"]
    legend_labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert legend_labels == {CRITERION_LABELS["predictive_auc"], CRITERION_LABELS["stability"]}


def test_m_selection_recovery_labels_override_the_default():
    evaluation = _tiny_m_selection_evaluation()
    fig = plot_m_selection_recovery(
        evaluation, labels={"stability": "Custom Criterion Name"},
    )
    ax = fig.axes[0]
    legend_labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert "Custom Criterion Name" in legend_labels


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


def test_theta_recovery_plots_difficulty_by_default():
    theta_true = np.array([0.2, 0.5, 0.8])
    theta_est = np.array([0.25, 0.45, 0.75])
    ax = plot_theta_recovery(theta_true=theta_true, theta_est=theta_est)
    offsets = np.asarray(ax.collections[0].get_offsets())
    np.testing.assert_allclose(sorted(offsets[:, 0]), sorted(1.0 - theta_true))
    assert "difficulty" in ax.get_xlabel()


def test_theta_recovery_can_opt_out_to_easiness():
    theta_true = np.array([0.2, 0.5, 0.8])
    theta_est = np.array([0.25, 0.45, 0.75])
    ax = plot_theta_recovery(theta_true=theta_true, theta_est=theta_est, difficulty=False)
    offsets = np.asarray(ax.collections[0].get_offsets())
    np.testing.assert_allclose(sorted(offsets[:, 0]), sorted(theta_true))
    assert "easiness" in ax.get_xlabel()


# --------------------------------------------------------------------------- #
# plot_parameter_recovery (mu + difficulty + pi, combined)
# --------------------------------------------------------------------------- #
def test_parameter_recovery_from_result_has_three_panels():
    cond = SimulationCondition(
        n_archetypes=2, n_learners=150, responses_per_learner=8, n_skills=4, n_iter=5
    )
    result = run_condition(cond, seed=0, return_fit=True)
    fig = plot_parameter_recovery(result)
    assert len(fig.axes) == 3
    n_items = result.data.theta.shape[0]
    # panel 1 (mu): one point per (matched component, skill) cell
    assert len(fig.axes[0].collections) > 0
    # panel 2 (difficulty): one point per item, plotted as 1 - theta
    theta_offsets = np.asarray(fig.axes[1].collections[0].get_offsets())
    assert len(theta_offsets) == n_items
    np.testing.assert_allclose(
        sorted(theta_offsets[:, 0]), sorted(1.0 - result.data.theta.ravel())
    )
    # panel 3 (pi): one point per matched component
    pi_offsets = np.asarray(fig.axes[2].collections[0].get_offsets())
    assert len(pi_offsets) == min(result.data.pi.size, result.model.pi.size)


def test_parameter_recovery_from_arrays_pi_panel_uses_mu_alignment():
    # pi's alignment must match mu's Hungarian permutation, not just index order.
    mu_true = np.array([[0.2, 0.8], [0.8, 0.2]])
    mu_est = np.array([[0.75, 0.25], [0.25, 0.75]])  # a row-swapped, noisy copy
    pi_true = np.array([0.3, 0.7])
    pi_est = np.array([0.65, 0.35])  # aligned to mu_est's (swapped) row order

    fig = plot_parameter_recovery(mu_true=mu_true, mu_est=mu_est, pi_true=pi_true, pi_est=pi_est)
    pi_offsets = np.asarray(fig.axes[2].collections[0].get_offsets())
    # mu's Hungarian match pairs true archetype 0 with (row-swapped) est 1 and
    # true archetype 1 with est 0 -- pi must follow that SAME pairing, not
    # naive index order: (true=0.3, recovered=0.35) and (true=0.7, recovered=0.65).
    np.testing.assert_allclose(pi_offsets, [[0.3, 0.35], [0.7, 0.65]])


def test_parameter_recovery_requires_result_or_mu_arrays():
    with pytest.raises(ValueError):
        plot_parameter_recovery()
