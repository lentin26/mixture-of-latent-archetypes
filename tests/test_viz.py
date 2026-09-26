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
    plot_recovery_distribution,
    plot_recovery_distribution_comparison,
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
    assert ax0.get_xlabel() == "Skill Index"
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
    assert ax.get_ylabel().startswith("Proficiency")


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
    assert ax.get_title() == "Component 0"


def test_recovery_distribution_xlabel_is_figure_level_not_per_panel():
    results = run_init_repeats(SMALL_VIZ, data_seed=0, n_repeats=4)
    fig = plot_component_recovery_distribution(results)
    visible = [a for a in fig.axes if a.get_visible()]
    assert fig.get_supxlabel() == "Skill Index"
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
    assert legend_labels == {"Random", "K-means", "Assumed"}
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
    assert "Difficulty" in theta_ax.get_xlabel()
    offsets = np.asarray(theta_ax.collections[0].get_offsets())
    n_items = random_results[0].data.theta.shape[0]
    expected_x = np.concatenate([1.0 - r.data.theta.ravel() for r in random_results])
    assert len(offsets) == n_items * len(random_results)
    np.testing.assert_allclose(sorted(offsets[:, 0]), sorted(expected_x))


# --------------------------------------------------------------------------- #
# plot_recovery_distribution (combined component + theta/pi panels)
# --------------------------------------------------------------------------- #
def _init_scheme_groups(n_repeats=3):
    from dataclasses import replace
    return {
        "Random": run_init_repeats(replace(SMALL_VIZ, initialization="random"), data_seed=0, n_repeats=n_repeats),
        "K-means": run_init_repeats(replace(SMALL_VIZ, initialization="k-means"), data_seed=0, n_repeats=n_repeats),
    }


def test_recovery_distribution_combined_has_component_panels_plus_theta_pi():
    groups = _init_scheme_groups()
    fig = plot_recovery_distribution(groups=groups, group_colors={"Random": "#d62728", "K-means": "#1f77b4"})
    visible = [a for a in fig.axes if a.get_visible()]
    # n_archetypes component panels + 1 theta panel + 1 pi panel
    assert len(visible) == SMALL_VIZ.n_archetypes + 2

    component_titles = {a.get_title() for a in visible if a.get_title().startswith("Component")}
    assert component_titles == {f"Component {i}" for i in range(SMALL_VIZ.n_archetypes)}

    theta_ax, pi_ax = visible[-2], visible[-1]
    assert "Difficulty" in theta_ax.get_xlabel()
    assert "Mixture Weight" in pi_ax.get_xlabel()


def test_recovery_distribution_combined_grid_is_always_three_columns():
    # SMALL_VIZ.n_archetypes == 2 -> 1 component row, but column 2 always
    # needs 2 rows (difficulty, mixture weight) -> 2 rows regardless
    groups = _init_scheme_groups()
    fig = plot_recovery_distribution(groups=groups)
    assert fig.axes[0].get_gridspec().nrows == 2
    assert fig.axes[0].get_gridspec().ncols == 3


def test_recovery_distribution_combined_grid_scales_rows_with_more_components():
    from dataclasses import replace
    four_archetype_condition = replace(SMALL_VIZ, n_archetypes=4)
    groups = {
        "Random": run_init_repeats(replace(four_archetype_condition, initialization="random"), data_seed=0, n_repeats=2),
        "K-means": run_init_repeats(replace(four_archetype_condition, initialization="k-means"), data_seed=0, n_repeats=2),
    }
    # 4 components -> 2 component rows (2x2 block), still 3 columns (2 for
    # components + 1 narrower one for the stacked difficulty/mixture panels)
    fig = plot_recovery_distribution(groups=groups)
    assert fig.axes[0].get_gridspec().nrows == 2
    assert fig.axes[0].get_gridspec().ncols == 3


def test_recovery_distribution_combined_ylabel_text_appears_once_not_per_row():
    # With a 2-row component block, every column-0 panel (one per row) used
    # to each get their own copy of the (long, rotated) y-axis label text,
    # which collided across rows when rendered -- fixed by giving the whole
    # figure one shared `fig.supylabel` instead of any individual panel's
    # own ylabel (component panels all pass ylabel=False now).
    from dataclasses import replace
    four_archetype_condition = replace(SMALL_VIZ, n_archetypes=4)
    groups = {
        "Random": run_init_repeats(replace(four_archetype_condition, initialization="random"), data_seed=0, n_repeats=2),
        "K-means": run_init_repeats(replace(four_archetype_condition, initialization="k-means"), data_seed=0, n_repeats=2),
    }
    fig = plot_recovery_distribution(groups=groups)
    component_axes = [a for a in fig.axes if a.get_title().startswith("Component")]
    non_empty_ylabels = [ax.get_ylabel() for ax in component_axes if ax.get_ylabel()]
    assert len(non_empty_ylabels) == 0
    assert fig._supylabel is not None and fig._supylabel.get_text()


def test_recovery_distribution_combined_shares_legend_and_group_names():
    groups = _init_scheme_groups()
    fig = plot_recovery_distribution(groups=groups)
    # one shared legend for the whole figure (drawn below it), not a
    # per-panel one -- no individual Axes carries its own legend.
    assert all(ax.get_legend() is None for ax in fig.axes)
    legend_labels = {t.get_text() for t in fig.legends[0].get_texts()}
    assert legend_labels == {"Random", "K-means", "Assumed"}


def test_recovery_distribution_combined_rejects_mismatched_true_mu():
    from dataclasses import replace
    group_a = run_init_repeats(SMALL_VIZ, data_seed=0, n_repeats=2)
    group_b = run_init_repeats(SMALL_VIZ, data_seed=1, n_repeats=2)  # different dataset -> different true mu
    with pytest.raises(ValueError, match="does not share the same true mu"):
        plot_recovery_distribution(groups={"A": group_a, "B": group_b})


def test_recovery_distribution_combined_saves_to_file(tmp_path):
    groups = _init_scheme_groups(n_repeats=2)
    save_path = tmp_path / "combined.pdf"
    plot_recovery_distribution(groups=groups, save=str(save_path))
    assert save_path.exists()


def test_recovery_distribution_combined_legend_is_one_horizontal_row_below_figure():
    groups = _init_scheme_groups()
    fig = plot_recovery_distribution(groups=groups)
    legend = fig.legends[0]
    # horizontal: as many columns as entries, not stacked in one column
    assert legend._ncols == len(legend.get_texts())
    # below the axes area, not overlapping it
    assert legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted()).y1 <= 0.05


def test_recovery_distribution_combined_shares_ticks_within_component_row():
    groups = _init_scheme_groups()  # SMALL_VIZ.n_archetypes == 2 -> both components fit in one row
    fig = plot_recovery_distribution(groups=groups)
    visible = [a for a in fig.axes if a.get_visible()]
    component_axes = visible[:2]  # n_archetypes == 2, then theta_ax, pi_ax
    # leftmost component panel keeps its tick labels
    assert any(t.get_visible() for t in component_axes[0].get_yticklabels())
    # second (non-leftmost) component panel shares the first's y-axis and hides its own labels
    assert component_axes[0].get_shared_y_axes().joined(component_axes[0], component_axes[1])
    assert all(not t.get_visible() for t in component_axes[1].get_yticklabels())


# --------------------------------------------------------------------------- #
# plot_recovery_distribution_comparison (two experiments, side by side)
# --------------------------------------------------------------------------- #
def _comparison_groups(n_repeats=2):
    from dataclasses import replace
    four_archetype_condition = replace(SMALL_VIZ, n_archetypes=4)
    left = {
        "Random": run_init_repeats(replace(four_archetype_condition, initialization="random"), data_seed=0, n_repeats=n_repeats),
        "K-means": run_init_repeats(replace(four_archetype_condition, initialization="k-means"), data_seed=0, n_repeats=n_repeats),
        "Informed": run_init_repeats(replace(four_archetype_condition, initialization="informed"), data_seed=0, n_repeats=n_repeats),
    }
    # a different data_seed -> a genuinely different true mu, on purpose:
    # the two sides are independent experiments, not required to match.
    right = {
        "K-means": run_init_repeats(replace(four_archetype_condition, initialization="k-means"), data_seed=1, n_repeats=n_repeats),
        "Informed": run_init_repeats(replace(four_archetype_condition, initialization="informed"), data_seed=1, n_repeats=n_repeats),
    }
    return left, right


def test_recovery_distribution_comparison_has_one_row_per_component_plus_two():
    left, right = _comparison_groups()
    fig = plot_recovery_distribution_comparison(
        left_groups=left, right_groups=right, left_title="Left", right_title="Right",
    )
    # 4 components + difficulty + mixture-weight = 6 rows, 2 columns each = 12 axes
    assert len(fig.axes) == 12


def test_recovery_distribution_comparison_sides_are_independent_datasets():
    left, right = _comparison_groups()
    fig = plot_recovery_distribution_comparison(
        left_groups=left, right_groups=right, left_title="Left", right_title="Right",
    )
    left_legend, right_legend = fig.legends
    assert {t.get_text() for t in left_legend.get_texts()} == {"Random", "K-means", "Informed", "Assumed"}
    assert {t.get_text() for t in right_legend.get_texts()} == {"K-means", "Informed", "Assumed"}


def test_recovery_distribution_comparison_row_labels_appear_once():
    left, right = _comparison_groups()
    fig = plot_recovery_distribution_comparison(
        left_groups=left, right_groups=right, left_title="Left", right_title="Right",
    )
    row_label_texts = {t.get_text() for t in fig.texts}
    for expected in ["Component 0", "Component 1", "Component 2", "Component 3", "Item Difficulty", "Mixture Weight"]:
        assert sum(t.get_text() == expected for t in fig.texts) == 1, expected
        assert expected in row_label_texts


def test_recovery_distribution_comparison_draws_a_divider_line():
    left, right = _comparison_groups()
    fig = plot_recovery_distribution_comparison(
        left_groups=left, right_groups=right, left_title="Left", right_title="Right",
    )
    from matplotlib.lines import Line2D
    divider_candidates = [a for a in fig.artists if isinstance(a, Line2D)]
    assert len(divider_candidates) >= 1


def test_recovery_distribution_comparison_rejects_mismatched_component_counts():
    left, _ = _comparison_groups()
    small_right = {"K-means": run_init_repeats(SMALL_VIZ, data_seed=1, n_repeats=2)}  # n_archetypes == 2, not 4
    with pytest.raises(ValueError, match="same count"):
        plot_recovery_distribution_comparison(
            left_groups=left, right_groups=small_right, left_title="Left", right_title="Right",
        )


def test_recovery_distribution_comparison_saves_to_file(tmp_path):
    left, right = _comparison_groups()
    save_path = tmp_path / "comparison.pdf"
    plot_recovery_distribution_comparison(
        left_groups=left, right_groups=right, left_title="Left", right_title="Right", save=str(save_path),
    )
    assert save_path.exists()


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
