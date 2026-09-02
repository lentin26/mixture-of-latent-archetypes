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

from src.simulations.factors import SimulationCondition
from src.simulations.run import run_condition
from src.simulations.viz import (
    align_recovered_components,
    plot_component_recovery,
    plot_components,
    plot_recovery_scatter,
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
