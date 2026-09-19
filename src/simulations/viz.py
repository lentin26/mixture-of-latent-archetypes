"""Visualize assumed vs. recovered MoLA components from a simulation run.

A *component* (archetype) is a row of ``mu`` giving, for each skill, the
probability that a learner of that archetype has mastered the skill. These
functions plot those rows as proficiency profiles -- proficiency on the y-axis,
skill index on the x-axis -- and overlay the data-generating ("assumed") profile
with the fitted ("recovered") one, matched by the same Hungarian alignment the
recovery metrics use.

Typical use::

    from src.simulations import run_condition
    from src.simulations.factors import SimulationCondition
    from src.simulations.viz import plot_component_recovery

    result = run_condition(SimulationCondition(n_archetypes=4), seed=0, return_fit=True)
    fig = plot_component_recovery(result)
    fig.savefig("figures/component_recovery.pdf")
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.simulations.dgp import SimulatedDataset
from src.simulations.factors import (
    BASELINE,
    FACTOR_LEVELS,
    SEPARATION_LEVELS,
    SimulationCondition,
)
from src.simulations.metrics import align_components
from src.simulations.run import SimulationResult
from src.style import humanize_label

ASSUMED_KW = dict(color="#1f77b4", marker="o", markersize=4, linewidth=1.8, label="assumed")
RECOVERED_KW = dict(
    color="#d62728", marker="x", markersize=5, linewidth=1.8, linestyle="--",
    label="recovered",
)

# Human-readable labels for simulation metrics and SimulationCondition
# factors, so a figure never shows a raw snake_case column name. Used by
# plot_ofat_sensitivity / plot_m_sensitivity via their `labels=` argument,
# which merges on top of this dict -- pass e.g. labels={"mu_rmse": "..."} to
# override just one entry for a single figure. Anything not listed here
# falls back to src.style.humanize_label's generic Title Case conversion.
METRIC_LABELS: dict[str, str] = {
    # recovery metrics (vs. the data-generating parameters)
    "mu_rmse": "Archetype Recovery RMSE",
    "pi_mae": "Mixture Weight MAE",
    "pi_rmse": "Mixture Weight RMSE",
    "theta_rmse": "Item Difficulty RMSE",
    "assignment_ari": "Cluster Assignment ARI",
    # predictive metrics (held-out responses)
    "holdout_auc": "Holdout AUC",
    "holdout_brier": "Holdout Brier Score",
    "holdout_nll": "Holdout Negative Log-Likelihood",
    "final_nll": "Final Training Negative Log-Likelihood",
    # fit diagnostics
    "n_em_iters": "EM Iterations",
    "train_time_sec": "Training Time (s)",
    "init_time_sec": "Initialization Time (s)",
    "fit_time_sec": "EM Fit Time (s)",
    "time_per_iter_sec": "Time per EM Iteration (s)",
    "n_observations": "Number of Observations",
    "effective_n_archetypes": "Effective Number of Archetypes",
    # SimulationCondition factors
    "n_archetypes": "Number of Archetypes",
    "n_components_fit": "Fitted Number of Archetypes",
    "n_learners": "Number of Learners",
    "responses_per_learner": "Responses per Learner",
    "n_skills": "Number of Skills",
    "n_items": "Number of Items",
    "archetype_separation": "Archetype Separation",
    "archetype_imbalance": "Archetype Imbalance",
    "initialization": "Initialization",
    "sampling": "Sampling",
    "item_coverage": "Item Coverage",
    "model_specification": "Model Specification",
    "response_function": "Response Function",
    "holdout_frac": "Holdout Fraction",
    "m_hat": "Recovered M",
}

# Friendly legend names for pick_m_by_criterion's seven M-selection rules,
# used by plot_m_selection_recovery the same way METRIC_LABELS is used
# elsewhere -- merges under a `labels=` override the same way.
CRITERION_LABELS: dict[str, str] = {
    "predictive_auc": "Predictive (holdout AUC)",
    "predictive_nll": "Predictive (holdout NLL)",
    "stability": "Stability (EM-init spread)",
    "separation_saturation": "Separation (saturation)",
    "separation_threshold": "Separation (threshold)",
    "mass_saturation": "Population Mass (saturation)",
    "mass_threshold": "Population Mass (threshold)",
}


def _as_matrix(mu) -> np.ndarray:
    return np.asarray(mu, dtype=float)


def _savefig(fig, save: str | Path) -> None:
    """Save ``fig`` to ``save``, creating the parent directory if needed.

    Matplotlib's own ``savefig`` raises ``FileNotFoundError`` if the target
    directory doesn't exist yet -- e.g. a notebook kernel whose working
    directory ended up somewhere other than the repo root (a restarted
    kernel that skipped the ``%cd ..`` cell, say) will resolve a relative
    path like "figures/plot.pdf" against the wrong place. Creating the
    directory here turns that into "it just works" instead of a crash; it
    does not fix a wrong working directory, so if the file lands somewhere
    unexpected, check `Path.cwd()`.
    """
    path = Path(save)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def align_recovered_components(
    mu_true, mu_est
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return ``(mu_true, mu_est, perm, n_matched)``.

    ``perm[i]`` is the estimated component index matched to true component ``i``
    (Hungarian match on the skills both models share). ``n_matched`` is
    ``min(n_true, n_est)``.
    """
    mu_true = _as_matrix(mu_true)
    mu_est = _as_matrix(mu_est)
    k = min(mu_true.shape[1], mu_est.shape[1])
    perm = align_components(mu_true[:, :k], mu_est[:, :k])
    n_matched = min(mu_true.shape[0], perm.size)
    return mu_true, mu_est, perm, n_matched


def _components_from_result(
    result: SimulationResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    mu_true, mu_est, perm, n_matched = align_recovered_components(
        result.data.mu, result.model.mu
    )
    pi_true = np.asarray(result.data.pi, dtype=float).ravel()
    pi_est = np.asarray(result.model.pi, dtype=float).ravel()
    return mu_true, mu_est, perm, n_matched, pi_true, pi_est


def _skill_axis(n_skills: int, skill_labels: Sequence | None):
    x = np.arange(n_skills)
    if skill_labels is None:
        return x, None
    labels = list(skill_labels)[:n_skills]
    return x, labels


def _style_profile_axes(ax, x, skill_labels, *, ylabel=True, xlabel=True):
    ax.set_ylim(-0.03, 1.03)
    if xlabel:
        ax.set_xlabel("skill index")
    if ylabel:
        ax.set_ylabel("proficiency  P(skill mastered)")
    ax.grid(alpha=0.3, linewidth=0.6)
    if skill_labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(skill_labels, rotation=45, ha="right", fontsize=8)


def plot_components(
    mu,
    *,
    ax: plt.Axes | None = None,
    skill_labels: Sequence | None = None,
    weights=None,
    cmap: str = "viridis",
    label_fmt: str = "component {m}",
    **line_kw,
):
    """Plot every row of ``mu`` (shape ``(M, K)``) as a proficiency profile.

    Use this to view one set of components on its own -- the assumed components,
    or the recovered ones. ``weights`` (length ``M``) is appended to each legend
    entry as ``(pi=...)`` when given.
    """
    mu = _as_matrix(mu)
    n_components, n_skills = mu.shape
    if ax is None:
        _, ax = plt.subplots(figsize=(max(5, 0.35 * n_skills), 4))
    x, labels = _skill_axis(n_skills, skill_labels)
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, n_components))
    for m in range(n_components):
        name = label_fmt.format(m=m)
        if weights is not None:
            name = f"{name} (pi={float(np.ravel(weights)[m]):.2f})"
        kw = dict(marker="o", markersize=4, linewidth=1.6, color=colors[m])
        kw.update(line_kw)
        ax.plot(x, mu[m], label=name, **kw)
    _style_profile_axes(ax, x, labels)
    ax.legend(fontsize=8, ncol=1)
    return ax


def plot_component_recovery(
    result: SimulationResult | None = None,
    *,
    mu_true=None,
    mu_est=None,
    pi_true=None,
    pi_est=None,
    skill_labels: Sequence | None = None,
    layout: str = "grid",
    max_cols: int = 4,
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
):
    """Overlay assumed and recovered proficiency profiles, component by component.

    Pass a :class:`SimulationResult` (from ``run_condition(..., return_fit=True)``)
    or the arrays directly via ``mu_true`` / ``mu_est`` (each ``(M, K)``);
    ``pi_true`` / ``pi_est`` are optional mixture weights shown in the titles.

    ``layout="grid"`` draws one panel per matched component; ``layout="overlay"``
    draws all components on a single axes (assumed solid, recovered dashed, one
    color per component). Returns the :class:`matplotlib.figure.Figure`.
    """
    if result is not None:
        mu_true, mu_est, perm, n_matched, pi_true, pi_est = _components_from_result(result)
    else:
        if mu_true is None or mu_est is None:
            raise ValueError("pass either `result` or both `mu_true` and `mu_est`")
        mu_true, mu_est, perm, n_matched = align_recovered_components(mu_true, mu_est)
        pi_true = None if pi_true is None else np.asarray(pi_true, dtype=float).ravel()
        pi_est = None if pi_est is None else np.asarray(pi_est, dtype=float).ravel()

    x_true, labels = _skill_axis(mu_true.shape[1], skill_labels)
    x_est, _ = _skill_axis(mu_est.shape[1], skill_labels)

    if layout == "overlay":
        fig, ax = plt.subplots(figsize=figsize or (max(6, 0.4 * mu_true.shape[1]), 4.5))
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
        for i in range(n_matched):
            c = colors[i % 10]
            ax.plot(x_true, mu_true[i], color=c, marker="o", markersize=3,
                    linewidth=1.6, label=f"component {i}")
            ax.plot(x_est, mu_est[perm[i]], color=c, marker="x", markersize=4,
                    linewidth=1.6, linestyle="--")
        _style_profile_axes(ax, x_true, labels)
        ax.set_title("assumed (solid) vs. recovered (dashed)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        if save is not None:
            _savefig(fig, save)
        return fig

    if layout != "grid":
        raise ValueError(f"unknown layout {layout!r}; use 'grid' or 'overlay'")

    ncols = min(max_cols, n_matched)
    nrows = math.ceil(n_matched / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize or (3.4 * ncols, 2.8 * nrows),
        squeeze=False, sharex=True, sharey=True,
    )
    flat = axes.ravel()
    k_shared = min(mu_true.shape[1], mu_est.shape[1])
    for i in range(n_matched):
        ax = flat[i]
        ax.plot(x_true, mu_true[i], **ASSUMED_KW)
        ax.plot(x_est, mu_est[perm[i]], **RECOVERED_KW)
        title = f"component {i}"
        if pi_true is not None and pi_est is not None:
            title += f"   pi {pi_true[i]:.2f} -> {pi_est[perm[i]]:.2f}"
        rmse = float(np.sqrt(np.mean(
            (mu_true[i, :k_shared] - mu_est[perm[i], :k_shared]) ** 2
        )))
        title += f"\nRMSE {rmse:.3f}"
        ax.set_title(title, fontsize=9)
        _style_profile_axes(
            ax, x_true, labels,
            ylabel=(i % ncols == 0),
        )
    for j in range(n_matched, len(flat)):
        flat[j].set_visible(False)
    flat[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Assumed vs. recovered components", fontsize=12)
    fig.tight_layout()
    if save is not None:
        _savefig(fig, save)
    return fig


def plot_recovery_scatter(
    result: SimulationResult | None = None,
    *,
    mu_true=None,
    mu_est=None,
    ax: plt.Axes | None = None,
    save: str | Path | None = None,
):
    """Scatter recovered vs. assumed skill proficiencies with the ``y = x`` line.

    Every point is one (component, skill) cell after alignment; tight clustering
    on the diagonal means good recovery. Returns the ``Axes``.
    """
    if result is not None:
        mu_true, mu_est, perm, n_matched, *_ = _components_from_result(result)
    else:
        if mu_true is None or mu_est is None:
            raise ValueError("pass either `result` or both `mu_true` and `mu_est`")
        mu_true, mu_est, perm, n_matched = align_recovered_components(mu_true, mu_est)

    k = min(mu_true.shape[1], mu_est.shape[1])
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
    for i in range(n_matched):
        ax.scatter(
            mu_true[i, :k], mu_est[perm[i], :k],
            s=28, alpha=0.75, color=colors[i % 10], label=f"component {i}",
        )
    ax.plot([0, 1], [0, 1], color="0.4", linewidth=1, linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("assumed proficiency")
    ax.set_ylabel("recovered proficiency")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.legend(fontsize=8)
    if save is not None:
        _savefig(ax.figure, save)
    return ax


def plot_theta_recovery(
    result: SimulationResult | None = None,
    *,
    theta_true=None,
    theta_est=None,
    ax: plt.Axes | None = None,
    difficulty: bool = True,
    save: str | Path | None = None,
):
    """Scatter recovered vs. assumed item difficulty, one point per item.

    Pass a :class:`SimulationResult` (from ``run_condition(..., return_fit=True)``)
    or ``theta_true`` / ``theta_est`` arrays directly. Unlike archetype recovery,
    items keep a fixed, unambiguous index -- there's no label-switching across a
    mixture's components -- so no Hungarian alignment step is needed here; each
    item is simply compared to itself.

    ``difficulty=True`` (the default) plots item *difficulty* (``1 - theta``,
    matching :meth:`MoLA.get_item_difficulty` and the standard IRT convention
    of higher = harder) rather than the raw model parameter ``theta``, which
    is *easiness*. Pass ``difficulty=False`` to plot raw ``theta`` instead.
    Returns the ``Axes``.
    """
    if result is not None:
        theta_true = result.data.theta
        theta_est = result.model.theta
    elif theta_true is None or theta_est is None:
        raise ValueError("pass either `result` or both `theta_true` and `theta_est`")
    theta_true = np.asarray(theta_true, dtype=float).ravel()
    theta_est = np.asarray(theta_est, dtype=float).ravel()
    if difficulty:
        theta_true = 1.0 - theta_true
        theta_est = 1.0 - theta_est
        xlabel, ylabel = "assumed item difficulty", "recovered item difficulty"
    else:
        xlabel, ylabel = "assumed item easiness (theta)", "recovered item easiness (theta)"

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(theta_true, theta_est, s=20, alpha=0.6, color="#1f77b4")
    ax.plot([0, 1], [0, 1], color="0.4", linewidth=1, linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3, linewidth=0.6)
    if save is not None:
        _savefig(ax.figure, save)
    return ax


def plot_parameter_recovery(
    result: SimulationResult | None = None,
    *,
    mu_true=None,
    mu_est=None,
    theta_true=None,
    theta_est=None,
    pi_true=None,
    pi_est=None,
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
):
    """All three fitted parameters' recovery, side by side in one figure:
    archetype skill mastery (``mu``), item difficulty (``1 - theta``), and
    mixture weights (``pi``) -- each a true-vs-recovered scatter with a
    ``y = x`` reference line.

    Pass a :class:`SimulationResult` (from ``run_condition(..., return_fit=True)``)
    or the three pairs of arrays directly (``theta_true``/``theta_est`` and
    ``pi_true``/``pi_est`` are optional -- their panels are left blank without
    them, since only ``mu`` is required to align components at all). The
    first two panels are exactly :func:`plot_recovery_scatter` and
    :func:`plot_theta_recovery` (with ``difficulty=True``, both still
    available standalone) drawn into this figure's axes; the third is new --
    ``pi`` uses the SAME Hungarian alignment the ``mu`` panel computes (via
    :func:`align_recovered_components`), so all three panels agree on which
    fitted component corresponds to which true archetype. Returns the
    :class:`matplotlib.figure.Figure`.
    """
    if result is not None:
        data_mu, model_mu = result.data.mu, result.model.mu
        theta_true, theta_est = result.data.theta, result.model.theta
        pi_true = np.asarray(result.data.pi, dtype=float).ravel()
        pi_est = np.asarray(result.model.pi, dtype=float).ravel()
    else:
        if mu_true is None or mu_est is None:
            raise ValueError(
                "pass either `result` or `mu_true`/`mu_est` (theta_true/theta_est "
                "and pi_true/pi_est are optional, but needed for those panels)"
            )
        data_mu, model_mu = mu_true, mu_est
        pi_true = None if pi_true is None else np.asarray(pi_true, dtype=float).ravel()
        pi_est = None if pi_est is None else np.asarray(pi_est, dtype=float).ravel()

    fig, axes = plt.subplots(1, 3, figsize=figsize or (12.0, 4.2))

    plot_recovery_scatter(mu_true=data_mu, mu_est=model_mu, ax=axes[0])
    axes[0].set_title("Archetype skill mastery", fontsize=9)

    if theta_true is not None and theta_est is not None:
        plot_theta_recovery(theta_true=theta_true, theta_est=theta_est, ax=axes[1], difficulty=True)
    axes[1].set_title("Item difficulty", fontsize=9)

    if pi_true is not None and pi_est is not None:
        _, _, perm, n_matched = align_recovered_components(data_mu, model_mu)
        m = min(pi_true.size, perm.size, n_matched)
        px, py = pi_true[:m], pi_est[perm[:m]]
        lims = [0.0, float(max(px.max(), py.max())) * 1.15]
        axes[2].scatter(px, py, s=40, color="#2ca02c")
        axes[2].plot(lims, lims, color="0.4", linewidth=1, linestyle=":")
        axes[2].set_xlim(lims)
        axes[2].set_ylim(lims)
        axes[2].set_xlabel("assumed mixture weight")
        axes[2].set_ylabel("recovered mixture weight")
        axes[2].set_aspect("equal")
        axes[2].grid(alpha=0.3, linewidth=0.6)
    axes[2].set_title("Mixture weights", fontsize=9)

    fig.tight_layout()
    if save is not None:
        _savefig(fig, save)
    return fig


def _mark_reference_level(ax, x_position: float) -> None:
    """Dotted vertical line marking a reference/baseline level on the x-axis."""
    ax.axvline(x_position, color="0.6", linestyle=":", linewidth=1)


def plot_component_recovery_distribution(
    results: Sequence[SimulationResult] | None = None,
    *,
    mu_true=None,
    mu_ests: Sequence | None = None,
    groups: dict[str, Sequence[SimulationResult]] | None = None,
    group_colors: dict[str, str] | None = None,
    skill_labels: Sequence | None = None,
    max_cols: int = 4,
    figsize: tuple[float, float] | None = None,
    repeat_label: str = "inits",
    save: str | Path | None = None,
):
    """Spread of recovered profiles across repeated fits sharing one true ``mu``.

    Pass a list of :class:`SimulationResult` (e.g. from ``run_init_repeats``,
    which fixes one dataset and varies only EM's initialization draw; or from
    ``run_sample_repeats``, which fixes the population and varies only the
    drawn sample -- either way every result must share the same true `mu`),
    or ``mu_true`` plus a list of ``mu_est`` arrays directly. Each repeat's
    recovered components are aligned to the true ones independently (the
    Hungarian match can pick a different permutation each time), then drawn
    one panel per true component: every repeat's aligned profile thin and
    translucent, their median bold-dashed, the true profile bold-solid
    (``ASSUMED_KW``). Tight bundles argue estimation stability -- under
    ``run_init_repeats``, that the answer doesn't depend on where EM started;
    under ``run_sample_repeats``, that the answer doesn't depend on which
    particular sample was drawn.

    Pass ``groups`` instead (e.g. ``{"Random": random_results, "K-means":
    kmeans_results}``) to overlay MULTIPLE repeat-sets in the same
    per-component panels -- every group's repeats are aligned to and drawn
    against the SAME true ``mu`` (taken from the first group's first
    result; every other group's own true ``mu`` must match it, since
    groups are assumed to be repeated fits of the same dataset under
    different schemes/conditions, not different datasets), each in its own
    color with its own median line and legend entry. ``group_colors`` maps
    group name -> color; groups without an entry cycle through ``tab10``.
    Mutually exclusive with ``results``/``mu_true``+``mu_ests`` (the
    original single-group convention, unchanged).

    Panel titles are just ``"component {i}"`` -- sample size and spread are
    left to be reported in text/a table alongside the figure (e.g. via
    :func:`src.simulations.metrics.repeat_alignment_spread`) rather than
    baked into the image; ``repeat_label`` (default ``"inits"``, pass
    ``"samples"`` for ``run_sample_repeats`` results) is kept only for
    callers that want it for their own labeling. The skill-index x-axis
    label is set once for the whole figure (``fig.supxlabel``) rather than
    repeated per panel. Returns the :class:`matplotlib.figure.Figure`.
    """
    if groups is not None:
        group_items = list(groups.items())
        mu_true = _as_matrix(group_items[0][1][0].data.mu)
        group_mu_ests = {name: [_as_matrix(r.model.mu) for r in res] for name, res in group_items}
        for name, res in group_items[1:]:
            if not np.allclose(_as_matrix(res[0].data.mu), mu_true):
                raise ValueError(
                    f"group {name!r} does not share the same true mu as the first group -- "
                    f"groups must be repeated fits of the SAME dataset, not different ones"
                )
    elif results is not None:
        mu_true = _as_matrix(results[0].data.mu)
        group_mu_ests = {repeat_label: [_as_matrix(r.model.mu) for r in results]}
    else:
        if mu_true is None or mu_ests is None:
            raise ValueError("pass `groups`, `results`, or both `mu_true` and `mu_ests`")
        mu_true = _as_matrix(mu_true)
        group_mu_ests = {repeat_label: [_as_matrix(m) for m in mu_ests]}

    n_true = mu_true.shape[0]
    k = min(mu_true.shape[1], min(m.shape[1] for ests in group_mu_ests.values() for m in ests))
    # per_component[i][group_name] = list of that group's aligned mu_est rows
    per_component: list[dict[str, list[np.ndarray]]] = [
        {name: [] for name in group_mu_ests} for _ in range(n_true)
    ]
    for name, ests in group_mu_ests.items():
        for mu_est in ests:
            perm = align_components(mu_true[:, :k], mu_est[:, :k])
            n_matched = min(n_true, perm.size)
            for i in range(n_matched):
                per_component[i][name].append(mu_est[perm[i], :k])

    matched = [i for i, groupmap in enumerate(per_component) if any(groupmap.values())]
    x, labels = _skill_axis(k, skill_labels)
    ncols = min(max_cols, len(matched))
    nrows = math.ceil(len(matched) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize or (3.4 * ncols, 2.8 * nrows),
        squeeze=False, sharex=True, sharey=True,
    )
    flat = axes.ravel()

    default_colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
    resolved_colors = {
        name: (group_colors or {}).get(name, default_colors[idx % 10])
        for idx, name in enumerate(group_mu_ests)
    }

    for panel, i in enumerate(matched):
        ax = flat[panel]
        for name, curves_list in per_component[i].items():
            if not curves_list:
                continue
            curves = np.stack(curves_list)  # (n_repeats, k)
            color = resolved_colors[name]
            for curve in curves:
                ax.plot(x, curve, color=color, alpha=0.15, linewidth=1)
            ax.plot(x, np.median(curves, axis=0), color=color,
                     linewidth=2, linestyle="--", label=name)
        ax.plot(x, mu_true[i, :k], **ASSUMED_KW)
        ax.set_title(f"component {i}", fontsize=9)
        _style_profile_axes(ax, x, labels, ylabel=(panel % ncols == 0), xlabel=False)
    for j in range(len(matched), len(flat)):
        flat[j].set_visible(False)
    flat[0].legend(fontsize=8, loc="lower right")
    fig.supxlabel("skill index", fontsize=9)
    fig.tight_layout()
    if save is not None:
        _savefig(fig, save)
    return fig


def plot_theta_pi_recovery_distribution(
    groups: dict[str, Sequence[SimulationResult]],
    *,
    group_colors: dict[str, str] | None = None,
    difficulty: bool = True,
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
):
    """Item difficulty and mixture-weight recovery across repeated fits,
    overlaid by group (e.g. initialization scheme) -- the ``theta``/``pi``
    companion to :func:`plot_component_recovery_distribution`'s ``groups``
    mode, using scatter rather than profile lines (neither ``theta`` nor
    ``pi`` has a skill-index axis to plot a profile against).

    Every repeat's (true, recovered) pairs are pooled per group and plotted
    at low opacity in that group's color -- density differences between
    groups read directly from how tightly the points cluster near the
    ``y = x`` line, the same translucent-overlay convention the ``mu``
    panels use for their repeat lines. Mixture weights are aligned via each
    repeat's own ``mu``-based Hungarian permutation against the first
    group's first result's true ``mu`` (all groups must share it -- see
    :func:`plot_component_recovery_distribution`'s ``groups`` docstring),
    matching :func:`plot_parameter_recovery`'s ``pi`` panel.

    ``difficulty=True`` (the default) plots ``1 - theta`` rather than raw
    ``theta``, matching :func:`plot_theta_recovery`. ``group_colors`` maps
    group name -> color; groups without an entry cycle through ``tab10``.
    Returns the :class:`matplotlib.figure.Figure`.
    """
    group_items = list(groups.items())
    data_mu_true = _as_matrix(group_items[0][1][0].data.mu)

    default_colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
    resolved_colors = {
        name: (group_colors or {}).get(name, default_colors[idx % 10])
        for idx, name in enumerate(groups)
    }

    group_theta: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    group_pi: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, results in group_items:
        th_true = np.concatenate([np.asarray(r.data.theta, dtype=float).ravel() for r in results])
        th_est = np.concatenate([np.asarray(r.model.theta, dtype=float).ravel() for r in results])
        if difficulty:
            th_true, th_est = 1.0 - th_true, 1.0 - th_est
        group_theta[name] = (th_true, th_est)

        pi_true_parts, pi_est_parts = [], []
        for r in results:
            perm = align_components(data_mu_true, _as_matrix(r.model.mu))
            pi_true = np.asarray(r.data.pi, dtype=float).ravel()
            pi_est = np.asarray(r.model.pi, dtype=float).ravel()
            m = min(pi_true.size, perm.size)
            pi_true_parts.append(pi_true[:m])
            pi_est_parts.append(pi_est[perm[:m]])
        group_pi[name] = (np.concatenate(pi_true_parts), np.concatenate(pi_est_parts))

    fig, (theta_ax, pi_ax) = plt.subplots(1, 2, figsize=figsize or (8.0, 4.0))

    for name, (th_true, th_est) in group_theta.items():
        theta_ax.scatter(th_true, th_est, s=14, alpha=0.15, color=resolved_colors[name], label=name)
    theta_ax.plot([0, 1], [0, 1], color="0.4", linewidth=1, linestyle=":")
    theta_ax.set_xlim(0, 1)
    theta_ax.set_ylim(0, 1)
    xlab = "difficulty" if difficulty else "easiness (theta)"
    theta_ax.set_xlabel(f"assumed item {xlab}")
    theta_ax.set_ylabel(f"recovered item {xlab}")
    theta_ax.set_aspect("equal")
    theta_ax.grid(alpha=0.3, linewidth=0.6)
    theta_ax.legend(fontsize=8)

    pi_all = np.concatenate([v for pair in group_pi.values() for v in pair])
    pi_lims = [0.0, float(pi_all.max()) * 1.15]
    for name, (pi_true, pi_est) in group_pi.items():
        pi_ax.scatter(pi_true, pi_est, s=28, alpha=0.3, color=resolved_colors[name], label=name)
    pi_ax.plot(pi_lims, pi_lims, color="0.4", linewidth=1, linestyle=":")
    pi_ax.set_xlim(pi_lims)
    pi_ax.set_ylim(pi_lims)
    pi_ax.set_xlabel("assumed mixture weight")
    pi_ax.set_ylabel("recovered mixture weight")
    pi_ax.set_aspect("equal")
    pi_ax.grid(alpha=0.3, linewidth=0.6)
    pi_ax.legend(fontsize=8)

    fig.tight_layout()
    if save is not None:
        _savefig(fig, save)
    return fig


def _resolve_per_row(value, n_rows: int, name: str) -> list:
    """Broadcast a scalar to ``n_rows`` copies, or validate a given
    sequence already has length ``n_rows`` (one per row/metric)."""
    if isinstance(value, (list, tuple)):
        if len(value) != n_rows:
            raise ValueError(
                f"{name} sequence must have length {n_rows} (one per metric), got {len(value)}"
            )
        return list(value)
    return [value] * n_rows


def _plot_ofat_panel(
    ax, results, factor, metric, baseline_dict, log_x, log_y, reference_line, label_map,
    *, show_title: bool, show_ylabel: bool, show_legend: bool,
) -> None:
    other_factors = [f for f in FACTOR_LEVELS if f != factor and f in results.columns]
    mask = np.ones(len(results), dtype=bool)
    for f in other_factors:
        mask &= results[f] == baseline_dict[f]
    subset = results.loc[mask]

    levels = [lv for lv in FACTOR_LEVELS[factor] if lv in set(subset[factor])]
    stats = subset.groupby(factor)[metric].agg(mean="mean", std="std").reindex(levels)

    numeric = all(isinstance(lv, (int, float)) and not isinstance(lv, bool) for lv in levels)
    if numeric:
        x = np.asarray(levels, dtype=float)
    else:
        x = np.arange(len(levels))

    ax.errorbar(x, stats["mean"], yerr=stats["std"], marker="o",
                capsize=3, color="#1f77b4")
    if reference_line == "linear" and numeric:
        x0, y0 = x[0], stats["mean"].iloc[0]
        if x0 > 0 and np.isfinite(y0) and y0 > 0:
            # y = y0 * (x / x0): the p=1 power law anchored at the
            # smallest level's own mean -- real-space formula, so it
            # reads as a straight line under log_x=log_y=True (the
            # slope-1 signature of linear cost) and equally correctly
            # under linear-linear axes.
            ax.plot(x, y0 * (x / x0), color="0.3", linewidth=1.2,
                     linestyle="--", label="linear reference")
    if log_x and numeric:
        ax.set_xscale("log")
        # explicit level ticks below are the only ones that should show --
        # matplotlib's automatic log-scale minor ticks would otherwise
        # collide with them when levels are closely spaced (e.g. 2,4,8,16).
        ax.minorticks_off()
    if log_y:
        ax.set_yscale("log")
    if baseline_dict[factor] in levels:
        ref_x = float(baseline_dict[factor]) if numeric else levels.index(baseline_dict[factor])
        _mark_reference_level(ax, ref_x)

    if numeric:
        ax.set_xticks(x)
    else:
        ax.set_xticks(np.arange(len(levels)))
    ax.set_xticklabels([str(lv) for lv in levels], rotation=30, ha="right", fontsize=8)
    if show_title:
        ax.set_title(humanize_label(factor, label_map), fontsize=9)
    ax.grid(alpha=0.3, linewidth=0.6)
    if show_ylabel:
        ax.set_ylabel(humanize_label(metric, label_map))
    # guard: a non-numeric factor's panel never gets a reference line, so
    # only add a legend where there's actually a labeled artist to show.
    if show_legend and ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc="upper left")


def plot_ofat_sensitivity(
    results: pd.DataFrame,
    factors: Sequence[str] | None = None,
    metric: str | Sequence[str] = "mu_rmse",
    baseline: SimulationCondition = BASELINE,
    log_x: bool = False,
    log_y: bool | Sequence[bool] = False,
    sharey: bool = False,
    reference_line: str | None | Sequence[str | None] = None,
    max_cols: int = 4,
    figsize: tuple[float, float] | None = None,
    labels: dict[str, str] | None = None,
    save: str | Path | None = None,
):
    """Small-multiples plot of ``metric`` vs. level, one panel per OFAT factor.

    ``results`` is the **per-run** table from ``run_design(ofat_design(...))``
    (one row per condition x seed, carrying every :class:`SimulationCondition`
    field since each row is built from ``condition.to_dict()``) -- pass that
    directly, not the output of ``summarize_results``, which already
    aggregates away the per-condition factor columns this needs; the
    mean/std per level are computed here instead. For each factor, holds
    every OTHER swept factor at ``baseline``'s value (an OFAT design
    guarantees exactly one group of rows per level survives that filter) and
    plots the ``metric`` mean +/- std across that factor's levels, ordered as
    in ``FACTOR_LEVELS`` (not alphabetically -- several factors are ordered
    categoricals like "low"/"medium"/"high"). Marks the baseline level with a
    dotted vertical line. Numeric factors plot on their real values (so
    ``log_x=True`` log-scales panels with wide ranges, e.g. n_learners
    spanning 500-50,000); non-numeric factors plot on categorical positions.

    ``metric`` can be a single name (the original behavior: one row of
    panels, wrapped across ``max_cols`` columns) or a sequence of names --
    one ROW per metric, one COLUMN per factor, e.g. comparing per-iteration
    fit time against EM iteration count side by side to see whether a
    scaling claim about one is confounded by the other varying too. In that
    mode ``max_cols`` is ignored (columns = factors, always).

    ``log_y`` and ``reference_line`` are per-row when ``metric`` is a
    sequence -- pass either a single value (broadcast to every row) or a
    sequence matching ``len(metric)``, e.g. ``log_y=[True, False]`` when one
    metric is naturally log-scaled and another (like a small bounded
    iteration count) isn't.

    ``log_y=True`` additionally log-scales a row's own y-axis. Use
    ``log_x=log_y=True`` together when checking a scaling law (e.g.
    ``train_time_sec`` vs. problem size) -- never ``log_x`` alone (linear
    y): mixing a log axis against a linear one is misleading either
    direction, e.g. genuinely linear cost plotted with only ``log_x`` bends
    upward and can look quadratic or exponential, since a straight line
    under log-x/linear-y actually corresponds to *logarithmic* growth, not
    linear. Comparing two log scales (``log_x=log_y=True``) is fine: a power
    law ``y ~ x**p`` is a straight line of slope ``p`` there, so linear cost
    (``p=1``) reads as a straight line -- ``reference_line="linear"`` draws
    that ``p=1`` line explicitly (anchored at the smallest level's own mean,
    scaled proportionally to ``x``) so a reader doesn't have to eyeball the
    slope themselves to see the data track it. Works under any ``log_x``/
    ``log_y`` combination, including linear-linear, since the reference
    itself is just ``y = y[0] * (x / x[0])`` evaluated in real (not log)
    coordinates -- only ``log_x`` xor ``log_y`` (one log, one linear) is
    where a straight reference line stops meaning "linear" visually, for the
    same reason mixing axis scales is discouraged above. A row whose metric
    isn't a power-law claim at all (e.g. a diagnostic like EM iteration
    count) should just pass ``None`` for that row instead.

    ``sharey=True`` puts every panel in a row on one common y-axis (only the
    leftmost panel gets tick labels, matplotlib's usual shared-axis
    behavior) -- useful for comparing *magnitudes* across factors directly,
    but risks visually flattening ("smooshing") a factor whose own range is
    much smaller than another panel's. Under ``log_y=True`` this is usually
    safe even with quite different per-panel ranges (equal *ratios* get
    equal visual spacing regardless of where a panel's data sits on the
    shared axis); under linear ``y``, a much wider co-panel range can still
    flatten a narrower one, so prefer independent axes (the default) there
    unless the ranges are already comparable. With multiple metrics, sharing
    is always scoped to *within* each row -- different metrics are
    different units/scales entirely, so they're never shared with each
    other regardless of this flag.

    ``labels`` merges on top of :data:`METRIC_LABELS` for the factor/metric
    names shown in panel titles and the y-axis label -- pass e.g.
    ``labels={"mu_rmse": "..."}`` to override just one entry for this figure;
    anything neither in ``labels`` nor :data:`METRIC_LABELS` falls back to
    :func:`src.style.humanize_label`'s generic Title Case conversion.
    Returns the :class:`matplotlib.figure.Figure`.
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    log_y_per_row = _resolve_per_row(log_y, len(metrics), "log_y")
    reference_line_per_row = _resolve_per_row(reference_line, len(metrics), "reference_line")
    for rl in reference_line_per_row:
        if rl not in (None, "linear"):
            raise ValueError(f"reference_line must be None or 'linear', got {rl!r}")

    factors = list(factors) if factors is not None else sorted(FACTOR_LEVELS)
    baseline_dict = baseline.to_dict()
    label_map = {**METRIC_LABELS, **(labels or {})}
    multi_metric = len(metrics) > 1

    if multi_metric:
        nrows, ncols = len(metrics), len(factors)
        plt_sharey = "row" if sharey else False
    else:
        ncols = min(max_cols, len(factors))
        nrows = math.ceil(len(factors) / ncols)
        plt_sharey = sharey

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize or (3.6 * ncols, 3.0 * nrows),
        squeeze=False, sharey=plt_sharey,
    )

    if multi_metric:
        for row, m in enumerate(metrics):
            for col, factor in enumerate(factors):
                _plot_ofat_panel(
                    axes[row][col], results, factor, m, baseline_dict,
                    log_x, log_y_per_row[row], reference_line_per_row[row], label_map,
                    show_title=(row == 0), show_ylabel=(col == 0), show_legend=(col == 0),
                )
        fig.suptitle(
            "OFAT Sensitivity: " + " / ".join(humanize_label(m, label_map) for m in metrics),
            fontsize=12,
        )
    else:
        flat = axes.ravel()
        for idx, factor in enumerate(factors):
            _plot_ofat_panel(
                flat[idx], results, factor, metrics[0], baseline_dict,
                log_x, log_y_per_row[0], reference_line_per_row[0], label_map,
                show_title=True, show_ylabel=(idx % ncols == 0), show_legend=(idx == 0),
            )
        for j in range(len(factors), len(flat)):
            flat[j].set_visible(False)
        fig.suptitle(f"OFAT Sensitivity: {humanize_label(metrics[0], label_map)}", fontsize=12)

    fig.tight_layout()
    if save is not None:
        _savefig(fig, save)
    return fig


def plot_m_sensitivity(
    results: Sequence[SimulationResult],
    true_n_archetypes: int,
    metrics: Sequence[str] = ("mu_rmse", "holdout_auc", "effective_n_archetypes"),
    figsize: tuple[float, float] | None = None,
    labels: dict[str, str] | None = None,
    save: str | Path | None = None,
):
    """One panel per metric in ``metrics``, x-axis = fitted ``n_components``,
    from a single-seed sweep (e.g. ``run_m_sweep``). ``"effective_n_archetypes"``
    is read live from each fitted model (:meth:`MoLA.effective_n_archetypes`);
    every other metric is read from ``result.metrics``. Marks
    ``true_n_archetypes`` with a dotted vertical line -- the archetype-count
    diagnostics (:meth:`MoLA.effective_n_archetypes`,
    :meth:`MoLA.redundancy_report`) should show recovery holding up, and the
    effective count saturating near the truth, even once the fitted model is
    given more components than it needs.

    ``labels`` merges on top of :data:`METRIC_LABELS` for the metric names
    shown in panel titles/y-labels, same as :func:`plot_ofat_sensitivity`.
    Returns the :class:`matplotlib.figure.Figure`.
    """
    m_values = [int(r.metrics["n_components_fit"]) for r in results]
    label_map = {**METRIC_LABELS, **(labels or {})}

    ncols = min(4, len(metrics))
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize or (3.6 * ncols, 3.0 * nrows), squeeze=False,
    )
    flat = axes.ravel()
    for idx, metric in enumerate(metrics):
        ax = flat[idx]
        if metric == "effective_n_archetypes":
            y = [r.model.effective_n_archetypes() for r in results]
        else:
            y = [r.metrics[metric] for r in results]
        ax.plot(m_values, y, marker="o", color="#1f77b4")
        _mark_reference_level(ax, true_n_archetypes)
        ax.set_xlabel(humanize_label("n_components_fit", label_map))
        ax.set_ylabel(humanize_label(metric, label_map))
        ax.set_title(humanize_label(metric, label_map), fontsize=9)
        ax.grid(alpha=0.3, linewidth=0.6)
    for j in range(len(metrics), len(flat)):
        flat[j].set_visible(False)
    fig.suptitle("Sensitivity to the number of archetypes", fontsize=12)
    fig.tight_layout()
    if save is not None:
        _savefig(fig, save)
    return fig


def plot_m_selection_recovery(
    evaluation: pd.DataFrame,
    criteria: Sequence[str] | None = None,
    separation_levels: Sequence[str] = SEPARATION_LEVELS,
    figsize: tuple[float, float] | None = None,
    labels: dict[str, str] | None = None,
    save: str | Path | None = None,
):
    """One panel per ``archetype_separation`` level, from the long-form table
    :func:`src.simulations.run.run_m_selection_evaluation` returns.

    Each panel plots true ``n_archetypes`` (x) against the mean recovered
    ``m_hat`` (y, +/- std across seeds), one line per criterion in
    ``criteria`` (default: every criterion present in ``evaluation``) --
    the seven independent rules from
    :func:`src.simulations.run.pick_m_by_criterion`, evaluated separately
    rather than combined into one answer. A dotted ``y = x`` line marks
    perfect recovery; a criterion's series sitting on that line across true
    ``M`` values recovers it reliably at that separation level, one above
    or below it is systematically over/under-counting.

    ``labels`` merges on top of both :data:`METRIC_LABELS` (for the axis
    labels) and :data:`CRITERION_LABELS` (for the legend), same pattern as
    :func:`plot_ofat_sensitivity`. Returns the :class:`matplotlib.figure.Figure`.
    """
    criteria = list(criteria) if criteria is not None else sorted(evaluation["criterion"].unique())
    label_map = {**METRIC_LABELS, **(labels or {})}
    criterion_label_map = {**CRITERION_LABELS, **(labels or {})}

    levels = [lv for lv in separation_levels if lv in set(evaluation["archetype_separation"])]
    ncols = len(levels)
    fig, axes = plt.subplots(1, ncols, figsize=figsize or (4.2 * ncols, 4.2), squeeze=False)
    flat = axes.ravel()

    n_archetypes_values = sorted(evaluation["n_archetypes"].unique())
    lims = [min(n_archetypes_values), max(n_archetypes_values)]
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(criteria), 1)))

    for panel, sep in enumerate(levels):
        ax = flat[panel]
        subset = evaluation[evaluation["archetype_separation"] == sep]
        for c, criterion in enumerate(criteria):
            crit_rows = subset[subset["criterion"] == criterion]
            stats = crit_rows.groupby("n_archetypes")["m_hat"].agg(["mean", "std"])
            stats = stats.reindex(n_archetypes_values)
            ax.errorbar(
                stats.index, stats["mean"], yerr=stats["std"],
                marker="o", capsize=3, color=colors[c],
                label=humanize_label(criterion, criterion_label_map),
            )
        ax.plot(lims, lims, color="0.4", linewidth=1, linestyle=":")
        ax.set_xlabel(humanize_label("n_archetypes", label_map))
        if panel == 0:
            ax.set_ylabel(humanize_label("m_hat", label_map))
        ax.set_title(f"{humanize_label('archetype_separation', label_map)}: {str(sep).title()}", fontsize=9)
        ax.grid(alpha=0.3, linewidth=0.6)
    flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Recovered vs. true number of archetypes, by criterion", fontsize=12)
    fig.tight_layout()
    if save is not None:
        _savefig(fig, save)
    return fig


def _pairwise_jaccard(Q: np.ndarray) -> np.ndarray:
    """Pairwise Jaccard similarity between every pair of Q-matrix columns
    (skills), as a flat array over the upper triangle (excludes the diagonal,
    which is trivially 1)."""
    Qb = (np.asarray(Q) > 0).astype(int)
    n_skills = Qb.shape[1]
    inter = Qb.T @ Qb
    sizes = np.diag(inter)
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, inter / union, 0.0)
    i, j = np.triu_indices(n_skills, k=1)
    return jaccard[i, j]


def plot_setup_diagnostics(
    data: SimulatedDataset,
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
):
    """Six-panel check that the simulated data matches the intended design.

    This validates the *simulator*, not MoLA -- pass one :class:`SimulatedDataset`
    (e.g. from ``generate_dataset``) and get: (1) users per archetype vs. the
    expected count from ``pi``, (2) responses per user (a spike at
    ``responses_per_learner`` under ``sampling="iid"``; a real distribution
    under ``"sparse"``), (3) items per skill, (4) skills per item, (5) the
    distribution of pairwise Q-matrix column (skill) similarity -- a right
    tail near 1 would flag an identifiability risk, (6) archetype separation,
    reusing :func:`plot_components` for the direct visual. Returns the
    :class:`matplotlib.figure.Figure`.
    """
    n_archetypes = data.mu.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=figsize or (12, 7))
    flat = axes.ravel()

    # 1. users per archetype
    ax = flat[0]
    counts = np.bincount(data.z, minlength=n_archetypes)
    expected = data.pi.ravel() * len(data.z)
    ax.bar(np.arange(n_archetypes), counts, color="#1f77b4", alpha=0.8, label="realized")
    ax.scatter(np.arange(n_archetypes), expected, color="#d62728", marker="_", s=400,
               linewidths=2, label="expected (pi * n_learners)", zorder=3)
    ax.set_xlabel("archetype")
    ax.set_ylabel("number of learners")
    ax.set_title("Users per archetype", fontsize=9)
    ax.legend(fontsize=7)

    # 2. responses per user
    ax = flat[1]
    responses_per_user = (~np.isnan(data.X)).sum(axis=1)
    ax.hist(responses_per_user, bins=min(20, len(np.unique(responses_per_user))), color="#1f77b4")
    ax.set_xlabel("responses per learner")
    ax.set_ylabel("number of learners")
    ax.set_title("Responses per user", fontsize=9)

    # 3. items per skill
    ax = flat[2]
    items_per_skill = np.asarray(data.Q.sum(axis=0)).ravel()
    ax.bar(np.arange(len(items_per_skill)), items_per_skill, color="#1f77b4")
    ax.set_xlabel("skill index")
    ax.set_ylabel("number of items")
    ax.set_title("Items per skill", fontsize=9)

    # 4. skills per item
    ax = flat[3]
    skills_per_item = np.asarray(data.Q.sum(axis=1)).ravel()
    bins = np.arange(skills_per_item.min(), skills_per_item.max() + 2) - 0.5
    ax.hist(skills_per_item, bins=bins, color="#1f77b4", rwidth=0.8)
    ax.set_xlabel("skills required")
    ax.set_ylabel("number of items")
    ax.set_title("Skills per item", fontsize=9)

    # 5. Q-matrix column (skill) similarity
    ax = flat[4]
    similarities = _pairwise_jaccard(data.Q)
    ax.hist(similarities, bins=20, range=(0, 1), color="#1f77b4")
    ax.set_xlabel("pairwise Jaccard similarity")
    ax.set_ylabel("number of skill pairs")
    ax.set_title(f"Q-matrix column similarity\nmax={similarities.max():.2f}", fontsize=9)

    # 6. archetype separation
    ax = flat[5]
    plot_components(data.mu, ax=ax, weights=data.pi)
    diffs = data.mu[:, None, :] - data.mu[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    i, j = np.triu_indices(n_archetypes, k=1)
    pairwise = dists[i, j]
    ax.set_title(f"Archetype separation\nmean pairwise distance={pairwise.mean():.2f}", fontsize=9)

    fig.tight_layout()
    if save is not None:
        _savefig(fig, save)
    return fig
