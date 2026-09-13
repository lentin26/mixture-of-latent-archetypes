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
from src.simulations.factors import BASELINE, FACTOR_LEVELS, SimulationCondition
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


def _style_profile_axes(ax, x, skill_labels, *, ylabel=True):
    ax.set_ylim(-0.03, 1.03)
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
    save: str | Path | None = None,
):
    """Scatter recovered vs. assumed item difficulty (``theta``), one point per item.

    Pass a :class:`SimulationResult` (from ``run_condition(..., return_fit=True)``)
    or ``theta_true`` / ``theta_est`` arrays directly. Unlike archetype recovery,
    items keep a fixed, unambiguous index -- there's no label-switching across a
    mixture's components -- so no Hungarian alignment step is needed here; each
    item is simply compared to itself. Returns the ``Axes``.
    """
    if result is not None:
        theta_true = result.data.theta
        theta_est = result.model.theta
    elif theta_true is None or theta_est is None:
        raise ValueError("pass either `result` or both `theta_true` and `theta_est`")
    theta_true = np.asarray(theta_true, dtype=float).ravel()
    theta_est = np.asarray(theta_est, dtype=float).ravel()

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(theta_true, theta_est, s=20, alpha=0.6, color="#1f77b4")
    ax.plot([0, 1], [0, 1], color="0.4", linewidth=1, linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("assumed item easiness (theta)")
    ax.set_ylabel("recovered item easiness (theta)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3, linewidth=0.6)
    if save is not None:
        _savefig(ax.figure, save)
    return ax


def _mark_reference_level(ax, x_position: float) -> None:
    """Dotted vertical line marking a reference/baseline level on the x-axis."""
    ax.axvline(x_position, color="0.6", linestyle=":", linewidth=1)


def plot_component_recovery_distribution(
    results: Sequence[SimulationResult] | None = None,
    *,
    mu_true=None,
    mu_ests: Sequence | None = None,
    skill_labels: Sequence | None = None,
    max_cols: int = 4,
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
):
    """Spread of recovered profiles across repeated fits of the SAME data.

    Pass a list of :class:`SimulationResult` sharing one ``data`` (e.g. from
    ``run_init_repeats``), or ``mu_true`` plus a list of ``mu_est`` arrays
    directly. Each repeat's recovered components are aligned to the true ones
    independently (the Hungarian match can pick a different permutation each
    time), then drawn one panel per true component: every repeat's aligned
    profile thin and translucent, their median bold-dashed, the true profile
    bold-solid (``ASSUMED_KW``). Use this to argue estimation stability --
    tight bundles mean EM lands in essentially the same place regardless of
    where it started. Returns the :class:`matplotlib.figure.Figure`.
    """
    if results is not None:
        mu_true = _as_matrix(results[0].data.mu)
        mu_ests = [_as_matrix(r.model.mu) for r in results]
    else:
        if mu_true is None or mu_ests is None:
            raise ValueError("pass either `results` or both `mu_true` and `mu_ests`")
        mu_true = _as_matrix(mu_true)
        mu_ests = [_as_matrix(m) for m in mu_ests]

    n_true = mu_true.shape[0]
    k = min(mu_true.shape[1], min(m.shape[1] for m in mu_ests))
    per_component: list[list[np.ndarray]] = [[] for _ in range(n_true)]
    for mu_est in mu_ests:
        perm = align_components(mu_true[:, :k], mu_est[:, :k])
        n_matched = min(n_true, perm.size)
        for i in range(n_matched):
            per_component[i].append(mu_est[perm[i], :k])

    matched = [i for i, curves in enumerate(per_component) if curves]
    x, labels = _skill_axis(k, skill_labels)
    ncols = min(max_cols, len(matched))
    nrows = math.ceil(len(matched) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize or (3.4 * ncols, 2.8 * nrows),
        squeeze=False, sharex=True, sharey=True,
    )
    flat = axes.ravel()
    for panel, i in enumerate(matched):
        ax = flat[panel]
        curves = np.stack(per_component[i])  # (n_repeats, k)
        for curve in curves:
            ax.plot(x, curve, color=RECOVERED_KW["color"], alpha=0.15, linewidth=1)
        ax.plot(x, np.median(curves, axis=0), color=RECOVERED_KW["color"],
                 linewidth=2, linestyle="--", label="recovered (median)")
        ax.plot(x, mu_true[i, :k], **ASSUMED_KW)
        spread = float(curves.std(axis=0).mean())
        ax.set_title(f"component {i}\nn={len(curves)} inits, mean sd={spread:.3f}", fontsize=9)
        _style_profile_axes(ax, x, labels, ylabel=(panel % ncols == 0))
    for j in range(len(matched), len(flat)):
        flat[j].set_visible(False)
    flat[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    if save is not None:
        _savefig(fig, save)
    return fig


def plot_ofat_sensitivity(
    results: pd.DataFrame,
    factors: Sequence[str] | None = None,
    metric: str = "mu_rmse",
    baseline: SimulationCondition = BASELINE,
    log_x: bool = False,
    log_y: bool = False,
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

    ``log_y=True`` additionally log-scales the (shared) y-axis. Use
    ``log_x=log_y=True`` together when checking a scaling law (e.g.
    ``train_time_sec`` vs. problem size): on log-log axes a power law
    ``y ~ x**p`` is a straight line of slope ``p``, so linear cost (``p=1``)
    reads as a straight line -- plotted with ``log_x`` alone (linear y),
    genuinely linear cost bends upward and can look quadratic or exponential,
    since a straight line under log-x/linear-y actually corresponds to
    *logarithmic* growth, not linear.

    ``labels`` merges on top of :data:`METRIC_LABELS` for the factor/metric
    names shown in panel titles and the y-axis label -- pass e.g.
    ``labels={"mu_rmse": "..."}`` to override just one entry for this figure;
    anything neither in ``labels`` nor :data:`METRIC_LABELS` falls back to
    :func:`src.style.humanize_label`'s generic Title Case conversion.
    Returns the :class:`matplotlib.figure.Figure`.
    """
    factors = list(factors) if factors is not None else sorted(FACTOR_LEVELS)
    baseline_dict = baseline.to_dict()
    label_map = {**METRIC_LABELS, **(labels or {})}

    ncols = min(max_cols, len(factors))
    nrows = math.ceil(len(factors) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize or (3.6 * ncols, 3.0 * nrows),
        squeeze=False, sharey=False,
    )
    flat = axes.ravel()
    for idx, factor in enumerate(factors):
        ax = flat[idx]
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
        ax.set_title(humanize_label(factor, label_map), fontsize=9)
        ax.grid(alpha=0.3, linewidth=0.6)
        if idx % ncols == 0:
            ax.set_ylabel(humanize_label(metric, label_map))
    for j in range(len(factors), len(flat)):
        flat[j].set_visible(False)
    fig.suptitle(f"OFAT Sensitivity: {humanize_label(metric, label_map)}", fontsize=12)
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
