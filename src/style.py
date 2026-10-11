"""Publication-quality Matplotlib styling for figures headed into a LaTeX paper.

Independent of ``src.mola`` / ``src.simulations`` on purpose, so any plotting
code in the repo can use it. Handles the two things Matplotlib's defaults get
wrong for a journal figure:

1. Fonts and line/marker weights sized for the printed page, not a full
   monitor -- a figure that looks fine full-screen reads as thin and tiny
   once shrunk to a single LaTeX column.
2. Sizing the figure in inches from the LaTeX column/text width in points, so
   ``\\includegraphics`` doesn't rescale it (rescaling blurs text and distorts
   line weights relative to the surrounding body text).

Typical use, once per notebook session::

    from src.style import publication_style, figsize

    with publication_style():
        fig = plot_component_recovery(result, figsize=figsize("acm-sigconf-column"))
        fig.savefig("figures/component_recovery.pdf")

``text.usetex`` stays ``False`` throughout: it requires a working local LaTeX
toolchain (``latex``/``dvipng``) to render every string, and a missing one
turns every plot call into a cryptic failure. Matplotlib's built-in ``cm``
mathtext font set plus a Computer-Modern-like serif family gets visually very
close to real LaTeX text without that dependency.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import matplotlib as mpl

# LaTeX's point is 1/72.27 inch (vs. Matplotlib's own "pt" units, 1/72 inch --
# close enough that most papers ignore the ~0.4% difference, but figure width
# should match \includegraphics exactly, so use the real LaTeX conversion).
PT_TO_INCH = 1.0 / 72.27

# Common document-class column/text widths in points, for figsize() below.
# These are reasonable defaults, NOT a substitute for your actual document's
# value -- get the exact number by adding \typeout{WIDTH:\the\columnwidth}
# (or \the\textwidth) next to your figure and reading it from the .log file
# after compiling, then pass that literal float to figsize() instead of a key.
COLUMN_WIDTHS_PT = {
    "acm-sigconf-column": 241.02039,  # acmart, \documentclass[sigconf]{acmart}, \columnwidth
    "acm-sigconf-text": 505.89,       # acmart sigconf, \textwidth (figure* spanning both columns)
    "ieee-column": 252.0,             # IEEEtran, \columnwidth
    "ieee-text": 516.0,               # IEEEtran, \textwidth
    "single-column-article": 469.75499,  # plain \documentclass{article}, \textwidth
}

# rcParams for a figure meant to sit in a printed, multi-column paper.
RC_PUBLICATION = {
    # Text: no LaTeX toolchain dependency (see module docstring). A Times-like
    # serif reads as part of the document rather than a screenshot dropped in.
    #
    # Deliberately NOT "Computer Modern Roman" / "cmr10": those are the names
    # of Matplotlib's *internal mathtext-only* font assets. If neither
    # "Nimbus Roman" nor "Times New Roman" is installed, Matplotlib's font
    # matcher falls through to "cmr10" by name -- silently substituting a
    # font meant only for math glyphs inside `$...$` as the plain-text body
    # font. That font is missing ordinary glyphs like the literal underscore,
    # so labels such as "n_learners" render as "n˙learners". Confirmed by
    # rendering with each candidate in isolation -- don't reintroduce them.
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,

    # Sizes tuned to stay legible once shrunk to a column width, not
    # Matplotlib's screen-sized defaults.
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 10,

    # Line/marker/spine weights that hold up at print size (Matplotlib's
    # defaults are tuned for a monitor and look spindly once shrunk).
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    # Vector-safe, embeddable text. Matplotlib's default PDF font type (3,
    # bitmap-in-vector) is rejected outright by some journal production
    # systems; type 42 (TrueType) keeps text as real, searchable/editable text.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    # Minimal chrome: a caption in the LaTeX source describes the figure, so
    # keep grid/legend unobtrusive and don't fight fig.tight_layout()/
    # bbox_inches="tight" (already used by every save= call in viz.py) with
    # Matplotlib's own auto-layout.
    "figure.autolayout": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.transparent": False,
}


def figsize(width_pt: float | str = "acm-sigconf-column", fraction: float = 1.0, aspect: float = 0.68) -> tuple[float, float]:
    """Figure size in inches that exactly fills ``fraction`` of a LaTeX width.

    ``width_pt`` is either a raw point value from your compiled document (see
    the module docstring) or a key into ``COLUMN_WIDTHS_PT`` for a few common
    document classes. ``aspect`` is the height/width ratio -- 0.68 is close to
    the golden ratio and a reasonable default for a single row of panels;
    lower it for a wide multi-panel figure, raise it for a near-square one.
    """
    pt = COLUMN_WIDTHS_PT[width_pt] if isinstance(width_pt, str) else width_pt
    width_in = pt * PT_TO_INCH * fraction
    return (width_in, width_in * aspect)


@contextmanager
def publication_style(overrides: dict | None = None) -> Iterator[None]:
    """Apply :data:`RC_PUBLICATION` (merged with ``overrides``) for figures
    created inside the ``with`` block; restores the prior rcParams on exit.

    Prefer this over :func:`apply` when only some of a notebook's figures are
    publication-bound, so a stray screen-oriented plot elsewhere in the same
    session isn't silently affected.
    """
    rc = {**RC_PUBLICATION, **(overrides or {})}
    with mpl.rc_context(rc):
        yield


def apply(overrides: dict | None = None) -> None:
    """Apply :data:`RC_PUBLICATION` globally for the rest of the process.

    Convenient at the top of a notebook where every remaining figure should
    be publication-styled. Persists until :func:`matplotlib.rcdefaults` is
    called or ``apply``/:func:`publication_style` is used again.
    """
    mpl.rcParams.update({**RC_PUBLICATION, **(overrides or {})})


def strip_titles(fig, keep_panel_titles: bool = True) -> None:
    """Remove the figure-level suptitle after a plotting call.

    A journal caption (``\\caption{...}`` in the LaTeX source) already
    describes the figure, so viz.py's ``fig.suptitle(...)`` calls are
    redundant -- and often actively unwanted -- in a submission figure. Set
    ``keep_panel_titles=False`` to also clear per-panel titles (e.g.
    "component 0"); leave it ``True`` to keep those, since they're often the
    only way to tell panels apart once the descriptive suptitle is gone.
    """
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    if not keep_panel_titles:
        for ax in fig.axes:
            ax.set_title("")


# A few statistics/ML acronyms worth keeping upper-case in the generic
# fallback below, so an unmapped name like "holdout_auc" still reads as
# "Holdout AUC" rather than "Holdout Auc".
_ACRONYMS = {"auc", "mae", "rmse", "nll", "ari", "em", "id"}


def humanize_label(name: str, overrides: dict[str, str] | None = None) -> str:
    """Turn a ``snake_case`` column/variable name into a readable label.

    Checks ``overrides`` first for an exact match on ``name`` (e.g. a
    domain-specific dict mapping ``"mu_rmse"`` -> ``"Archetype Recovery
    RMSE"``); anything not listed there falls back to a generic
    ``"the_name"`` -> ``"The Name"`` conversion, capitalizing a short list of
    common statistical acronyms (``_ACRONYMS``) instead of title-casing them.
    The fallback means a plot never shows a raw ``snake_case`` name just
    because nobody has labeled that particular metric yet -- but a curated
    ``overrides`` dict, kept next to the code that owns those names, should
    still be the primary source for anything that matters.
    """
    if overrides and name in overrides:
        return overrides[name]
    words = name.replace("_", " ").split(" ")
    return " ".join(w.upper() if w.lower() in _ACRONYMS else w.capitalize() for w in words)
