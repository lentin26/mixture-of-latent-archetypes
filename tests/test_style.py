"""Tests for src/style.py (publication-quality figure styling).

Run from the repo root:

    pytest tests/test_style.py -q
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; no display required

import matplotlib.pyplot as plt
import pytest

from src.style import RC_PUBLICATION, apply, figsize, humanize_label, publication_style, strip_titles


@pytest.fixture(autouse=True)
def _reset_rcparams():
    # apply() mutates global rcParams; make sure one test can't leak into another.
    original = dict(plt.rcParams)
    yield
    plt.rcParams.update(original)
    plt.close("all")


# --------------------------------------------------------------------------- #
# figsize
# --------------------------------------------------------------------------- #
def test_figsize_accepts_a_preset_name():
    width, height = figsize("acm-sigconf-column")
    assert width > 0 and height > 0
    assert height < width  # default aspect < 1


def test_figsize_accepts_a_raw_point_value():
    width, _ = figsize(300.0)
    # 300pt at fraction=1.0 should be a bit under 300/72 inches (LaTeX pt is
    # slightly smaller than an inch/72), definitely not equal to 300.
    assert 4.0 < width < 4.2


def test_figsize_fraction_scales_linearly():
    full = figsize("acm-sigconf-column", fraction=1.0)
    half = figsize("acm-sigconf-column", fraction=0.5)
    assert half[0] == pytest.approx(full[0] / 2)
    assert half[1] == pytest.approx(full[1] / 2)


def test_figsize_aspect_controls_height():
    wide = figsize("acm-sigconf-column", aspect=0.5)
    tall = figsize("acm-sigconf-column", aspect=1.0)
    assert wide[0] == tall[0]  # same width
    assert wide[1] < tall[1]   # smaller aspect -> shorter figure


def test_figsize_rejects_unknown_preset():
    with pytest.raises(KeyError):
        figsize("not-a-real-venue")


# --------------------------------------------------------------------------- #
# publication_style / apply
# --------------------------------------------------------------------------- #
def test_publication_style_applies_inside_and_restores_outside():
    before = plt.rcParams["font.size"]
    with publication_style():
        assert plt.rcParams["font.size"] == RC_PUBLICATION["font.size"]
        assert plt.rcParams["text.usetex"] is False
    assert plt.rcParams["font.size"] == before


def test_publication_style_accepts_overrides():
    with publication_style({"font.size": 14}):
        assert plt.rcParams["font.size"] == 14
        # everything else from RC_PUBLICATION still applies
        assert plt.rcParams["pdf.fonttype"] == RC_PUBLICATION["pdf.fonttype"]


def test_apply_mutates_global_state_until_reset():
    apply()
    assert plt.rcParams["font.size"] == RC_PUBLICATION["font.size"]
    plt.rcParams.update(matplotlib.rcParamsDefault)


# --------------------------------------------------------------------------- #
# strip_titles
# --------------------------------------------------------------------------- #
def test_strip_titles_clears_suptitle_by_default():
    fig, ax = plt.subplots()
    ax.set_title("component 0")
    fig.suptitle("Some descriptive title")

    strip_titles(fig)

    assert fig._suptitle.get_text() == ""
    assert ax.get_title() == "component 0"  # panel titles kept by default


def test_strip_titles_can_also_clear_panel_titles():
    fig, ax = plt.subplots()
    ax.set_title("component 0")
    fig.suptitle("Some descriptive title")

    strip_titles(fig, keep_panel_titles=False)

    assert fig._suptitle.get_text() == ""
    assert ax.get_title() == ""


def test_strip_titles_tolerates_a_figure_with_no_suptitle():
    fig, ax = plt.subplots()
    strip_titles(fig)  # must not raise


# --------------------------------------------------------------------------- #
# humanize_label
# --------------------------------------------------------------------------- #
def test_humanize_label_uses_an_exact_override():
    assert humanize_label("mu_rmse", {"mu_rmse": "Archetype Recovery RMSE"}) == "Archetype Recovery RMSE"


def test_humanize_label_falls_back_to_title_case():
    assert humanize_label("n_learners") == "N Learners"
    assert humanize_label("n_learners", {"mu_rmse": "..."}) == "N Learners"  # unrelated override ignored


def test_humanize_label_keeps_known_acronyms_upper_case_in_fallback():
    assert humanize_label("holdout_auc") == "Holdout AUC"
    assert humanize_label("theta_rmse") == "Theta RMSE"


def test_humanize_label_handles_a_single_word():
    assert humanize_label("sampling") == "Sampling"
