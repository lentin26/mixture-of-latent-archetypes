"""Tests for the low-dim (fraction-subtraction) MoLA fitting pipeline
(``src/experiments/low_dim.py``), specifically ``select_mola_m``/
``fit_mola``'s M-selection. Uses tiny synthetic data (no real download).

Run from the repo root:

    pytest tests/test_low_dim.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.low_dim import fit_mola, select_mola_m


def _tiny_X_and_Q(n_learners=60, n_items=12, n_skills=4, seed=0):
    rng = np.random.default_rng(seed)
    Q = (rng.random((n_items, n_skills)) < 0.5).astype(float)
    Q[Q.sum(axis=1) == 0, 0] = 1.0
    X = (rng.random((n_learners, n_items)) < 0.5).astype(float)
    return X, Q


def test_select_mola_m_returns_a_grid_member_and_covers_every_candidate():
    X, Q = _tiny_X_and_Q()
    m_grid = [2, 3, 4]

    best_m, nll_by_m = select_mola_m(X, Q, m_grid, seed=0, max_iter=5)

    assert best_m in m_grid
    assert set(nll_by_m) == set(m_grid)
    assert all(np.isfinite(v) for v in nll_by_m.values())


def test_select_mola_m_is_seed_deterministic():
    X, Q = _tiny_X_and_Q()
    m_grid = [2, 3, 4]

    best_a, nll_a = select_mola_m(X, Q, m_grid, seed=0, max_iter=5)
    best_b, nll_b = select_mola_m(X, Q, m_grid, seed=0, max_iter=5)

    assert best_a == best_b
    assert nll_a == nll_b


def test_fit_mola_records_selected_m_per_split_when_m_grid_given():
    X, Q = _tiny_X_and_Q(n_learners=80)
    Q_df = pd.DataFrame(Q)
    splits = [
        (pd.DataFrame(X[:60]), pd.DataFrame(X[60:])),
        (pd.DataFrame(X[10:70]), pd.DataFrame(X[70:])),
    ]

    [result] = fit_mola(splits, Q_df, m_grid=[2, 3, 4])

    assert result["selected_m"], "selected_m should have one entry per split"
    assert len(result["selected_m"]) == len(splits)
    assert all(m in [2, 3, 4] for m in result["selected_m"])


def test_fit_mola_defaults_to_fixed_m_without_m_grid():
    X, Q = _tiny_X_and_Q(n_learners=80)
    Q_df = pd.DataFrame(Q)
    splits = [(pd.DataFrame(X[:60]), pd.DataFrame(X[60:]))]

    [result] = fit_mola(splits, Q_df)

    assert result["selected_m"] == [6]
