"""Tests for the MoLA model package (``src/mola``).

Run from the repo root:

    pytest tests/test_mola.py -q

Every test fits on a deliberately tiny problem so the full EM path runs in a
fraction of a second.  The suite covers ``Train`` (construction, the EM loop,
stopping/stability checks, sufficient-statistic bookkeeping) and ``MoLA`` (the
posterior, predictive and psi/difficulty read-outs layered on top).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from src.mola import MoLA, Train

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
N_LEARNERS = 250
N_ITEMS = 15
N_SKILLS = 4
N_COMPONENTS = 3


def _toy_q(rng: np.random.Generator) -> np.ndarray:
    """A single-skill-per-item Q-matrix with every skill used at least once."""
    Q = np.zeros((N_ITEMS, N_SKILLS))
    Q[np.arange(N_ITEMS), rng.integers(0, N_SKILLS, size=N_ITEMS)] = 1.0
    for k in range(N_SKILLS):
        if Q[:, k].sum() == 0:
            Q[rng.integers(0, N_ITEMS), k] = 1.0
    return Q


def _toy_responses(rng: np.random.Generator) -> np.ndarray:
    """Latent-class response matrix with ~20% missing entries (dense, NaN-coded)."""
    mu_true = rng.uniform(0.1, 0.9, size=(N_COMPONENTS, N_SKILLS))
    z = rng.integers(0, N_COMPONENTS, size=N_LEARNERS)
    Q = _toy_q(rng)
    prob = (mu_true[z] @ Q.T) / Q.sum(axis=1)
    X = (rng.random((N_LEARNERS, N_ITEMS)) < prob).astype(float)
    X[rng.random((N_LEARNERS, N_ITEMS)) < 0.2] = np.nan
    return X


@pytest.fixture
def toy() -> dict:
    rng = np.random.default_rng(0)
    Q = _toy_q(rng)
    X = _toy_responses(np.random.default_rng(1))
    return {"Q": Q, "X": X}


def _make_model(toy: dict, **overrides) -> MoLA:
    params = {
        "Q_matrix": toy["Q"],
        "n_components": N_COMPONENTS,
        "n_iter": 10,
        "random_seed": 1,
        "use_psuedo_likelihood": False,
    }
    params.update(overrides)
    return MoLA(**params)


@pytest.fixture
def fitted(toy: dict) -> MoLA:
    model = _make_model(toy)
    model.fit(toy["X"])
    return model


@pytest.fixture
def fitted_pseudo(toy: dict) -> MoLA:
    model = _make_model(toy, use_psuedo_likelihood=True)
    model.fit(toy["X"])
    return model


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def test_defaults_when_no_kwargs():
    model = MoLA()
    assert model.n_components == 3
    assert model.model == "MoLA-id"
    assert model.Q is None
    assert model.a is None and model.b is None


def test_invalid_model_name_raises():
    with pytest.raises(Exception, match="Model must be one of"):
        MoLA(model="not-a-model")


def test_dense_q_matrix_is_row_normalized(toy):
    model = MoLA(Q_matrix=toy["Q"])
    row_sums = np.asarray(model.Q.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0)


def test_sparse_q_matrix_is_kept_as_passed(toy):
    sparse_q = csr_matrix(toy["Q"])
    model = MoLA(Q_matrix=sparse_q)
    assert issparse(model.Q)
    # untouched: single-skill rows still sum to exactly 1, multi-skill would not
    np.testing.assert_array_equal(model.Q.toarray(), toy["Q"])


# --------------------------------------------------------------------------- #
# parameter initialization
# --------------------------------------------------------------------------- #
def test_init_fit_sets_shapes_and_stopping_state(toy):
    model = _make_model(toy)
    model._init_fit()

    assert model.n_items == N_ITEMS
    assert model.n_skills == N_SKILLS
    assert model.mu.shape == (N_COMPONENTS, N_SKILLS)
    assert model.theta.shape == (N_ITEMS, 1)
    assert model.pi.shape == (N_COMPONENTS, 1)
    assert np.all((model.mu > 0) & (model.mu < 1))
    assert np.all((model.theta > 0) & (model.theta < 1))
    assert np.isclose(model.pi.sum(), 1.0)
    assert model.i == 0
    assert model.stop is False
    assert model.nll_trace == []


def test_init_is_deterministic_given_random_seed(toy):
    a, b = _make_model(toy), _make_model(toy)
    a._init_fit()
    b._init_fit()
    np.testing.assert_array_equal(a.mu, b.mu)


# --------------------------------------------------------------------------- #
# X transforms
# --------------------------------------------------------------------------- #
def test_convert_dense_to_sparse_roundtrip(fitted, toy):
    X = toy["X"]
    sparse_x = fitted.convert_dense_to_sparse(X)

    observed = np.isfinite(X)
    assert sparse_x.nnz == observed.sum()
    assert set(np.unique(sparse_x.data)).issubset({0.0, 1.0})
    # correct responses survive the round trip; missing entries read back as 0
    dense_back = sparse_x.toarray()
    np.testing.assert_array_equal(dense_back[observed], np.nan_to_num(X)[observed])


def test_create_X1_and_X2_are_complementary_on_observed_entries(fitted, toy):
    sparse_x = fitted.convert_dense_to_sparse(toy["X"])
    X1, X2 = fitted.create_X1_and_X2(sparse_x.copy())
    combined = (X1 + X2).toarray()
    observed = np.isfinite(toy["X"])
    np.testing.assert_array_equal(combined[observed], 1.0)
    np.testing.assert_array_equal(combined[~observed], 0.0)


def test_create_X_mask_is_boolean_and_marks_every_observation(fitted, toy):
    sparse_x = fitted.convert_dense_to_sparse(toy["X"])
    mask = fitted.create_X_mask(sparse_x.copy())
    assert mask.dtype == bool
    assert mask.nnz == np.isfinite(toy["X"]).sum()
    assert np.all(mask.data)


# --------------------------------------------------------------------------- #
# the EM loop
# --------------------------------------------------------------------------- #
def test_fit_produces_parameters_in_range(fitted):
    assert fitted.mu.shape == (N_COMPONENTS, N_SKILLS)
    assert np.all((fitted.mu >= 0.0) & (fitted.mu <= 1.0))

    assert fitted.theta.shape == (N_ITEMS, 1)
    assert np.all((fitted.theta >= 0.0) & (fitted.theta <= 1.0))

    assert fitted.pi.shape == (N_COMPONENTS, 1)
    assert np.all(fitted.pi >= 0.0)
    assert np.isclose(fitted.pi.sum(), 1.0)


def test_fit_records_a_finite_nll_trace(fitted):
    assert 1 <= len(fitted.nll_trace) <= fitted.n_iter
    assert np.all(np.isfinite(fitted.nll_trace))


def test_fit_is_deterministic_given_random_seed(toy):
    a, b = _make_model(toy), _make_model(toy)
    a.fit(toy["X"])
    b.fit(toy["X"])
    np.testing.assert_allclose(a.mu, b.mu)
    np.testing.assert_allclose(a.theta, b.theta)
    np.testing.assert_allclose(a.pi, b.pi)
    np.testing.assert_allclose(a.nll_trace, b.nll_trace)


def test_fit_accepts_presparsified_input(toy):
    dense_fit = _make_model(toy)
    dense_fit.fit(toy["X"])

    sparse_fit = _make_model(toy)
    sparse_fit.fit(sparse_fit.convert_dense_to_sparse(toy["X"]))

    np.testing.assert_allclose(dense_fit.mu, sparse_fit.mu)


@pytest.mark.parametrize("use_pseudo", [False, True], ids=["exact", "pseudo"])
def test_training_converges_monotonically(toy, use_pseudo):
    """EM must never increase the negative log-likelihood from one sweep to the next."""
    model = _make_model(toy, n_iter=30, use_psuedo_likelihood=use_pseudo)
    model.fit(toy["X"])

    trace = np.asarray(model.nll_trace, dtype=float)
    assert trace.size >= 2
    steps = np.diff(trace)
    # allow only floating-point-scale slack, not real uphill moves
    tol = 1e-6 * np.abs(trace).max()
    assert np.all(steps <= tol), f"NLL increased mid-training: max step {steps.max():.3e}"


def test_convergence_below_tolerance_stops_before_max_iters(toy):
    model = _make_model(toy, n_iter=500, tol=1e-2)
    model.fit(toy["X"])
    assert len(model.nll_trace) < 500


def test_max_iterations_caps_the_loop(toy):
    model = _make_model(toy, n_iter=4, tol=0.0)
    model.fit(toy["X"])
    assert len(model.nll_trace) == 4


# --------------------------------------------------------------------------- #
# stability / stopping helpers
# --------------------------------------------------------------------------- #
def test_check_for_stability_passes_after_a_clean_fit(fitted):
    fitted.check_for_stability()  # must not raise


def test_check_for_stability_rejects_unnormalized_pi(fitted):
    fitted.pi = fitted.pi * 2.0
    with pytest.raises(AssertionError, match="Mixture components"):
        fitted.check_for_stability()


def test_check_for_stability_rejects_out_of_range_theta(fitted):
    fitted.theta = fitted.theta.copy()
    fitted.theta[0, 0] = 1.5
    with pytest.raises(AssertionError, match="theta"):
        fitted.check_for_stability()


def test_check_stopping_criteria_trips_at_max_iterations(toy):
    model = _make_model(toy, n_iter=3)
    model._init_fit()
    model.nll_trace = [10.0]
    for _ in range(3):
        model.check_stopping_criteria()
    assert model.stop is True


def test_check_stopping_criteria_trips_on_flat_nll(toy):
    model = _make_model(toy, n_iter=100, tol=1e-4)
    model._init_fit()
    model.nll_trace = [100.0, 100.0]
    model.check_stopping_criteria()
    assert model.stop is True


# --------------------------------------------------------------------------- #
# likelihood / posterior read-outs
# --------------------------------------------------------------------------- #
def test_get_log_likelihood_is_nonpositive(fitted, toy):
    sparse_x = fitted.convert_dense_to_sparse(toy["X"])
    X1, X2 = fitted.create_X1_and_X2(sparse_x.copy())
    ll = fitted.get_log_likelihood(X1, X2)
    assert ll.shape == (N_LEARNERS, N_COMPONENTS)
    assert np.all(ll <= 1e-12)


def test_get_posterior_rows_are_probability_vectors(fitted, toy):
    post = fitted.get_posterior(fitted.convert_dense_to_sparse(toy["X"]))
    assert post.shape == (N_LEARNERS, N_COMPONENTS)
    assert np.all((post >= 0.0) & (post <= 1.0))
    np.testing.assert_allclose(post.sum(axis=1), 1.0, rtol=1e-6)


def test_get_posterior_is_exp_of_log_posterior(fitted, toy):
    sparse_x = fitted.convert_dense_to_sparse(toy["X"])
    np.testing.assert_allclose(
        fitted.get_posterior(sparse_x),
        np.exp(fitted.get_log_posterior(sparse_x)),
    )


def test_normalize_log_probabilities_yields_a_normalized_posterior(fitted, toy):
    sparse_x = fitted.convert_dense_to_sparse(toy["X"])
    unnorm = fitted.get_unnormalized_log_post(sparse_x)
    assert unnorm.shape == (N_LEARNERS, N_COMPONENTS)
    norm = np.exp(fitted.normalize_log_probabilities(unnorm))
    np.testing.assert_allclose(norm.sum(axis=1), 1.0, rtol=1e-6)


def test_get_avg_nll_from_data_returns_total_and_count(fitted, toy):
    sparse_x = fitted.convert_dense_to_sparse(toy["X"])
    total_nll, n_obs = fitted.get_avg_nll_from_data(sparse_x)
    assert np.isfinite(total_nll)
    assert total_nll > 0.0
    assert n_obs == np.isfinite(toy["X"]).sum()


# --------------------------------------------------------------------------- #
# predictive read-outs
# --------------------------------------------------------------------------- #
def test_pred_item_probas_are_in_the_unit_interval(fitted, toy):
    probs = fitted.pred_item_probas(toy["X"])
    assert probs.shape == (N_LEARNERS, N_ITEMS)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_pred_item_probas_accepts_dense_or_sparse(fitted, toy):
    dense = fitted.pred_item_probas(toy["X"])
    sparse = fitted.pred_item_probas(fitted.convert_dense_to_sparse(toy["X"]))
    np.testing.assert_allclose(dense, sparse)


@pytest.mark.xfail(
    reason="pred_item_probas_from_resp does not transpose a/b on the item_idxs path",
    strict=True,
)
def test_pred_item_probas_item_subset_matches_full(fitted, toy):
    idxs = [0, 3, 7, 11]
    full = fitted.pred_item_probas(toy["X"])
    subset = fitted.pred_item_probas(toy["X"], item_idxs=idxs)
    assert subset.shape == (N_LEARNERS, len(idxs))
    np.testing.assert_allclose(subset, full[:, idxs])


def test_pred_item_probas_from_resp_identity_gives_component_curves(fitted):
    eye = np.eye(N_COMPONENTS)
    curves = fitted.pred_item_probas_from_resp(eye)
    assert curves.shape == (N_COMPONENTS, N_ITEMS)
    assert np.all((curves >= 0.0) & (curves <= 1.0))


def test_get_pred_nll_is_nonnegative_and_finite(fitted, toy):
    value = fitted.get_pred_nll(np.nan_to_num(toy["X"]))
    assert np.isfinite(value)
    assert value >= 0.0


# --------------------------------------------------------------------------- #
# covariance / correlation helpers
# --------------------------------------------------------------------------- #
def test_get_item_cov_is_square_and_symmetric(fitted):
    cov = fitted.get_item_cov()
    assert cov.shape == (N_ITEMS, N_ITEMS)
    np.testing.assert_allclose(cov, cov.T, atol=1e-10)
    assert np.all(np.diag(cov) >= -1e-12)


def test_convert_cov_to_corr_has_unit_diagonal(fitted):
    corr = fitted.convert_cov_to_corr(fitted.get_item_cov())
    np.testing.assert_allclose(np.diag(corr), 1.0)


def test_get_skill_score_cov_is_skill_by_skill_and_symmetric(fitted):
    cov = fitted.get_skill_score_cov()
    assert cov.shape == (N_SKILLS, N_SKILLS)
    np.testing.assert_allclose(cov, cov.T, atol=1e-10)


# --------------------------------------------------------------------------- #
# difficulty / ability read-outs
# --------------------------------------------------------------------------- #
def test_get_item_difficulty_is_one_minus_theta(fitted):
    diff = fitted.get_item_difficulty()
    assert diff.shape == (N_ITEMS,)
    np.testing.assert_allclose(diff, (1.0 - fitted.theta).ravel())


def test_get_item_difficulty_honours_item_subset(fitted):
    idxs = [2, 5, 9]
    np.testing.assert_allclose(
        fitted.get_item_difficulty(idxs), fitted.get_item_difficulty()[idxs]
    )


def test_get_user_ability_shape(fitted, toy):
    ability = fitted.get_user_ability(fitted.convert_dense_to_sparse(toy["X"]))
    assert ability.shape == (N_LEARNERS, N_ITEMS)
    assert np.all(np.isfinite(ability))


# --------------------------------------------------------------------------- #
# model metadata
# --------------------------------------------------------------------------- #
def test_get_params_exposes_the_fitted_arrays(fitted):
    params = fitted.get_params()
    assert set(params) == {"mu", "theta", "pi", "Q_matrix"}
    assert params["mu"] is fitted.mu


def test_get_param_count_sums_free_parameters(fitted):
    expected = fitted.mu.size + fitted.theta.size + fitted.pi.size
    assert fitted.get_param_count() == expected


def test_model_size_helpers_are_positive(fitted):
    assert fitted.get_model_size_in_mb() > 0.0
    assert fitted.get_est_model_size_in_mb() > 0.0


def test_config_model_overrides_hyperparameters(fitted):
    fitted.config_model(n_components=7, tol=1e-3, n_iter=42)
    assert fitted.n_components == 7
    assert fitted.tol == 1e-3
    assert fitted.n_iter == 42


# --------------------------------------------------------------------------- #
# inference-time aggregate ("ab") parameters
# --------------------------------------------------------------------------- #
def test_update_ab_params_populates_log_scale_arrays(fitted):
    fitted.update_ab_params(use_norm=True)
    assert fitted.a.shape == (N_ITEMS, N_COMPONENTS)
    assert fitted.b.shape == (N_ITEMS, N_COMPONENTS)
    assert np.all(np.isfinite(fitted.a)) and np.all(np.isfinite(fitted.b))


def test_get_log_likelihood_uses_ab_path_once_populated(fitted, toy):
    sparse_x = fitted.convert_dense_to_sparse(toy["X"])
    X1, X2 = fitted.create_X1_and_X2(sparse_x.copy())
    before = fitted.get_log_likelihood(X1, X2)

    fitted.update_ab_params(use_norm=True)
    X1, X2 = fitted.create_X1_and_X2(fitted.convert_dense_to_sparse(toy["X"]).copy())
    after = fitted.get_log_likelihood(X1, X2)

    assert after.shape == before.shape
    assert np.all(after <= 1e-12)


# --------------------------------------------------------------------------- #
# pseudo-likelihood-only read-outs (need the Beta pseudo-counts)
# --------------------------------------------------------------------------- #
def test_pseudo_likelihood_fit_sets_beta_pseudocounts(fitted_pseudo):
    for attr in ("mu_a", "mu_b", "theta_a", "theta_b"):
        assert getattr(fitted_pseudo, attr) is not None
    assert fitted_pseudo.mu_a.shape == (N_COMPONENTS, N_SKILLS)
    assert fitted_pseudo.theta_a.shape == (N_ITEMS, 1)


def test_get_proficiency_cov_is_skill_by_skill(fitted_pseudo, toy):
    sparse_x = fitted_pseudo.convert_dense_to_sparse(toy["X"])
    assert fitted_pseudo.get_proficiency_cov(sparse_x).shape == (N_SKILLS, N_SKILLS)
    assert fitted_pseudo.get_proficiency_cov_from_comp(0).shape == (N_SKILLS, N_SKILLS)


def test_sample_user_skill_posterior_shape_and_range(fitted_pseudo, toy):
    sparse_x = fitted_pseudo.convert_dense_to_sparse(toy["X"])
    draws = fitted_pseudo.sample_user_skill_posterior(sparse_x[0], None, n_samples=16)
    assert draws.shape == (16, N_SKILLS)
    assert np.all((draws >= 0.0) & (draws <= 1.0))


def test_get_skill_posterior_returns_a_density_grid(fitted_pseudo, toy):
    sparse_x = fitted_pseudo.convert_dense_to_sparse(toy["X"])
    density = fitted_pseudo.get_skill_posterior(sparse_x[0], skill_idx=0)
    assert density.shape == (100,)
    assert np.all(density >= 0.0)


# --------------------------------------------------------------------------- #
# component-redundancy diagnostics
# --------------------------------------------------------------------------- #
def test_get_effective_attr_dim_within_participation_ratio_bounds(fitted):
    value = fitted.get_effective_attr_dim()
    assert np.isfinite(value)
    ceiling = min(N_COMPONENTS - 1, N_SKILLS)
    assert 1.0 <= value <= ceiling + 1e-9


def test_get_effective_attr_dim_collapses_to_one_for_duplicate_profiles(fitted):
    fitted.mu = np.tile(fitted.mu[[0]], (N_COMPONENTS, 1))
    assert fitted.get_effective_attr_dim() == pytest.approx(1.0)


def test_get_effective_attr_dim_rises_when_profiles_spread_out(fitted):
    spread = fitted.mu.copy()
    fitted.mu = np.tile(spread[[0]], (N_COMPONENTS, 1))
    collapsed = fitted.get_effective_attr_dim()
    fitted.mu = spread
    assert fitted.get_effective_attr_dim() > collapsed


def test_get_effective_attr_dim_handles_single_component(toy):
    model = _make_model(toy, n_components=1)
    model.mu = np.full((1, N_SKILLS), 0.5)
    assert model.get_effective_attr_dim() == 1.0


def test_component_redundancy_report_structure_and_ranges(fitted, toy):
    report = fitted.component_redundancy_report(toy["X"])

    assert report["effective_attr_dim"] == fitted.get_effective_attr_dim()
    assert 1.0 <= report["effective_n_components"] <= N_COMPONENTS + 1e-9

    i, j = report["closest_pair"]
    assert 0 <= i < j < N_COMPONENTS
    assert report["closest_pair_distance"] >= 0.0

    assert report["weight_argmin"] == int(fitted.pi.ravel().argmin())

    usage = report["posterior_usage"]
    assert usage.shape == (N_COMPONENTS,)
    np.testing.assert_allclose(usage.sum(), 1.0, rtol=1e-6)


def test_component_redundancy_report_omits_usage_without_responses(fitted):
    assert "posterior_usage" not in fitted.component_redundancy_report()


def test_component_redundancy_report_flags_duplicated_profile(fitted):
    fitted.mu = fitted.mu.copy()
    fitted.mu[2] = fitted.mu[0]
    report = fitted.component_redundancy_report()
    assert report["closest_pair"] == (0, 2)
    assert report["closest_pair_distance"] == pytest.approx(0.0)
    assert report["effective_attr_dim"] < N_COMPONENTS - 1


def test_component_redundancy_report_flags_underused_component(fitted):
    fitted.pi = np.array([[0.98], [0.01], [0.01]])
    report = fitted.component_redundancy_report()
    assert report["weight_argmin"] in (1, 2)
    assert report["weight_min"] == pytest.approx(0.01)
    assert report["effective_n_components"] < 1.5


# --------------------------------------------------------------------------- #
# known-broken methods — xfail so a future fix flips them green
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(reason="MoLA.get_mu_slice is referenced but never defined", strict=True)
def test_pred_skill_probas(fitted, toy):
    fitted.pred_skill_probas(fitted.convert_dense_to_sparse(toy["X"]))


@pytest.mark.xfail(reason="MoLA.get_mu_slice is referenced but never defined", strict=True)
def test_get_prior_exp_skill_prof(fitted):
    fitted.get_prior_exp_skill_prof()
