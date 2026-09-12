from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.sparse import csr_matrix, issparse
import time
from tqdm import tqdm
from scipy.special import expit


class _MoLABase:
    """
    EM engine shared by the MoLA estimators.

    Not used directly -- instantiate :class:`src.mola.MoLA`, which layers the
    posterior/predictive read-outs on top of this fitting loop.
    """

    _MODELS = ("MoLA-id", "MoLA-ad")

    def __init__(
        self,
        Q=None,
        n_components: int = 3,
        *,
        max_iter: int = 50,
        tol: float = 1e-5,
        pseudo_likelihood: bool = False,
        mu_prior: tuple = (2, 2),
        theta_prior: tuple = (4, 5),
        pi_prior: float = 1.0,
        random_state: int | None = 746,
        model: str = "MoLA-id",
    ):
        """
        Parameters
        ----------
        Q : (n_items, n_skills) array or sparse matrix, optional
            Binary item->skill map (the Q-matrix). Row-normalized on assignment,
            so the fitted model is the compensatory ("MoLA-id") form. May be left
            ``None`` and supplied later via :meth:`fit`.
        n_components : int
            Number of latent archetypes.
        max_iter : int
            Maximum number of EM iterations.
        tol : float
            Relative negative-log-likelihood change below which EM stops.
        pseudo_likelihood : bool
            Fit with the pseudo-likelihood M-step, which also stores the Beta
            pseudo-counts that the posterior-sampling read-outs need.
        mu_prior, theta_prior : (float, float)
            ``(alpha, beta)`` of the Beta prior on the archetype profiles and on
            item easiness.
        pi_prior : float
            Dirichlet concentration for the mixture weights.
        random_state : int or None
            Seed for parameter initialization.
        model : {"MoLA-id", "MoLA-ad"}
            Response function. ``"MoLA-id"`` (item difficulty) is the default.
        """
        if model not in self._MODELS:
            raise ValueError(f"model must be one of {self._MODELS}, got {model!r}")

        self.n_components = n_components
        self.Q = self._normalize_q(Q)

        self.max_iter = max_iter
        self.tol = tol
        self.pseudo_likelihood = pseudo_likelihood
        self.mu_prior = tuple(mu_prior)
        self.theta_prior = tuple(theta_prior)
        self.pi_prior = pi_prior
        self.random_state = random_state
        self.model = model

        # convergence bookkeeping
        self.nll_trace = []
        self.log_likelihood = None
        self.mu_trace = []
        self.log_post = None

        # inference-time log-scale aggregates, populated lazily (see update_ab_params)
        self.a = None
        self.b = None

        # Beta pseudo-counts, populated only when pseudo_likelihood is True
        self.mu_a = self.mu_b = None
        self.theta_a = self.theta_b = None

    # ------------------------------------------------------------------ #
    # input normalization
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_q(Q):
        """Return ``Q`` as a row-normalized CSR float matrix (rows sum to 1)."""
        if Q is None:
            return None
        Q = csr_matrix(Q, dtype=float)
        row_sums = np.asarray(Q.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        return Q.multiply(1.0 / row_sums[:, None]).tocsr()

    def _prepare_X(self, X):
        """Accept a dense ndarray (NaN = missing) or a SciPy sparse matrix."""
        if isinstance(X, np.ndarray):
            return self._to_sparse(X)
        return X

    def _to_sparse(self, X):
        """
        Convert a dense binary (0/1, NaN = missing) array to the internal CSR
        encoding: observed responses are stored as 0/1, missing entries are
        structural zeros.
        """
        X1 = X.copy()
        X1[X1 == 0] = -1
        X1[np.isnan(X1)] = 0
        X1 = csr_matrix(X1)
        X1.data = (X1.data + 1) / 2
        return X1

    def _init_mu(self, n_skills, n_components):
        """
        Initialize skill proficiency components.
        Set each component to different mastery "levels" with noise.
        """
        eps = np.finfo(float).eps  # smallest positive float
        mu = np.random.uniform(low=eps, high=1 - eps, size=(n_components, n_skills))

        return mu

    def _init_theta(self, n_items):
        """
        Initialize item slipping parameter.
        """
        theta = 0.7 * np.ones((n_items, 1))
        return theta

    def _init_pi(self, n_components):
        """
        Initialize global component probabilities.
        """
        pi = np.ones((n_components, 1)) / n_components
        return pi

    def _init_fit(self, Q_matrix=None):
        """
        Initialize model parameters, stopping criteria variables
        and log likelihood.
        """
        if Q_matrix is not None:
            self.Q = Q_matrix

        # get counts
        n_components = self.n_components
        self.n_items, self.n_skills = self.Q.shape

        # init model params
        if self.random_state:
            # set seed
            np.random.seed(self.random_state)

        self.mu = self._init_mu(self.n_skills, n_components)
        self.theta = self._init_theta(self.n_items)
        self.pi = self._init_pi(n_components)

        # # Update aggregate parmams
        # self.update_ab_params()

        # # append parameter to parameter trace
        # self.mu_trace.append(self.mu)
        
        self.i = 0                                        # init stopping variables
        self.stop = False                                 # init stopping flag
        self.nll_trace = []                               # reset nll trace
        # nll = self.get_nll()
        # self.nll_trace.append(nll)

    def get_params(self):
        return {
            'mu': self.mu,
            'theta': self.theta,
            'pi': self.pi,
            'Q': self.Q,
        }
    
    def get_param_count(self):
        param_count = 0
        param_count = np.prod(self.mu.shape) \
            + np.prod(self.theta.shape) \
            + np.prod(self.pi.shape)
        
        return param_count
    
    def get_model_size_in_mb(self):
        """
        Return size of model (including Q-matrix) in Mb.
        
        :param self: Description
        """
        model_size = 0
        model_size += self.pi.nbytes
        model_size += self.mu.nbytes
        model_size += self.theta.nbytes
        model_size += self.Q.data.nbytes

        return model_size / 1e6
    
    def get_est_model_size_in_mb(self):
        """
        Return dictionary giving the estimated model size in MBs.
        """
        # Get model parameter count
        param_count = self.get_param_count()

        # Assume float 64 dtype for model params
        dtype = np.float64
        # Compute size in bytes per element
        bytes_per_element = np.dtype(dtype).itemsize

        # Total memory in bytes
        total_bytes = param_count * bytes_per_element

        # Optional: convert to MB
        total_mb = total_bytes / (1024**2)

        return total_mb

    def create_X1_X2_X_mask(self, X, A=None):

        X1, X2 = self.create_X1_and_X2(X)
        X_mask = self.create_X_mask(X)

        if A is not None:
            # Apply attempt matrix to response matrices
            X1.data = X1.data * A.data
            X2.data = X2.data * A.data
            X_mask.data = X_mask.data * A.data

        self.X_mask = X_mask
        return X1, X2, X_mask

    def create_X_mask(self, X1):
        """
        Create mask, where True signifies an item attempt.
        X_mask has the same shape as the data X
        """
        X_mask = X1.copy()
        if isinstance(X_mask, csr_matrix):
            X_mask.data[:] = 1
        else:
            X_mask
        return X_mask.astype(bool)

    def create_X1_and_X2(self, X):
        """
         where a 0 element indicates
        a missing datapoint and a 1 indicates a non-missing datapoint.
        """
        X1 = X
        if issparse(X):
            # convert -1 to explicitly store 0s (incorrect responses)
            # Non-stored 0s are considered missing values
            X1.data[X1.data[:] == -1] = 0
            # create copy of item response data
            X2 = X.copy()
            # invert original response matrix for non-missing datapoints
            X2.data = 1 - X2.data
        else:
            X[X == -1] = 0
            X2 = X.copy()
            X2 = 1 - X1

        return X1, X2

    def check_for_stability(self):
        """
        Run tests to check for numerical stability.
        """
        # # check for numerical stability over posterior
        # post = np.exp(self.log_post).sum(axis=1)
        # assert (abs(post - 1) <= 0.001).all, \
        #     f"Posterior does not sum to 1, instead sums to: {post}."

        # check that mixture components sum to 1
        assert np.abs(1 - self.pi.sum()) < 0.001, \
            f"Mixture components do not sum to 1, instead sum to {self.pi.sum()}."
        
        assert np.all((self.theta >= 0) & (self.theta <= 1)), \
            "`theta` has values outside of the range [0, 1]."

    def check_stopping_criteria(self):
        """
        Check for EM convergence.
        """
        # update iter counter
        self.i += 1
        # check max EM iterations criteria 
        if self.i > self.max_iter - 1:
            self.stop = True

        # # check parameter convergence
        # if (np.abs(self.mu_trace[-1] - self.mu_trace[-2]) < 1e-3).all():
        #     self.stop = True

        # check NLL convergence
        if len(self.nll_trace) > 1:
            a, b = self.nll_trace[-1], self.nll_trace[-2] 
            if np.abs(b - a) / a < self.tol:
                self.stop = True

    def get_avg_nll_from_data(self, X):
        # X1 and X2 are item response and inverse item response datum
        X1, X2 = self.create_X1_and_X2(X)

        # Update log likelihood (m, k)
        log_likelihood = self.get_log_likelihood(X1, X2)

        log_pi = np.log(self.pi + 1e-12)  # add small value to prevent log(0)
        log_weighted_likelihood = log_likelihood + log_pi.T  # shape (n_samples, n_components)

        # Use logsumexp for numerical stability
        logsum = logsumexp(log_weighted_likelihood, axis=1)  # shape (n_samples,)

        total_nll = -logsum.sum()

        if isinstance(X, csr_matrix):
            n_obs = len(X.data)
        else:
            n_obs = (~np.isnan(X)).sum()
        return total_nll, n_obs

    def get_nll(self):
        """
        Calculate negative log likelihood p(x). Update log likeihood p(x | z).
        """
        # add beta priors?
        # nll -= (self.mu_prior[0] * np.log(self.mu)).sum().sum()
        # nll -= (self.mu_prior[1] * np.log(1 - self.mu)).sum().sum()

        # self.log_likelihood has shape (n_samples, n_components)
        # and self.pi has shape (n_components,)
        log_pi = np.log(self.pi)  # add small value to prevent log(0)
        log_weighted_likelihood = self.log_likelihood + log_pi.T  # shape (n_samples, n_components)

        # Use logsumexp for numerical stability
        logsum = logsumexp(log_weighted_likelihood, axis=1)  # shape (n_samples,)

        nll = -logsum.sum()
        return nll
    
    def update_ab_params(self, use_norm:bool=True):
        """
        Update aggregate params for MoLA. We are computing 
        self.a = log(a) and self.b = log(b). 

        These pre-computated aggregated artifacts are only used during
        inference, not training.
        """
        a, b = self.compute_ab_params(use_norm=use_norm)
        self.a = a
        self.b = b

    def compute_ab_params(self, **kwargs):
        """Only used during inference"""
        # Compute aggregate parameters a and b for MoLA.
        # mu_t has shape (n_skills, n_components)
        mu_t = kwargs['mu'].T if 'mu' in kwargs else self.mu.T
        theta = kwargs['theta'] if 'theta' in kwargs else self.theta  
        theta = theta.reshape(-1, 1) 

        # Modify Q-matrix
        # n_j = self.Q.sum(axis=1)
        Q_mod = self.Q # / n_j        

        a = np.log(theta) + Q_mod @ np.log(mu_t)
        b = np.log(1 - theta) + Q_mod @ np.log(1 - mu_t)

        if kwargs.get('use_norm', True):
            norm = 0
            norm += theta * (Q_mod @ self.mu.T)
            norm += (1 - theta) * (Q_mod @ (1 - self.mu.T))
            neg_log_norm = -np.log(norm)
            a += neg_log_norm
            b += neg_log_norm

        return a, b
    
    def get_log_posterior(self, X):
        """
        Apply user evidence to compute log posterior.
        """
        # X1 and X2 are item response and inverse item response datum
        X1, X2 = self.create_X1_and_X2(X)

        # compute log posterior
        unnorm_log_post = self.get_log_likelihood(X1, X2) + np.log(self.pi.T)
        # normalize the log posterior using log-sum-exp (responsibilities)
        log_post = unnorm_log_post - logsumexp(unnorm_log_post, axis=1)[:, None]

        return log_post

    def fit(self, X, A=None):
        """
        Fit data using EM algorithm. 
        
        X: sparse SciPy array of shape (n_users, n_items):
            Assumes X has 0, 1 for incorrect, correct, respectively.

        A: sparse SciPy array of shape (n_users, n_items):
            Attempt number for each user on each item.
        """
        # convert dense to sparse if needed
        if isinstance(X, np.ndarray):
            X = self._to_sparse(X)

        # X1 and X2 are item response and inverse item response datum
        self._init_fit()
        X1, X2, X_mask = self.create_X1_X2_X_mask(X, A)
        self.log_likelihood = self.get_log_likelihood(X1, X2)
        
        # iterate until stopping criteria is met 
        while not self.stop:
            self.e_step(X1, X2)
            self.m_step(X1, X_mask, A)
            
            self.check_for_stability()
            self.check_stopping_criteria()
            
            nll = self.get_nll()
            self.nll_trace.append(nll)

    def get_log_likelihood(self, X1, X2):
        """
        Compute the current likelihood.

        A: attempt number matrix
        """
        # Use aggregate "ab" artifacts during inference
        if (self.a is not None) | (self.b is not None):     
            # compute likelihood terms
            L1 = X1 @ self.a
            L2 = X2 @ self.b

            # compute log likelihood
            log_likelihood = L1 + L2
        # Otherwise compute "on-the-fly" (e.g., model training)
        else:
            theta = self.theta.reshape(-1, 1)
            # a and b are shape (J, M), large during training when J is large
            # we can avoid the large intermediary by changing the order of operations
            # We assume that J > K.
            tmp1 = X1 @ self.Q  # shape (N, K)
            tmp2 = X1 @ np.log(self.theta).reshape(-1, 1)  # shape (N, 1)
            L1 = tmp1 @ np.log(self.mu.T) + tmp2  # shape (N, M)

            tmp1 = X2 @ self.Q  # shape (N, K)
            tmp2 = X2 @ np.log(1 - self.theta).reshape(-1, 1)  # shape (N, 1)
            L2 = tmp1 @ np.log(1 - self.mu.T) + tmp2  # shape (N, M)

            # P_{jm} - correct/incorrect response probability for all items j
            # and components m
            log_P1 = np.log(theta) + (self.Q @ np.log(self.mu.T))
            log_P2 = np.log(1 - theta) + (self.Q @ np.log(1 - self.mu.T))

            # Compute the per-feature normalization matrix (J, M)
            # Note: Ensure log_P2 uses log(1-mu) if it represents the "0" case
            log_P_norm = np.logaddexp(log_P1, log_P2) 
            # self.log_P = log_P1 - log_P_norm

            # Subtract from the data terms
            # (N, M) - (M,) broadcasts correctly across the N dimension
            log_likelihood = L1 + L2 # - log_norm_sample
            if not self.pseudo_likelihood:
                # log_P_norm is (J, M)
                # X1 and X2 are (N, J)
                obs_mask = X1 + X2  # Shape (N, J)

                # Use matrix multiplication to sum the normalization 
                # only for observed items per sample
                # (N, J) @ (J, M) -> (N, M)
                log_norm_data = obs_mask @ log_P_norm

                # Now the dimensions (N, M) match L1 + L2
                log_likelihood = (L1 + L2) - log_norm_data

        # return log likelihood
        assert (log_likelihood <= 1e-12).all()
        return log_likelihood

    def e_step(self, X1, X2):
        """
        Compute the responsibility of each latent variable. Note that
        we compute the updated the log likelihood in the 'check_stopping_criteria'
        call to avoid computing it twice per EM iteration.
        """
        # Update log-likelihood
        self.log_likelihood = self.get_log_likelihood(X1, X2)
        # calculate unnormalized log posterior
        unnorm_log_post = self.log_likelihood + np.log(self.pi.T)

        # normalize the log posterior using log-sum-exp
        # Use scipy logsumexp for numerical stability
        self.log_post = unnorm_log_post - logsumexp(unnorm_log_post, axis=1)[:, None]

        # check for numerical stability over posterior responsibilities
        post = np.exp(self.log_post).sum(axis=1)
        assert (abs(post - 1) <= 0.001).all, \
            f"Posterior does not sum to 1, instead sums to: {post}."

    def m_step(self, X1, X_mask, A):
        """
        Update model parameters to maximized expectation of the complete 
        data log-likelihood wrt to the posterior of the latent variable.
        """
        # Compute sufficient statistics
        stats = self.compute_sufficient_stats(X1, X_mask)
        # Update model parameters
        self.update_model_params(*stats)

    def update_model_params(self, I_mk, R_mk, I_jm, R_jm, I_m, R_m):

        # Update Prior component assignment
        self.pi = (I_m + self.pi_prior) / (R_m + self.pi_prior*I_m.shape[0])
        # Update other parameters (psuedo-likelihood)
        if self.pseudo_likelihood:
            
            # Unpack beta priors 
            mu_a, mu_b = self.mu_prior
            theta_a, theta_b = self.theta_prior

            I_j = I_jm.sum(axis=1).reshape(-1, 1)
            R_j = R_jm.sum(axis=1).reshape(-1, 1)

            # Store pseudo correct, incorrect counts
            self.mu_a, self.mu_b = mu_a + I_mk, mu_b + R_mk - I_mk
            self.theta_a, self.theta_b   = I_j + theta_a, R_j - I_j + theta_b

            self.mu = self.mu_a / (self.mu_a + self.mu_b)
            self.theta = self.theta_a / (self.theta_a + self.theta_b)
        else:

            # Collect old variables
            mu_mk = self.mu
            theta_j = self.theta

            # Compute item probabilities
            p_jm = self.compute_P_jm()
            self.mu = self.update_mu(p_jm, mu_mk, I_mk, R_mk)

            # Re-compute item probabilities before difficulty update
            p_jm = self.compute_P_jm()
            self.theta = self.update_theta(p_jm, theta_j, I_jm, R_jm)

        # append parameter to parameter trace
        if False:
            self.mu_trace.append(self.mu)

    def compute_P_jm(self):
        log_P1 = np.log(self.theta) + (self.Q @ np.log(self.mu.T))
        log_P2 = np.log(1 - self.theta) + (self.Q @ np.log(1 - self.mu.T))
        logit = log_P1 - log_P2
        # This is numerically stable for large datasets
        return expit(logit)

    def update_theta(self, p_jm, theta_old, I_jm, R_jm):

        # Compute correct vs total attempts on each item in each component m
        R_j = R_jm.sum(axis=1).reshape(-1, 1)
        I_j = I_jm.sum(axis=1).reshape(-1, 1)

        # 2. Compute the weighted sum of residuals across components
        # (J, M) @ (M,) -> (J,)
        # This represents: sum_m Gamma_m * p_jm
        weighted_p = (p_jm * R_jm).sum(axis=1).reshape(-1, 1)

        # sum_m Gamma_m * (1 - beta_j) 
        # Note: sum_m Gamma_m is just N (total number of samples)
        term_const = R_j * theta_old

        # residual_sum is (J,)
        residual_sum = weighted_p - term_const

        # 3. Compute B_j
        # Stability: prevent division by zero if theta is 0 or 1
        eps = 1e-12
        denom = np.clip(theta_old * (1 - theta_old), eps, None)
        B_matrix = (1 / denom) * residual_sum

        # 4. Quadratic Solver for new beta
        # Coefficients: a*x^2 + b*x + c = 0
        # Using your established logic: [B, -(B+1), 1-mu_avg]
        # Since beta is item-specific, we need an item-specific mu reference
        # mu_avg could be the average mu across skills for that item: (Q_prop @ mu.T)
        # Add priors
        a, b = self.theta_prior

        # B_matrix = 0

        A = B_matrix
        B = -(B_matrix + (R_j + a + b))
        C = (I_j + a) # (J,)

        disc = B**2 - 4 * A * C
        assert (disc >= 0).all()
        sqrt_d = np.sqrt(np.maximum(disc, 0))

        # 2. Compute the stable intermediate variable 'V'
        # np.sign(b) ensures we always add matching signs, preventing cancellation
        V = -0.5 * (B + np.sign(B) * sqrt_d)
        
        # 3. Handle the edge case where b=0 and 4ac=0 concurrently
        # This prevents V from being strictly zero
        V = np.where(V == 0, 1e-15, V)

        # 3. Calculate both possible root paths element-by-element
        root_muller = C / V
        root_standard = V / (A + 1e-15)

        # 4. Compute the target root (Müller's form)
        # As a -> 0, this transitions cleanly to: c / -b (which is I_j / N_total)
        # The correct stable root depends strictly on the sign of B
        new_theta = np.where(B < 0, root_muller, root_standard)

        # Safety fallback for B=0 or numerical edge cases
        # new_theta = np.where(np.abs(a) < 1e-12, I_j / (R_j + 1e-15), new_theta)
        return np.clip(new_theta, 1e-10, 1 - 1e-10)

    def update_mu(self, p_jm, mu_mk, I_mk, R_mk):
        """
        M-step update for mu using a quadratic solver.
        gamma: (N, M) - responsibilities
        Q: (J, K) - skill tags
        n_j: (J,) - skills per item
        p_old: (J, M) - current probabilities
        mu_old: (M, K) - current mu parameters
        theta: (J, 1) or (J, M) - the other parameter in your root eq
        """
        post_t = np.exp(self.log_post).T
        L_mk = ((post_t @ self.X_mask) * p_jm.T) @ self.Q
        
        # 2. Compute the residual sum term (M, K)
        # Using .T to align dimensions for (M, K) result
        res_sum = L_mk - (mu_mk * R_mk)

        # 3. Compute B_mk
        # Stability: Clip mu to avoid division by zero
        eps = 1e-10
        denom = np.clip(mu_mk * (1 - mu_mk), eps, None)
        # B_matrix = (1 / denom) * res_sum

        res_sum /= denom  # res_sum now becomes your B_matrix (0 new memory!)
        B_matrix = res_sum 

        # 4. Solve the Quadratic Equation: 
        # Coefficients: a*x^2 + b*x + c = 0
        a, b = self.mu_prior

        A = B_matrix
        B = -(B_matrix + (R_mk + a + b))
        C = I_mk + a

        disc = B**2 - 4 * A * C
        sqrt_d = np.sqrt(np.maximum(disc, 0))

        # 2. Compute the stable intermediate variable 'V'
        # np.sign(b) ensures we always add matching signs, preventing cancellation
        V = -0.5 * (B + np.sign(B) * sqrt_d)
        
        # 3. Handle the edge case where b=0 and 4ac=0 concurrently
        # This prevents V from being strictly zero
        V = np.where(V == 0, 1e-15, V)
        
        # 4. Compute the target root (Müller's form)
        # As a -> 0, this transitions cleanly to: c / -b (which is I_j / N_total)
        root_muller = C / V
        root_standard = V / (A + 1e-15)

        # The correct stable root depends strictly on the sign of B
        new_mu = np.where(B < 0, root_muller, root_standard)

        # Safety fallback for B=0 or numerical edge cases
        # new_mu = np.where(np.abs(a) < 1e-12, I_mk / (R_mk + 1e-15), new_mu)
        return np.clip(new_mu, 1e-10, 1 - 1e-10)
    
    def update_mu_from_stats(self, B_matrix, R_mk, I_mk):

        # 4. Solve the Quadratic Equation: 
        # Coefficients: a*x^2 + b*x + c = 0
        a, b = self.mu_prior

        A = B_matrix
        B = -(B_matrix + (R_mk + a + b))
        C = I_mk + a

        disc = B**2 - 4 * A * C
        sqrt_d = np.sqrt(np.maximum(disc, 0))

        # 2. Compute the stable intermediate variable 'V'
        # np.sign(b) ensures we always add matching signs, preventing cancellation
        V = -0.5 * (B + np.sign(B) * sqrt_d)
        
        # 3. Handle the edge case where b=0 and 4ac=0 concurrently
        # This prevents V from being strictly zero
        V = np.where(V == 0, 1e-15, V)
        
        # 4. Compute the target root (Müller's form)
        # As a -> 0, this transitions cleanly to: c / -b (which is I_j / N_total)
        root_muller = C / V
        root_standard = V / (A + 1e-15)

        # The correct stable root depends strictly on the sign of B
        new_mu = np.where(B < 0, root_muller, root_standard)

        # Safety fallback for B=0 or numerical edge cases
        # new_mu = np.where(np.abs(a) < 1e-12, I_mk / (R_mk + 1e-15), new_mu)
        self.mu = np.clip(new_mu, 1e-10, 1 - 1e-10)

    def update_theta_from_stats(self, B_matrix, R_jm, I_jm):

        # 4. Quadratic Solver for new beta
        R_j = R_jm.sum(axis=1).reshape(-1, 1)
        I_j = I_jm.sum(axis=1).reshape(-1, 1)

        a, b = self.theta_prior

        A = B_matrix
        B = -(B_matrix + (R_j + a + b))
        C = (I_j + a) # (J,)

        disc = B**2 - 4 * A * C
        assert (disc >= 0).all()
        sqrt_d = np.sqrt(np.maximum(disc, 0))

        # 2. Compute the stable intermediate variable 'V'
        # np.sign(b) ensures we always add matching signs, preventing cancellation
        V = -0.5 * (B + np.sign(B) * sqrt_d)
        
        # 3. Handle the edge case where b=0 and 4ac=0 concurrently
        # This prevents V from being strictly zero
        V = np.where(V == 0, 1e-15, V)

        # 3. Calculate both possible root paths element-by-element
        root_muller = C / V
        root_standard = V / (A + 1e-15)

        # 4. Compute the target root (Müller's form)
        # As a -> 0, this transitions cleanly to: c / -b (which is I_j / N_total)
        # The correct stable root depends strictly on the sign of B
        new_theta = np.where(B < 0, root_muller, root_standard)

        # Safety fallback for B=0 or numerical edge cases
        # new_theta = np.where(np.abs(a) < 1e-12, I_j / (R_j + 1e-15), new_theta)
        self.theta = np.clip(new_theta, 1e-10, 1 - 1e-10)

    def update_pi_from_stats(self, R_m, I_m):
        self.pi = (I_m + self.pi_prior) / (R_m + self.pi_prior*I_m.shape[0])
    
    def update_model_params_incrementally(self, I1, R1, I2, R2, I3, R3):
        # Unpack beta priors 
        mu_a, mu_b = self.mu_prior
        theta_a, theta_b = self.theta_prior

        # Store pseudo correct, incorrect counts
        self.mu_a, self.mu_b = mu_a + I1, mu_b + R1 - I1
        self.theta_a, self.theta_b = I2 + theta_a, R2 - I2 + theta_b

        # Update parameters incrementally
        self.mu = self.mu_a / (self.mu_a + self.mu_b)
        self.theta = self.theta_a / (self.theta_a + self.theta_b)
        self.pi = (I3 + self.pi_prior) / (R3 + self.pi_prior*I3.shape[0])

        # # update aggregate params
        # self.update_ab_params()

    def compute_sufficient_stats(self, X1, X_mask):
        # get posterior
        post = np.exp(self.log_post)
        post_t = post.T

        # counts for mu
        I_mk = post_t @ (X1 @ self.Q)
        R_mk = post_t @ (X_mask @ self.Q)

        # counts for theta (1 - difficulty)
        I_jm = (post_t @ X1).T
        R_jm = (post_t @ X_mask).T

        # assert all(R_j >= I_j), \
        #     "Sufficient statistics error: R2 has values less than I2."

        # counts for pi (component weights)
        I_m = post.sum(axis=0).reshape(-1, 1)
        R_m = X1.shape[0]

        return I_mk, R_mk, I_jm, R_jm, I_m, R_m

    def update_sufficient_stats(self, I_mk, R_mk, I_jm, R_jm, I_m, R_m):
        """
        Update sufficient statistics for batch.
        """
        self.I1 += I_mk
        self.R1 += R_mk
        self.I2 += I_jm.sum(axis=1)
        self.R2 += R_jm.sum(axis=1)
        self.I3 += I_m
        self.R3 += R_m

    def reset_sufficient_stats(self):
        """
        Set all sufficient statistics to 0.
        """
        self.I1 = 0
        self.R1 = 0
        self.I2 = 0
        self.R2 = 0
        self.I3 = 0
        self.R3 = 0

    def get_sufficient_stats(self):
        """
        Return current sufficient statistics.
        """
        return self.I1, self.R1, self.I2, self.R2, self.I3, self.R3
    
    def _pivot_X(self, batch, shape):
        """
        Pivot item response table to shape (n_users, n_items).
        """
        # Pivot item response data
        batch["userId"] = pd.factorize(batch["userId"])[0]
        # 1. Create a sequential count of items for EACH user (0, 1, 2, 3...)
        # Since data is already ordered, this counts their timeline chronologically
        # user_attempt_counts = batch.groupby("userId").cumcount()

        # # 2. Divide by 10 using integer division (//) to create "Window blocks"
        # # Rows 0-9 become block 0, rows 10-19 become block 1, etc.
        # batch["window_id"] = user_attempt_counts // 1

        # # 3. Combine userId and window_id to create a unique block string
        # # Example: "userA_block0", "userA_block1"
        # batch["user_window_key"] = batch["userId"].astype(str) + "_w" + batch["window_id"].astype(str)

        # # 4. Factorize this combined key! 
        # # This gives you a zero-indexed row index for every unique 10-item block in the batch
        # batch["userId"], uniques = pd.factorize(batch["user_window_key"])

        # # update shape (previously assumed 1 user = 1 row, now we have user X windows)
        # shape = (len(uniques), shape[1])

        # # 5. Determine the local matrix shape based on the number of unique 10-item blocks
        # num_local_windows = len(uniques)
        # local_shape = (num_local_windows, shape)  # (Local 10-item Blocks, Total Items)
        X_batch = self._create_sparse_pivot(
            data=batch, 
            shape=shape, 
            row_name='userId', 
            col_name='itemId',
            val_name="isCorrect"
        )

        return X_batch
    
    def _pivot_A(self, batch, shape):
        """
        Pivot user itempt table to shape (n_users, n_items).
        """
        batch["attemptsRemaining"] = \
            batch.groupby('userId')['attemptNum'].transform('max') - batch['attemptNum'] + 1
        A_batch = self._create_sparse_pivot(
            data=batch,
            shape=shape, 
            row_name='userId', 
            col_name='itemId',
            val_name='attemptsRemaining'
        )

        return A_batch
    
    def _process_batch(self, part):
        """
        Read Dask batch into memotry and create sparse pivots.

        Return sparse pivots X and A.
        """
        # Convert a single partition to pandas
        batch = part.compute().reset_index()  # only this partition is loaded into memory

        # Get counts
        n_users, n_items = batch.userId.nunique(), self.Q.shape[0]
        shape = (n_users, n_items)

        # Create sparse pivots from dataframes
        X_batch = self._pivot_X(batch, shape)
        A_batch = self._pivot_A(batch, shape)

        return X_batch, A_batch
    

    def partial_fit(self, df, Q_matrix):
        """
        Read batch files from S3. Iterate through all batches, accumulating sufficient
        statistics in the E-step and calculating the M-step at the end of each full 
        iteration.

        train_parts: training partitions.
        Q: ndarray of shape (n_items, n_skills): Q-matrix.
        A: ndarray of shape (n_users, n_items): Attempt number matrix.
        """
        print('Starting MoLA training...')

        partition_sizes = df.map_partitions(len).compute().tolist()
        print("Min. partition item response count:", min(partition_sizes))
        print("Max. partition item response count:", max(partition_sizes))

        train_parts = df.to_delayed()
        print('Training batch count:', len(train_parts))
        self.Q = Q_matrix
        self.nll_trace = []
        iter_times = []
        tot_batch_process_time = 0

        i = 1
        self._init_fit()
        while not self.stop:
            st = time.time()

            nll = 0 
            batch_process_time = 0
            self.reset_sufficient_stats()

            # Iterate through entire dataset
            for part in train_parts:
                # process batch
                st1 = time.time()
                X_batch, A_batch = self._process_batch(part)
                batch_process_time += time.time() - st1
                
                # tqdm.write('Creating mask...')
                X1, X2, X_mask = self.create_X1_X2_X_mask(X_batch, A_batch)

                # E-step
                self.e_step(X1, X2)

                # Update sufficient statistics
                stats = self.compute_sufficient_stats(X1, X_mask)
                self.update_sufficient_stats(*stats)

                # Update negative-log-likelihood
                nll += self.get_nll()

            # M-step
            stats = self.get_sufficient_stats()
            self.update_model_params(*stats)

            # Check stopping criteria
            self.check_for_stability()
            self.check_stopping_criteria()

            # Stop timer
            iter_time = time.time() - st

            self.nll_trace.append(nll)
            iter_times.append(iter_time)
            tot_batch_process_time += batch_process_time
            print(f' >iter_num={i}, nll={nll}, iter_time={iter_time}, total_batch_process_time={batch_process_time}')
            i += 1
        
        # Store training metrics for downstream logging
        self.metrics = {
            'n_batches': len(train_parts),
            "em_iters": i,
            "nll": nll,
            "train_time": sum(iter_times),
            "total_batch_process_time": tot_batch_process_time,
            "min_response_count": min(partition_sizes),
            "max_partition_count": max(partition_sizes)
        }

    def batch_fit(self, bucket, etl_source, etl_date):
        """
        Deprecated: replaced with `parital_fit`.

        Read batch files from S3. Iterate through all batches, accumulating sufficient
        statistics in the E-step and calculating the M-step at the end of each full 
        iteration.
        """
        print('Starting MoLA training...')

        # Read Q-matrix from S3
        key_q = f'{etl_source}/{etl_date}/outputs/artifacts/q.npz'
        self.Q = self.read_sparse_npz_from_s3(bucket, key_q)

        prefix = f'{etl_source}/{etl_date}/processed/train'
        keys = self.s3_list_objects(bucket, prefix)

        i = 1
        self._init_fit()
        while not self.stop:
            st = time.time()

            nll = 0
            self.reset_sufficient_stats()
            # Iterate through entire dataset
            for key in tqdm(keys):
                # Fetch batch
                X_batch = self.read_sparse_npz_from_s3(bucket, key)
                # tqdm.write('Creating mask...')
                X1, X2, X_mask = self.create_X1_X2_X_mask(X_batch)

                # E-step
                self.e_step(X1, X2)

                # Update sufficient statistics
                stats = self.compute_sufficient_stats(X1, X_mask)
                self.update_sufficient_stats(*stats)

                # Update negative-log-likelihood
                nll += self.get_nll()

            # M-step
            self.compute_sufficient_stats(X1, X_mask)

            stats = self.get_sufficient_stats()
            self.update_model_params(*stats)

            # Check stopping criteria
            self.check_for_stability()
            self.check_stopping_criteria()

            self.nll_trace.append(nll)

            print(f' >iter_num={i}, nll={nll}, iter_time={time.time() - st})  #, batch_size={X_batch.shape[0]}')
            i += 1

    def _create_sparse_pivot(self, data, shape, row_name, col_name, val_name=None):
        """
        Convert data into sparse one-hot encoded arrays.
        """
        # assume 1 column for values if not specified
        if val_name is None:
            data[val_name] = 1

        # remove nulls
        data = data[~data[[row_name, col_name]].isna().any(axis=1)]
        # drop duplicates
        data = data[~data[[row_name, col_name]].duplicated()]

        # get unique row and column indices
        # Use .ravel() to ensure 1-D arrays (scipy.sparse requirement)
        rows = data[row_name].values.ravel()
        cols = data[col_name].values.ravel()
        values = data[val_name].values.ravel()

        # create sparse binary matrix
        return csr_matrix((values, (rows, cols)), shape=shape)
