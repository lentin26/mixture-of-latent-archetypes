
import numpy as np
from scipy.special import logsumexp, logit
from scipy.stats import beta
from scipy.sparse import SparseEfficiencyWarning, csr_matrix
import warnings
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
from src.mola import Train


class MoLA(Train):
    """
    MoLA with item difficulty (Lentini 2025).

    Theta is related to difficulty by the following relationship:
        theta = (1 - d_j)^{n_j}
    """
    def __init__(self, **kwargs):
        Train.__init__(self, **kwargs)

    def get_item_cov(self, skill_idxs=None):
        """
        Compute item covaraince matrix.
        """
        M, J= self.n_components, self.Q.shape[0]       

        # item-response probabilities for each distinct profile
        post = np.diag(np.ones(M))
        sigma = self.pred_item_probas_from_resp(post)

        # Sigma_m as diagonals (M, J, J)
        Sigma = np.zeros((M, J, J))
        rows = np.arange(J)
        Sigma[np.arange(M)[:, None], rows, rows] = sigma * (1 - sigma)

        # Outer products sigma_m sigma_m^T (M, J, J)
        outer = sigma[:, :, None] * sigma[:, None, :]

        pi = self.pi.ravel()
        # Weighted sum over components
        weighted_sum = np.tensordot(pi, Sigma + outer, axes=([0], [0]))  # shape (J, J)

        # Compute E[sigma]
        E_sigma = (pi[:, None] * sigma).sum(axis=0)  # shape (J,)
        cov_x = 4 * (weighted_sum - np.outer(E_sigma, E_sigma))
        
        return cov_x
    
    def convert_cov_to_corr(self, cov):
        """
        Convert covariance matrix to correlation matrix
        """
        std = np.sqrt(np.diag(cov))
        return cov / np.outer(std, std)           

    def get_skill_score_cov(self, skill_idxs:list=None):
        """
        Calculate skill expected raw score covariance matrix.
        """
        L = self.get_item_cov(skill_idxs)
        Q = self.Q

        return Q.T @ L @ Q

    def get_exp_skill_correct_attempt_corr(self, skill_idxs):
        """
        Calculate expected skill correct attempt score covariance matrix.
        """
        # attribute covariance
        cov = self.get_exp_skill_score_cov(skill_idxs)
        # convert cov into corr matrix
        R = self.convert_cov_to_corr(cov)    

        return R
    
    def get_skill_score_cor(self, m, skill_idxs=None):
        S = self.get_skill_score_cov(m, skill_idxs)
        return self.convert_cov_to_corr(S)

    def get_item_corr(self, skill_idxs):
        """
        Calculate attribute covariance matrix.
        """
        # compute covariance
        Λ = self.get_item_cov(self.Q, skill_idxs) 
        # convert cov into corr matrix
        R = self.convert_cov_to_corr(Λ) 

        return R

    def get_proficiency_cov(self, X, skill_idxs:list=None):
        g = self.get_posterior(X)

        if skill_idxs is not None:
            mu_a, mu_b = self.mu_a[:, skill_idxs], self.mu_b[:, skill_idxs]
        else:
             mu_a, mu_b = self.mu_a, self.mu_b
             
        mu_var = mu_a * mu_b / ((mu_a + mu_b)**2 * (mu_a + mu_b + 1))
        mu_exp = mu_a / (mu_a + mu_b)
        psi_exp = g @ mu_exp

        P = psi_exp.T @ psi_exp  

        # Expand mu_exp into covariance-like matrices
        M = np.einsum("ij,ik->ijk", mu_exp, mu_exp)  # shape (n_skills, d, d)

        # Add diagonal variances
        M += np.array([np.diag(v) for v in mu_var])  # shape (n_skills, d, d)

        # Weighted sum across skills
        S = np.einsum("i,ijk->jk", g[0], M)
        S -= P

        return S
    
    def get_proficiency_cov_from_comp(self, m=0, skill_idxs:list=None):
        """
        Compute covariance due to a single component. 
        
        :param self: Description
        :param m: Description
        :param skill_idxs: Description
        :type skill_idxs: list
        """
        # Collapse responsbilities to a single component
        g = np.zeros(shape=(1, self.n_components))
        g[0, m] = 1

        if skill_idxs is not None:
            mu_a, mu_b = self.mu_a[:, skill_idxs], self.mu_b[:, skill_idxs]
        else:
             mu_a, mu_b = self.mu_a, self.mu_b
             
        mu_var = mu_a * mu_b / ((mu_a + mu_b)**2 * (mu_a + mu_b + 1))
        mu_exp = mu_a / (mu_a + mu_b)
        psi_exp = g @ mu_exp

        P = psi_exp.T @ psi_exp  

        # Expand mu_exp into covariance-like matrices
        M = np.einsum("ij,ik->ijk", mu_exp, mu_exp)  # shape (n_skills, d, d)

        # Add diagonal variances
        M += np.array([np.diag(v) for v in mu_var])  # shape (n_skills, d, d)

        # Weighted sum across skills
        S = np.einsum("i,ijk->jk", g[0], M)
        S -= P

        return S
    
    def get_unnormalized_log_post(self, X, item_idxs:list=None):
        """
        Get unnormalized responsbilties.
        """
        X1, X2 = self.create_X1_and_X2(X)
        # compute log posterior
        unnorm_log_post = self.get_log_likelihood(X1, X2) + np.log(self.pi.T)

        return unnorm_log_post
    
    def normalize_log_probabilities(self, unnormalized_proba):
        """
        Return normalized probabilities from unnormalized log probabilities.
        """
        log_post = unnormalized_proba - logsumexp(unnormalized_proba, axis=1)[:, None]

        return log_post  # np.exp(log_post)

    # def get_log_posterior(self, X):
    #     """
    #     Apply user evidence to compute log posterior.
    #     """
    #     X1, X2 = self.create_X1_and_X2(X)
    #     # compute log posterior
    #     unnorm_log_post = self.get_log_likelihood(X1, X2) + np.log(self.pi.T)
    #     log_post = unnorm_log_post - logsumexp(unnorm_log_post, axis=1)[:, None]

    #     return log_post

    def get_posterior(self, X):
        """
        Exponentiate log posterior. Returns component probabilities.
        """
        return np.exp(self.get_log_posterior(X))

    def pred_attr_probas(self, X, skill_idxs:list=None, return_post:bool=False):
        """
        X: array of shape (n_datapoints, n_items) representing emprical item
            response data.
        Q: array of shape (n_items, n_attributes) representing binary relationship
            between items and attributes.
        mask: array of shape (n_datapoints, n_items) is 1 whereever X is not null 
            and 0 otherwise.

        Returns:
            array of shape (n_datapoints, n_attributes) representing estimated
            attribute proficiency scores.
        """
        # get mixture weights \gamma(z_{im})
        post = self.get_posterior(X)
        profiles = post @ self.get_mu_slice(skill_idxs)

        if return_post:
            return profiles, post

        return profiles

    def get_effective_attr_dim(self):
        """Measure effective attribute space dimensionality"""
        M = self.mu  # shape: (n_components, n_attributes)

        # Center manually (important for affine span)
        M_centered = M - M.mean(axis=0, keepdims=True)

        # Compute covariance across components
        cov = np.cov(M_centered, rowvar=False)

        eigenvalues, _ = np.linalg.eigvalsh(cov)

        eff_dim = (eigenvalues).sum()**2 / (eigenvalues**2).sum()
        return eff_dim
    
    def pred_skill_probas_from_post(self, g, skill_idxs:list=None):
        """
        Predict skill (attribute) probabilities from pre-computed responsiblities (g).
        Responsiblities come from the `get_posterior(X)` call.

        g (nd-array): responsiblities (commonly represented as a lowercase gamma).
        """
        return g @ self.get_mu_slice(skill_idxs)

    def pred_skill_probas(self, X, return_post=False):
       return self.pred_attr_probas(X, skill_idxs=None, return_post=False)
    
    def pred_item_probas_numba(self, X, user_idxs, item_idxs):
        """
        Memory usage: O(#observations x components) NOT O(users x items)
        
        :param user_idxs: array of user indices
        :param item_idxs: array of item indices (same length)
        """
        import numba

        # get user-component responsibilities
        g = self.get_posterior(X)       # shape (n_users, n_components)

        log_mu = np.log(self.mu)        # shape (n_components, n_skills)
        log_1m_mu = np.log(1 - self.mu) 
        log_theta = np.log(self.theta).ravel()
        log_1m_theta = np.log(1 - self.theta).ravel()

        # Access internal arrays
        self.Q = self.Q.tocsr()
        Q_indptr = self.Q.indptr     # shape (n_rows + 1,)
        Q_indices = self.Q.indices   # shape (nnz,)  

        # Q is CSR: shape (n_items, n_skills)
        rows, cols = [], []
        for i in range(self.Q.indptr.shape[0]-1):
            for p in range(self.Q.indptr[i], self.Q.indptr[i+1]):
                rows.append(i)
                cols.append(self.Q.indices[p])
        Q_rows = np.array(rows, dtype=np.int64)
        Q_cols = np.array(cols, dtype=np.int64)  
        Q_indptr_start = self.Q.indptr 

        @numba.njit(parallel=True)
        def compute_probs(user_idxs, item_idxs):
            n = len(item_idxs)
            out = np.empty(n, dtype=np.float64)
            n_components = g.shape[1]

            for j in numba.prange(n):  # parallel over user-item pairs
                u = user_idxs[j]
                i = item_idxs[j]
                s1 = log_theta[i]
                s0 = log_1m_theta[i]

                # sum over all skills for this item
                start = Q_indptr_start[i]
                end   = Q_indptr_start[i+1]
                for idx in range(start, end):
                    k = Q_cols[idx]
                    temp1 = 0.0
                    temp0 = 0.0
                    for c in range(n_components):
                        temp1 += g[u, c] * log_mu[c, k]
                        temp0 += g[u, c] * log_1m_mu[c, k]
                    s1 += temp1
                    s0 += temp0

                out[j] = 1.0 / (1.0 + np.exp(s0 - s1))

            return out

        return compute_probs(user_idxs, item_idxs)
    
    def get_pred_nll(self, X):
        """
        Compute NLL for predictive cross entropy.
        """
        eps = 1e-15 
        p_j = np.clip(self.pred_item_probas(X), eps, 1 - eps)
        log_lik = X*np.log(p_j) + (1-X)*np.log(1-p_j)

        # Return average neg log lik per response
        return -log_lik.sum() / np.prod(X.shape)

    def pred_item_probas(self, X, profiles=None, item_idxs:list=None):
        """
        Predicts the probability of a correct response for selected items.

        Parameters
        ----------
        X : array-like of shape (n_datapoints, n_items)
            Empirical item response data for the individuals being evaluated.

        item_idx : list of int
            Indices of the items for which to compute predicted probabilities.
            If None, probabilities are returned for all items.

        profiles : array-like, optional
            Deprecated.

        Returns
        -------
        np.ndarray of shape (n_datapoints, n_selected_items)
            Predicted probabilities of a correct response for each individual–
            item pair.
        """
        # convert dense to sparse if needed
        if isinstance(X, np.ndarray):
            X = self.convert_dense_to_sparse(X)

        g = self.get_posterior(X)
        return self.pred_item_probas_from_resp(g, item_idxs)

    def pred_item_probas_from_resp(self, post, item_idxs=None):
        """
        Predict correct item response probabilities from proficiency scores (psi)
        """
        # Inference relies on "ab" parameters
        if (self.a is None) | (self.b is None):
            self.update_ab_params(use_norm=True)

        # return post @ self.mu @ self.Q.T / self.Q.sum(axis=1).T
        a = self.a[item_idxs, :] if item_idxs is not None else self.a
        b = self.b[item_idxs, :] if item_idxs is not None else self.b

        if item_idxs is None:
            log_p1 = a.T  # (N, J)
            log_p2 = b.T  # (N, J)
        else:
            log_p1 = self.a[item_idxs, :] 
            log_p2 = self.b[item_idxs, :]
        
        # Component-wise sigmoid of shape (M, J)
        s = 1 / (1 + np.exp(log_p2 - log_p1))

        # Return mixture of sigmoids
        return post @ s
    
    def reshape_to_row_vectors(self, x):
        if x.ndim == 1:
            return x.reshape(1, -1)
        return x

    def sample_user_skill_posterior(self, x, mask, n_samples):
        """
        params;
            x: array-like of shape (1, n_items)
                response vector for a user
            mask: array-like of shape (1, n_items)
                non-missing value mask for a user
            trace_matrix: array-like of shape (n_samples, n_skills)
                posterior samples of proficiencies drawn.
        """
        # init count trace matrix
        trace_matrix = np.zeros([n_samples, self.n_skills])

        # 1. for user i sample from \gamma(z_{im})
        post = np.exp(self.get_log_posterior(x))

        # init component indices
        components = np.arange(post.shape[1])

        # 2. for each attribute k sample from from Beta posterior
        for n in range(n_samples):
            # sample component index
            m = np.random.choice(components, p=post.ravel())
            trace_matrix[n] = [
                np.random.beta(a=self.mu_a[m, k], b=self.mu_b[m, k])
                for k in range(self.n_skills)
            ]

        return trace_matrix

    def get_skill_posterior(self, x, skill_idx):
        """
        Posterior is a weighted combination of Beta distributions
        params;
            x: array-like of shape (1, n_items)
                user-item response vector
            x_mask: array-like of shape (1, n_items)
                user-non-missng-item indicator vector
            skill_idx (int): skill index
        """
        k = skill_idx
        post = np.exp(self.get_log_posterior(x))

        x = np.linspace(0, 1, 100)
        pdf = np.array([beta.pdf(x, self.mu_a[m, k], self.mu_b[m, k]) for m in range(self.n_components)])
        # take weighted combination of beta PDFs
        return (post @ pdf).ravel()

    def get_item_difficulty(self, item_idxs=None):
        """
        1 - theta tends to be smaller for items with multiple problems. Therefore,
        we compute item difficulty as the (1 - theta)^(1 / T) where T gives the number 
        skills for each item.
        """
        item_diff = (1 - self.theta).ravel()
        if item_idxs is None:
            # compute non-transformed difficulties
            return item_diff
        else:
            return item_diff[item_idxs]

    def get_user_ability(self, X):
        """
        Compute user latent ability.
        """
        # (n_users, n_comps)
        g = self.get_posterior(X)
        logit_mu = logit(self.mu)

        return g @ logit_mu @ self.Q.T

    def convert_dense_to_sparse(self, X):
        """
        Safely convert binary (0 and 1) Numpy array into sparse SciPy CSR.
        """
        X1 = X.copy()
        X1[X1 == 0] = -1
        X1[np.isnan(X1)] = 0
        X1 = csr_matrix(X1)
        X1.data = (X1.data + 1) / 2
        return X1
    
    def get_prior_exp_skill_prof(self, skill_idxs:list=None):
        """
        Get the prior (before seeing data) expected
        skill proficiency skill indexes, `skill_idxs`.
        """
        mu = self.get_mu_slice(skill_idxs)
        return (self.pi.T @ mu).flatten()
    
    def sample_prior_skill_prof(self, skill_idxs:list=None):
        """
        Sample from the prior skill proficiency.
        """
        if skill_idxs is not None:
            mu = self.mu[:, skill_idxs]
        else:
            mu = self.mu
            skill_idxs = np.arange(self.mu.shape[1])

        pi = self.pi.flatten()
        M = len(pi)

        # Sample latent component
        m = np.random.choice(np.arange(M), p=pi)
        mu = mu[m, :]

        # Sample proficiency (psi) from latent component
        sample_psi = [
            np.random.beta(a=self.mu_a[m, k], b=self.mu_b[m, k])
            for k in skill_idxs
        ]

        return sample_psi
    
    def get_updated_post(self, item_idxs, x, old_unorm_post):
        """
        Incrementally update user posterior responbilites.
        WARNING: update it not idempotent. Sending the same
        request twice will update results twice.

        params:
            x (int): 0 or 1 (incorrect or correct, respectively)
            old_gamma_bar (array-like): old user unnormalized responsibilities.
        returns:
            new_gamma (array-like): updated user unnormalized responsibilities. 
        """
        a = self.a[item_idxs, :].reshape(1, -1)
        b = self.b[item_idxs, :].reshape(1, -1)

        # Updated responsbilities
        new_unorm_post =a**x * b**(1-x) * old_unorm_post
        # 
        # Updated proficies
        psi = (new_unorm_post @ self.mu).flatten()

        return psi, new_post


