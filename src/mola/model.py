
import numpy as np
from scipy.special import logsumexp, logit
from scipy.stats import beta
from scipy.sparse import SparseEfficiencyWarning
import warnings
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
from src.mola.train import _MoLABase


class MoLA(_MoLABase):
    """
    MoLA with item difficulty (Lentini 2025).

    Theta is related to difficulty by the following relationship:
        theta = (1 - d_j)^{n_j}

    Fitting and hyper-parameters live on :class:`_MoLABase`; this class adds the
    posterior, predictive and psi/difficulty read-outs.
    """

    def get_item_cov(self, skill_idxs=None):
        """
        Compute item covaraince matrix.
        """
        M, J= self.n_components, self.Q.shape[0]       

        # item-response probabilities for each distinct profile
        post = np.diag(np.ones(M))
        sigma = self._item_proba_from_resp(post)

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

    def get_proficiency_cov(self, X, skill_idxs:list=None):
        g = self.predict_proba(X)

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

    def predict_proba(self, X):
        """
        Archetype responsibilities for each learner.

        Parameters
        ----------
        X : (n_learners, n_items) dense ndarray (NaN = missing) or SciPy sparse.

        Returns
        -------
        (n_learners, n_components) ndarray whose rows sum to 1.
        """
        return np.exp(self.get_log_posterior(self._prepare_X(X)))

    def get_mu_slice(self, skill_idxs:list=None):
        """
        Return the archetype-profile matrix ``mu`` restricted to ``skill_idxs``.

        ``mu`` has shape ``(n_components, n_skills)``; this selects the columns
        named by ``skill_idxs`` (a list/array of skill indices). ``None`` returns
        every skill, i.e. ``mu`` unchanged.
        """
        if skill_idxs is None:
            return self.mu
        return self.mu[:, skill_idxs]

    def predict_skill_proba(self, X, skill_idxs:list=None):
        """
        Expected skill mastery for each learner: ``predict_proba(X) @ mu``.

        Parameters
        ----------
        X : (n_learners, n_items) dense ndarray (NaN = missing) or SciPy sparse.
        skill_idxs : list of int, optional
            Restrict the output to these skill columns.

        Returns
        -------
        (n_learners, n_skills) ndarray in ``[0, 1]``.
        """
        return self.predict_proba(X) @ self.get_mu_slice(skill_idxs)

    def effective_n_archetypes(self):
        """
        Effective number of distinct archetype profiles: the participation ratio
        (``(sum lambda)**2 / sum lambda**2``) of the centered ``mu`` matrix.

        The value lies in ``[1, min(n_components - 1, n_skills)]``: it is ~1 when
        every archetype profile lines up along a single direction (highly
        redundant archetypes) and rises toward its ceiling when the profiles vary
        along independent directions.
        """
        n_components = self.mu.shape[0]
        if n_components < 2:
            return float(n_components)

        # np.cov re-centers internally; centering here as well is harmless.
        mu_centered = self.mu - self.mu.mean(axis=0, keepdims=True)
        cov = np.atleast_2d(np.cov(mu_centered, rowvar=False))

        # eigvalsh returns just the (ascending) eigenvalues; clip the tiny
        # negative values it produces for near-singular covariances.
        eigenvalues = np.clip(np.linalg.eigvalsh(cov), 0.0, None)

        total = eigenvalues.sum()
        if total <= 1e-12:
            # all profiles identical -> a single effective dimension
            return 1.0
        return float(total**2 / np.square(eigenvalues).sum())

    def redundancy_report(self, X=None):
        """
        Diagnose whether any archetypes have become redundant.

        Parameters
        ----------
        X : (n_learners, n_items) dense ndarray or SciPy sparse, optional
            Learner responses.  When supplied, the report also measures how much
            posterior mass each archetype actually carries.

        Returns
        -------
        dict
            ``effective_n_archetypes``
                Participation ratio of ``mu`` (see :meth:`effective_n_archetypes`)
                -- profile-space spread.
            ``effective_n_components``
                Inverse Simpson index ``1 / sum(pi**2)`` of the mixture weights:
                the number of archetypes carrying non-trivial prior mass.
            ``weight_min`` / ``weight_argmin``
                The smallest mixture weight and the archetype it belongs to.
            ``closest_pair``
                ``(i, j)`` of the two archetypes with the most similar profiles
                (``None`` when there is only one archetype).
            ``closest_pair_distance``
                Euclidean distance between that pair's ``mu`` rows; a small value
                means a duplicated archetype.
            ``posterior_usage``
                Present only when ``X`` is given: mean posterior responsibility
                per archetype, shape ``(n_components,)``.  Near-zero entries are
                archetypes no learner loads onto.
        """
        pi = self.pi.ravel()
        n_components = pi.shape[0]

        report = {
            "effective_n_archetypes": self.effective_n_archetypes(),
            "effective_n_components": float(1.0 / np.square(pi).sum()),
            "weight_min": float(pi.min()),
            "weight_argmin": int(pi.argmin()),
        }

        if n_components >= 2:
            diff = self.mu[:, None, :] - self.mu[None, :, :]
            dist = np.sqrt(np.square(diff).sum(axis=-1))
            np.fill_diagonal(dist, np.inf)
            i, j = np.unravel_index(np.argmin(dist), dist.shape)
            report["closest_pair"] = (int(min(i, j)), int(max(i, j)))
            report["closest_pair_distance"] = float(dist[i, j])
        else:
            report["closest_pair"] = None
            report["closest_pair_distance"] = float("nan")

        if X is not None:
            report["posterior_usage"] = self.predict_proba(X).mean(axis=0)

        return report

    def skill_proba_from_posterior(self, g, skill_idxs:list=None):
        """
        Expected skill mastery from pre-computed responsibilities ``g`` (as
        returned by :meth:`predict_proba`): ``g @ mu``.
        """
        return g @ self.get_mu_slice(skill_idxs)

    def score(self, X):
        """
        Mean predictive negative log-likelihood per observed response.

        ``X`` is a dense 0/1 array (NaN entries are ignored) or SciPy sparse.
        """
        X = np.asarray(X.todense() if not isinstance(X, np.ndarray) else X, dtype=float)
        eps = 1e-15
        p = np.clip(self.predict_item_proba(X), eps, 1 - eps)
        mask = ~np.isnan(X)
        ll = X[mask] * np.log(p[mask]) + (1 - X[mask]) * np.log(1 - p[mask])
        return float(-ll.sum() / mask.sum())

    def predict_item_proba(self, X, item_idxs:list=None):
        """
        Predicted probability of a correct response.

        Parameters
        ----------
        X : (n_learners, n_items) dense ndarray (NaN = missing) or SciPy sparse.
        item_idxs : list of int, optional
            Restrict the output to these item columns.

        Returns
        -------
        (n_learners, n_selected_items) ndarray in ``[0, 1]``.
        """
        g = self.predict_proba(X)
        return self._item_proba_from_resp(g, item_idxs)

    def _item_proba_from_resp(self, post, item_idxs=None):
        """
        Predict correct item response probabilities from proficiency scores (psi)
        """
        # Inference relies on "ab" parameters
        if (self.a is None) | (self.b is None):
            self.update_ab_params(use_norm=True)

        # a, b are the log-scale "ab" artifacts of shape (J, M); restrict to the
        # requested items up front so both branches below stay identical.
        a = self.a[item_idxs, :] if item_idxs is not None else self.a
        b = self.b[item_idxs, :] if item_idxs is not None else self.b

        # Component-wise sigmoid of shape (M, J_selected)
        s = 1 / (1 + np.exp(b.T - a.T))

        # Return mixture of sigmoids, shape (N, J_selected)
        return post @ s

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
        post = np.exp(self.get_log_posterior(self._prepare_X(x)))

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
        post = np.exp(self.get_log_posterior(self._prepare_X(x)))

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
        Compute user latent ability, shape (n_learners, n_items).
        """
        g = self.predict_proba(X)
        logit_mu = logit(self.mu)
        return g @ logit_mu @ self.Q.T

    def get_prior_exp_skill_prof(self, skill_idxs:list=None):
        """
        Prior (before seeing data) expected skill proficiency: ``pi @ mu``.
        """
        mu = self.get_mu_slice(skill_idxs)
        return (self.pi.T @ mu).flatten()

