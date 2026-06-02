import numpy as np
import itertools
from scipy.sparse import spmatrix
import pandas as pd
from scipy.special import logsumexp
import numpy as np
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri


class DINA:
    """
    Vanilla DINA Cognitive Diagnositc Model (CDM) 
    EM optimizer

    Source: De La Torre, Jimmy. "DINA model and parameter estimation: A didactic." 
    Journal of educational and behavioral statistics 34.1 (2009): 115-130.
    """
    def fit(self, X, Q):
        """
        Fit DINA using R package.
        
        :param self: Description
        """
        def sparse_to_nan_df(X):
            X = X.tocoo()
            dense = np.full(X.shape, np.nan)
            dense[X.row, X.col] = X.data
            return pd.DataFrame(dense)

        if not isinstance(X, pd.DataFrame):
            X = sparse_to_nan_df(X)

        # Drop rows with all missing values
        X = X[~X.isna().all(axis=1)]
        
        if not isinstance(Q, pd.DataFrame):
            Q = pd.DataFrame(Q.toarray())

        with localconverter(ro.default_converter + pandas2ri.converter):
            ro.globalenv['X_train'] = ro.conversion.py2rpy(X)
            ro.globalenv['Q_matrix'] = ro.conversion.py2rpy(Q)

        ro.r("""
            # Fit the CDM model
            train_time <- system.time({{
                est_cdm <- CDM::din(
                    X_train, 
                    Q_matrix
                    # maxit = 20
                )
            }})['elapsed']

            # Extract arameters
            item_par <- coef(est_cdm)

            # Extract guess and slip using names
            guess <- item_par[grep("_guess$", names(item_par))]
            slip  <- item_par[grep("_slip$",  names(item_par))]

            # Remove names and ensure correct order
            guess <- as.numeric(guess)
            slip  <- as.numeric(slip)

            # Do parameter clipping
            guess <- pmin(pmax(guess, 0.01), 0.99)
            slip  <- pmin(pmax(slip,  0.01), 0.99)
        """)

    def enumerate_profiles(self, K):
        return np.array(list(itertools.product([0, 1], repeat=K)))  # (2^K, K)

    def log_likelihood(self, X, eta, guess, slip):
        """
        X: (N, J)
        eta: (J, L)
        guess, slip: (J,)
        returns: (N, L)
        """
        X1 = X
        if isinstance(X1, spmatrix):
            X2 = X1.copy()
            X2.data = 1 - X1.data
        
        else:
            # Dense implementation
            X2 = 1 - X1
            X1 = np.where(X == 1, 1, 0)
            X2 = np.where(X == 0, 1, 0)

        # Depending on the profile and item, p_correct is either guessing or slipping
        # You are guessing if you are not in a profile which contains mastery of all 
        # skills need for the item
        p_correct = eta * (1 - slip[:, None]) + (1 - eta) * guess[:, None]  # (J, L)
        log_like = X1 @ np.log(p_correct) + X2 @ np.log(1 - p_correct)

        return log_like  # (N, L)

    def posterior_predictive(self, skill_profile_probas, eta, guess, slip):
        """
        returns: (N, J)
        """
        p_correct = eta * (1 - slip[:, None]) + (1 - eta) * guess[:, None]  # (J, L)
        return skill_profile_probas @ p_correct.T  # (N, J)

    def pred_item_probas(self, X):
        with localconverter(ro.default_converter + pandas2ri.converter):
            guess = np.asarray(ro.r('guess'))  # (J,)
            slip  = np.asarray(ro.r('slip'))   # (J,)
            Q     = np.asarray(ro.r('Q_matrix'))      # (J, K)

        assert (guess != 1.).all(), "Guess contains 1s"
        assert (slip != 0.).all(), "Slip contains 0s"

        K = Q.shape[1]
        A = self.enumerate_profiles(K)              # (L, K)
        eta = (A[None, :, :] >= Q[:, None, :]).all(axis=2).astype(int)  # (J, L)

        log_like = self.log_likelihood(X, eta, guess, slip)
        log_post = log_like - logsumexp(log_like, axis=1, keepdims=True)
        skill_profile_probas = np.exp(log_post)  # (N, L)

        item_probas = self.posterior_predictive(skill_profile_probas, eta, guess, slip)
        return item_probas
    
    def get_marginal_nll(self, X, prior=None):
        """
        X: (N, J) binary responses
        eta: (J, L) success matrix per profile
        guess, slip: (J,)
        prior: (L,) optional profile prior. Defaults to uniform.
        Returns total log-likelihood across N examinees.
        """
        if isinstance(X, spmatrix):
            print('foo')
            X.data[X.data == 0] = -1
            X = X.toarray().astype(float)
            X[X == 0] = np.nan
            X[X == -1] = 0

        with localconverter(ro.default_converter + pandas2ri.converter):
            guess = np.asarray(ro.r('guess'))  # (J,)
            slip  = np.asarray(ro.r('slip'))   # (J,)
            Q     = np.asarray(ro.r('Q_matrix'))      # (J, K)

        K = Q.shape[1]
        A = self.enumerate_profiles(K)              # (L, K)
        eta = (A[None, :, :] >= Q[:, None, :]).all(axis=2).astype(int)  # (J, L)

        L = eta.shape[1]

        if prior is None:
            prior = np.ones(L) / L  # uniform prior

        # 1. Convert responses to 0/1
        X1 = np.where(X == 1, 1, 0)
        X2 = 1 - X1

        # 2. Compute log p(x_i | alpha)
        p_correct = eta * (1 - slip[:, None]) + (1 - eta) * guess[:, None]  # (J, L)
        log_like = X1 @ np.log(p_correct) + X2 @ np.log(1 - p_correct)        # (N, L)

        # 3. Add log prior
        log_weighted = log_like + np.log(prior + 1e-12)  # (N, L)

        # 4. Marginalize with logsumexp
        log_marginal = logsumexp(log_weighted, axis=1)   # (N,)

        if isinstance(X, np.ndarray) | isinstance(X, pd.DataFrame):
            n_obs = (~np.isnan(np.array(X))).sum()
        else:
            n_obs = len(X.data)

        total_nll = -log_marginal.sum()
        mean_nll =  total_nll / n_obs # total log-likelihood
        return total_nll, mean_nll, n_obs
    
    def get_pred_nll(self, X):
        """
        Compute average NLL from predictive cross-entropy.
        """
        p_j = self.pred_item_probas(X)
        X = np.array(X)
        log_lik = X*np.log(p_j) + (1-X)*np.log(1-p_j)

        # Return average neg log lik per response
        return -log_lik.sum() / np.prod(X.shape)
