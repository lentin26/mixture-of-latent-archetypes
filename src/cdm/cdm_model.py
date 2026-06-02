import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from rpy2.robjects import r
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.sparse import issparse


class CDMModel:
    """
    Python wrapper for CDM / GDINA models.
    All inference (in- and out-of-sample) is done in R.
    """

    def __init__(self, model_type="DINA"):
        """
        model_type: "DINA", "GDINA", "HO-DINA"
        """
        assert model_type in {"DINA", "GDINA", "HO-DINA"}
        self.model_type = model_type
        self._fitted = False

    # -------------------------
    # Fit model in R
    # -------------------------
    def fit(self, X, Q):
        if not isinstance(X, pd.DataFrame):
            # X = pd.DataFrame(X.toarray() if isinstance(X, spmatrix) else X)
            X.data[X.data == 0] = -1
            X = pd.DataFrame(X.toarray().astype(float))
            X[X == 0] = np.nan
            X[X == -1] = 0
        if not isinstance(Q, pd.DataFrame):
            Q = pd.DataFrame(Q.toarray() if isinstance(Q, spmatrix) else Q)

        # Drop rows with all missing values
        X = X[~X.isna().all(axis=1)]

        with localconverter(ro.default_converter + pandas2ri.converter):
            ro.globalenv["X_train"] = ro.conversion.py2rpy(X)
            ro.globalenv["Q_matrix"] = ro.conversion.py2rpy(Q)

        if self.model_type == "DINA":
            ro.r("""
                library(CDM)
                 
                 
                # Create all possible profiles manually
                K <- ncol(Q_matrix)
                profiles <- as.matrix(expand.grid(rep(list(c(0, 1)), K)))

                # Fit the model using the explicit profiles
                est_model <- CDM::gdina(
                    dat = X_train,
                    q.matrix = Q_matrix,
                    rule = "DINA",
                    maxit     = 50,
                    skillclass = profiles  # Passing the matrix directly often avoids the HLM bug
                )
                """)

        elif self.model_type == "GDINA":
            ro.r("""
                library(CDM)
                 
                X_df <- as.data.frame(X_train)
                X_agg <- X_df |>
                    dplyr::group_by(dplyr::across(everything())) |>
                    dplyr::summarise(freq = dplyr::n(), .groups = "drop")
                                
                est_model <- CDM::gdina(
                    dat = as.matrix(X_agg[, -ncol(X_agg)]),
                    q.matrix = Q_matrix,
                    maxit     = 20,     # default is often overkill
                    weights   = X_agg$freq,
                )
            """)

        elif self.model_type == "HO-DINA":
            ro.r("""
                # Create all possible profiles manually
                K <- ncol(Q_matrix)
                profiles <- as.matrix(expand.grid(rep(list(c(0, 1)), K)))

                # Fit the model using the explicit profiles
                est_model <- CDM::gdina(
                    dat = X_train,
                    q.matrix = Q_matrix,
                    rule = "DINA",
                    maxit     = 50,
                    skillclass = profiles,  # Passing the matrix directly often avoids the HLM bug
                    att.dist = "higher.order" # Structural logic (HO)
                )
                """)

        self._fitted = True

    # -------------------------
    # Out-of-sample item prediction
    # -------------------------
    def pred_item_probas(self, X):
        """
        Posterior predictive item probabilities.
        Returns: (N, J) numpy array
        """
        assert self._fitted, "Call fit() first."

        if isinstance(X, spmatrix):
            # X.data[X.data == 0] = -1
            # X = pd.DataFrame(X.toarray())
            # X[X == 0] = np.nan
            # X[X == -1] = 0

            X_dense = X.toarray().astype(float)

            # Nonzero entries = observed (either 1 or stored 0 if truly stored)
            observed_mask = X.copy()
            observed_mask.data = np.ones_like(observed_mask.data)
            observed_mask = observed_mask.toarray().astype(bool)

            # Missing = not observed
            X_dense[~observed_mask] = np.nan

            X = pd.DataFrame(X_dense)

        # if not isinstance(X, pd.DataFrame):
        #     X = pd.DataFrame(X.toarray() if isinstance(X, spmatrix) else X)

        with localconverter(ro.default_converter + pandas2ri.converter):
            ro.globalenv["X_test"] = ro.conversion.py2rpy(X)

        if self.model_type == "DINA":
            ro.r("""
                library(CDM)

                pred <- IRT.posterior(
                    object = est_model,
                    resp.pattern = X_new,
                    what = "prob"
                )
            """)

        else:  # GDINA and HO-DINA
            ro.r("""
            # 1. Get IRFs [n_items x 2^n_skills]
            irfs_success <- est_model$pjk[,2 , ]

            # 2. get 2^n_skills prior probability for each profile
            prior <- est_model$attribute.patt$class.prob

            # Adding a tiny constant (1e-10) prevents log(0) errors
            log_P <- log(irfs_success + 1e-10)
            log_Q <- log(1 - irfs_success + 1e-10)
                
            # Ensure X_train is a numeric matrix (N x J)
            X_test_mat <- as.matrix(X_test)

            # # 2. Compute the Log-Likelihood matrix (N x L)
            # # [N x J] %*% [J x L] sums (X * logP)
            # # [(N x J)] %*% [J x L] sums ((1-X) * logQ)
            # log_lik_matrix <- (X_test_mat %*% log_P) + ((1 - X_test_mat) %*% log_Q)
                 
            # 1. Create a version of the data where NaNs are 0
            X_calc <- X_test_mat
            X_calc[is.na(X_calc)] <- 0

            # 2. Create an indicator matrix (1 if observed, 0 if masked/NaN)
            observed <- matrix(1, nrow=nrow(X_test_mat), ncol=ncol(X_test_mat))
            observed[is.na(X_test_mat)] <- 0

            # 3. Calculate log-likelihood
            # We only sum log_P when X=1 AND observed=1
            # We only sum log_Q when X=0 AND observed=1
            log_lik_matrix <- (X_calc %*% log_P) + ((observed - X_calc) %*% log_Q)
                
            # 3. Incorporate the Prior (L-dimensional vector)
            # log(Posterior) proportional to log(Likelihood) + log(Prior)
            log_prior <- log(prior + 1e-10)
            log_posterior_unnorm <- sweep(log_lik_matrix, 2, log_prior, FUN = "+")

            # 4. Normalize to get actual Posterior Probabilities (N x L)
            # Use the log-sum-exp trick for numerical stability
            post_probs <- t(apply(log_posterior_unnorm, 1, function(x) {
            x_shifted <- x - max(x)
            exp(x_shifted) / sum(exp(x_shifted))
            }))
                
            # [N x L] %*% [L x J]
            item_predictions <- post_probs %*% t(irfs_success)
            """)

        with localconverter(ro.default_converter + pandas2ri.converter):
            y_proba = ro.conversion.rpy2py(ro.r("item_predictions"))

        return y_proba
    
    def get_pred_nll(self, X):
        """
        Compute average NLL from predictive cross-entropy.
        """
        p_j = self.pred_item_probas(X)

        # X.data[X.data == 0] = -1
        if issparse(X):
            rows, cols = X.nonzero()
            assert (p_j[rows, cols] > 0).all() & (p_j[rows, cols] < 1).all()
            X1 = X.copy()
            X1.data[X1.data == -1] = 0
            X2 = X1.copy()
            X2.data = (1 - X2.data)
            X1 = X1.toarray()
            X2 = X2.toarray()

        else:
            X1 = np.array(X)
            X2 = 1 - X1
            
        log_lik = X1*np.log(p_j) + X2*np.log(1-p_j)

        # Return average neg log lik per response
        nll = -log_lik.sum() / np.prod(X.shape)
        assert not np.isnan(nll)

        return nll
    
    def get_nll(self, X):
        if isinstance(X, spmatrix):
            X.data[X.data == 0] = -1
            X.data[X.data]
            X = pd.DataFrame(X.toarray())
            X[X == 0] = np.nan
            X[X == -1] = 0

        with localconverter(ro.default_converter + pandas2ri.converter):
            ro.globalenv["X_test"] = ro.conversion.py2rpy(X)

        ro.r("""
            # 1. Get IRFs [n_items x 2^n_skills]
            irfs_success <- est_model$pjk[,2 , ]

            # 2. get 2^n_skills prior probability for each profile
            prior <- est_model$attribute.patt$class.prob

            # Adding a tiny constant (1e-10) prevents log(0) errors
            log_P <- log(irfs_success + 1e-10)
            log_Q <- log(1 - irfs_success + 1e-10)
                
            # Ensure X_train is a numeric matrix (N x J)
            X_test_mat <- as.matrix(X_test)

            # # 2. Compute the Log-Likelihood matrix (N x L)
            # # [N x J] %*% [J x L] sums (X * logP)
            # # [(N x J)] %*% [J x L] sums ((1-X) * logQ)
            # log_lik_matrix <- (X_test_mat %*% log_P) + ((1 - X_test_mat) %*% log_Q)
                 
            # 1. Create a version of the data where NaNs are 0
            X_calc <- X_test_mat
            X_calc[is.na(X_calc)] <- 0

            # 2. Create an indicator matrix (1 if observed, 0 if masked/NaN)
            observed <- matrix(1, nrow=nrow(X_test_mat), ncol=ncol(X_test_mat))
            observed[is.na(X_test_mat)] <- 0

            # 3. Calculate log-likelihood
            # We only sum log_P when X=1 AND observed=1
            # We only sum log_Q when X=0 AND observed=1
            log_lik_matrix <- (X_calc %*% log_P) + ((observed - X_calc) %*% log_Q)
                
            # 3. Incorporate the Prior (L-dimensional vector)
            # log(Posterior) proportional to log(Likelihood) + log(Prior)
            log_prior <- log(prior + 1e-10)
            log_posterior_unnorm <- sweep(log_lik_matrix, 2, log_prior, FUN = "+")
                 
            # Compute marginal log-likelihood for each examinee
                log_marginal <- apply(log_posterior_unnorm, 1, function(x) {
                m <- max(x)
                m + log(sum(exp(x - m)))
            })
             
            n_obs <- sum(observed)
            total_loglik <- sum(log_marginal)
            mean_loglik <- total_loglik / n_obs
            """)
        
        with localconverter(ro.default_converter + pandas2ri.converter):
            mean_nll = -ro.conversion.rpy2py(ro.r("mean_loglik"))
            total_nll = -ro.conversion.rpy2py(ro.r("mean_loglik"))
            n_obs = ro.conversion.rpy2py(ro.r("n_obs"))

        return total_nll, mean_nll, n_obs

    def get_gdina_posteriors(mod_r):
        """
        Given an R GDINA model object, returns:
        - post: N x L posterior probabilities for latent classes
        - X_pred: N x J predicted probabilities for each item
        """
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2.robjects.conversion import localconverter

        import numpy as np

        # Use a conversion context
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            # ---------- 1. Get number of skills ----------
            print('Hi')
            K = 8 #ro.r("est_model$info$K")[0]  # number of skills
            print(K)
            L = 2 ** K                       # number of latent classes

            # ---------- 2. Get item probabilities per latent class ----------
            probs_list = ro.r("coef(est_model)")  
            # convert to Python list of numpy arrays
            probs_list_py = [np.array(x) for x in probs_list]
            print(probs_list_py[0])
            print(probs_list_py[1])

            # stack into L x J matrix
            # coef_mat = np.column_stack(probs_list_py)  # shape: (L, J)

            # ---------- 3. Get posterior over latent classes ----------
            post = ro.r('personparm(est_model, what="EAP")')  # N x L
            post = np.array(post)
            print(post.shape)

            # ---------- 4. Compute predicted item probabilities ----------
            X_pred = post @ coef_mat  # N x J






# import numpy as np
# import pandas as pd
# from scipy.sparse import spmatrix
# from scipy.special import logsumexp
# from rpy2.robjects.conversion import localconverter
# import rpy2.robjects as ro
# from rpy2.robjects import pandas2ri


# class HODINA:
#     """
#     Higher-Order DINA (HO-DINA).
#     Structural model is used ONLY for estimation.
#     Prediction uses marginalization over α.
#     """

#     def __init__(self, eps=1e-6):
#         self.pi_lj = None   # (L, J)
#         self.L = None
#         self.J = None
#         self.eps = eps

#     # --------------------------------------------------
#     # Fit model in R and extract π_{ℓj}
#     # --------------------------------------------------
#     def fit(self, X, Q):
#         if not isinstance(X, pd.DataFrame):
#             X = pd.DataFrame(X.toarray() if isinstance(X, spmatrix) else X)

#         if not isinstance(Q, pd.DataFrame):
#             Q = pd.DataFrame(Q.toarray())

#         self.J = X.shape[1]

#         with localconverter(ro.default_converter + pandas2ri.converter):
#             ro.globalenv["X_train"] = ro.conversion.py2rpy(X)
#             ro.globalenv["Q_matrix"] = ro.conversion.py2rpy(Q)

#         ro.r("""
#             library(CDM)

#             est_hodina <- gdina(
#                 dat = X_train,
#                 q.matrix = Q_matrix,
#                 rule = "DINA",
#                 # HOGDINA = 1,
#                 progress = FALSE
#             )

#             # probitem is long-format; reshape to (L × J)
#             df <- est_hodina$probitem

#             # Inspect column names (for debugging once)
#             print(colnames(df))

#             # Long-format probabilities -> L x J matrix
#             pi_mat <- xtabs(
#                 prob ~ skillcomb + itemno,
#                 data = est_hodina$probitem
#             )
#             pi_mat <- as.matrix(pi_mat)
#         """)

#         with localconverter(ro.default_converter + pandas2ri.converter):
#             self.pi_lj = np.asarray(ro.r("pi_mat"), dtype=float)

#         self.pi_lj = np.clip(self.pi_lj, self.eps, 1 - self.eps)
#         self.L = self.pi_lj.shape[0]

#         return self

#     # --------------------------------------------------
#     # Posterior over latent profiles
#     # --------------------------------------------------
#     def posterior(self, X):
#         X = X.toarray() if isinstance(X, spmatrix) else np.asarray(X)

#         X1 = X[:, None, :]
#         X0 = 1 - X1

#         log_pi = np.log(self.pi_lj)[None, :, :]
#         log_1m = np.log(1 - self.pi_lj)[None, :, :]

#         log_like = (X1 * log_pi + X0 * log_1m).sum(axis=2)

#         # uniform prior (structural model already used in estimation)
#         log_post = log_like - np.log(self.L)
#         log_post -= logsumexp(log_post, axis=1, keepdims=True)

#         return np.exp(log_post)

#     # --------------------------------------------------
#     # Posterior predictive item probabilities
#     # --------------------------------------------------
#     def pred_item_probas(self, X):
#         post = self.posterior(X)       # (N, L)
#         return post @ self.pi_lj       # (N, J)


# import numpy as np
# from scipy.sparse import spmatrix
# import pandas as pd
# from scipy.special import logsumexp
# import numpy as np
# from rpy2.robjects.conversion import localconverter
# import rpy2.robjects as ro
# from rpy2.robjects import pandas2ri


# class HODINA:
#     """
#     Higher-Order DINA Model.
#     Traits govern the probability of skill mastery.
#     """
#     def fit(self, X, Q):
#         if not isinstance(X, pd.DataFrame):
#             X = pd.DataFrame(X.toarray()) if isinstance(X, spmatrix) else pd.DataFrame(X)
#         if not isinstance(Q, pd.DataFrame):
#             Q = pd.DataFrame(Q.toarray())

#         with localconverter(ro.default_converter + pandas2ri.converter):
#             ro.globalenv['X_train'] = ro.conversion.py2rpy(X)
#             ro.globalenv['Q_matrix'] = ro.conversion.py2rpy(Q)

#         ro.r("""
#             library(CDM)
#             # rule="DINA" with HOGDINA structural model
#             est_hodina <- CDM::gdina(X_train, Q_matrix, rule="DINA", 
#                                     HOGDINA=1, progress=FALSE)
            
#             # item_probs_attr is (L, J)
#             item_probs_attr <- est_hodina$probitem
#         """)

#     def pred_item_probas(self, X):
#         # The out-of-sample prediction logic for HO-DINA is identical to GDINA
#         # once the constrained item_probs_attr is extracted.
#         with localconverter(ro.default_converter + pandas2ri.converter):
#             pi_lj = np.asarray(ro.r('item_probs_attr')).T 
            
#         X1 = X.toarray() if isinstance(X, spmatrix) else np.array(X)
#         X0 = 1 - X1
#         pi_lj = np.clip(pi_lj, 1e-6, 1 - 1e-6)
        
#         log_like = X1 @ np.log(pi_lj) + X0 @ np.log(1 - pi_lj)
#         log_post = log_like - logsumexp(log_like, axis=1, keepdims=True)
#         post_probas = np.exp(log_post)
        
#         return post_probas @ pi_lj.T