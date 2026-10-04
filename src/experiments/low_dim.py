from src.utils.datasets import download_frac_subtr_to_pandas
from src.cdm.cdm_model import CDMModel
from src.mola import MoLA
from src.cdm.dina import DINA
from src.simulations.metrics import masked_nll
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

# Use LaTeX-style publication formatting
rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})


def get_y_proba_and_true(model, X_df):
    np.random.seed(254)
    # 1. Mask 10-20% of the entries randomly
    X_masked = X_df.astype(float, copy=True)
    mask = np.random.rand(*X_masked.shape) < 0.2
    X_masked[mask] = np.nan # Set some values to NA
    
    # 2. Get posteriors based ONLY on the non-masked items (must handle NAs internally).
    # MoLA exposes predict_item_proba; the CDM wrappers still use pred_item_probas.
    predict_item = getattr(model, "predict_item_proba", None) or model.pred_item_probas
    y_proba_all = predict_item(X_masked)
    
    # 3. Only pull the probabilities for the items that were hidden
    y_proba = y_proba_all[mask]
    if not isinstance(X_df, np.ndarray):
        X = X_df.to_numpy()
    else:
        X = X_df
    y_true = X[mask]

    y_true[y_true == -1] = 0
    return y_proba, y_true

def get_brier_score(y_true, y_prob):
    return float(((np.array(y_true) - np.array(y_prob))**2).sum() / len(y_true))

def run_experiment(model_name, splits, Q_df):

    auc_trains, auc_tests = [], []
    train_nlls, test_nlls = [], []
    brier_tests, brier_trains = [], []
    y_true_tests, y_probas_tests = [], []

    # Instantiate model
    if model_name == "GDINA":
        model = CDMModel("GDINA")
    elif model_name == "HO-DINA":
        model = CDMModel("HO-DINA")
    elif model_name == "DINA":
        model = DINA()

    for X_train_df, X_test_df in splits:
        model.fit(X_train_df, Q_df)

        # Predictions
        y_proba_train, y_true_train = get_y_proba_and_true(model, X_train_df)
        y_proba_test, y_true_test = get_y_proba_and_true(model, X_test_df)

        y_true_tests.append(y_true_test)
        y_probas_tests.append(y_proba_test)

        # NLL score, computed on the same masked held-out (y_proba, y_true)
        # pairs as AUC/Brier below -- NOT model.get_pred_nll(X), which
        # infers each learner's profile from their own full, unmasked
        # response vector and then scores against those same responses
        # (via R's IRT.posterior(resp.pattern=X, ...)), a self-referential
        # goodness-of-fit statistic rather than a held-out predictive one.
        train_nlls.append(masked_nll(y_true_train, y_proba_train))
        test_nlls.append(masked_nll(y_true_test, y_proba_test))

        # AUC scores
        auc_trains.append(roc_auc_score(y_true_train, y_proba_train))
        auc_tests.append(roc_auc_score(y_true_test, y_proba_test))

        # Brier scores
        brier_tests.append(get_brier_score(y_proba_test, y_true_test))
        brier_trains.append(get_brier_score(y_proba_train, y_true_train))

    return {
        "model": model_name,
        "nll_train": train_nlls,
        "nll_test": test_nlls,
        "auc_train": auc_trains,
        "auc_test": auc_tests,
        "brier_train": brier_trains,
        "brier_test": brier_tests,
        "y_probas_test": y_probas_tests,
        "y_true_test": y_true_tests
    }

def select_mola_m(X_train, Q_matrix, m_grid, m_select_frac=0.2, seed=0, **mola_kwargs):
    """Picks MoLA's number of archetypes `M` from `m_grid` by held-out
    predictive log-likelihood -- the same model-selection procedure
    `src.experiments.high_dim.select_mola_m` uses, adapted to this file's
    dense wide-array data (no missing entries, unlike EdNet) rather than
    long-format response tables.

    Splits `X_train`'s rows (learners) into `param_fit`
    (`1 - m_select_frac`) and `m_select` (`m_select_frac`) by a seeded
    permutation. For each candidate `M`: fits `MoLA(Q_matrix,
    n_components=M)` on `param_fit`, then scores held-out NLL on
    `m_select` via `get_y_proba_and_true` -- the same random-item-mask
    mechanism already used everywhere else in this file to simulate
    predictive scoring, so the mask is identical across candidate `M`s
    (a fair comparison) and consistent with how every other model here
    is scored. Returns `(best_m, nll_by_m)`.
    """
    rng = np.random.default_rng(seed)
    n_learners = X_train.shape[0]
    perm = rng.permutation(n_learners)
    n_select = int(round(m_select_frac * n_learners))
    select_idx, param_idx = perm[:n_select], perm[n_select:]
    X_param, X_select = X_train[param_idx], X_train[select_idx]

    nll_by_m = {}
    for m in m_grid:
        model_params = {
            'Q': Q_matrix,
            'n_components': m,
            'mu_prior': (2, 2),
            'theta_prior': (2, 2),
            'pi_prior': 2,
            'tol': 1e-5,
            'max_iter': 50,
            'pseudo_likelihood': False,
            'random_state': int(rng.integers(0, 2**31 - 1)),
        }
        model_params.update(mola_kwargs)
        model = MoLA(**model_params)
        model.fit(X_param)

        y_proba, y_true = get_y_proba_and_true(model, X_select)
        nll_by_m[m] = masked_nll(y_true, y_proba)

    best_m = min(nll_by_m, key=nll_by_m.get)
    return best_m, nll_by_m


def fit_mola(splits, Q_df, m_grid=None):
    """Fits MoLA on every split. `m_grid`, if given, selects `M` per
    split via `select_mola_m` (held-out predictive log-likelihood on a
    slice of that split's training learners, mirroring the high-dim
    experiment's protocol) before refitting on the *full* training set
    with the selected `M` -- `m_select` only ever informs which `M` to
    use, never contributes to the final fitted parameters. `m_grid=None`
    (the default) keeps the old fixed `n_components=6` behavior.
    """
    auc_trains, auc_tests = [], []
    nll_trains, nll_tests = [], []
    brier_trains, brier_tests = [], []
    y_true_tests, y_probas_tests = [], []
    selected_ms = []
    final_results = []
    for split_i, (X_train_df, X_test_df) in enumerate(splits):

        X_train = X_train_df.to_numpy()
        X_test = X_test_df.to_numpy()
        Q_matrix = Q_df.to_numpy()

        X_train[X_train == -1] = 0
        X_test[X_test == -1] = 0

        if m_grid is not None:
            selected_m, _ = select_mola_m(X_train, Q_matrix, m_grid, seed=split_i)
        else:
            selected_m = 6
        selected_ms.append(selected_m)

        # Configure model parameters
        model_params = {
            'Q': Q_matrix,
            'n_components': selected_m,
            'mu_prior': (2, 2),
            'theta_prior': (2, 2),
            'pi_prior': 2,
            'tol': 1e-5,
            'max_iter': 50,
            'pseudo_likelihood': False,
        }

        # Instantiate the model
        model = MoLA(**model_params)

        # Fit the model to the data
        model.fit(X_train)

        # Prediction
        y_proba_train, y_true_train = get_y_proba_and_true(model, X_train)
        y_proba_test, y_true_test = get_y_proba_and_true(model, X_test)

        y_true_tests.append(y_true_test)
        y_probas_tests.append(y_proba_test)

        # AUC scores
        auc_trains.append(roc_auc_score(y_true_train, y_proba_train))
        auc_tests.append(roc_auc_score(y_true_test, y_proba_test))

        # Brier scores
        brier_tests.append(get_brier_score(y_proba_test, y_true_test))
        brier_trains.append(get_brier_score(y_proba_train, y_true_train))

        # NLL score, computed on the same masked held-out (y_proba, y_true)
        # pairs as AUC/Brier above -- NOT model.score(X), which infers each
        # learner's archetype posterior from their own full, unmasked
        # response vector and then scores against those same responses, a
        # self-referential goodness-of-fit statistic rather than a
        # held-out predictive one (see run_experiment's matching note).
        nll_trains.append(masked_nll(y_true_train, y_proba_train))
        nll_tests.append(masked_nll(y_true_test, y_proba_test))

    result = {
        "model": "MoLA",
        "nll_train": nll_trains,
        "nll_test": nll_tests,
        "auc_train": auc_trains,
        "auc_test": auc_tests,
        "brier_train": brier_trains,
        "brier_test": brier_tests,
        "y_probas_test": y_probas_tests,
        "y_true_test": y_true_tests,
        "selected_m": selected_ms,
    }

    final_results.append(result)

    return final_results

def plot_calibrations(final_results, model_names, n_bins=10, save=None):
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # Create 2x2 layout
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)
    axes = axes.flatten()

    for i, model_name in enumerate(model_names):

        ax = axes[i]

        results = [r for r in final_results if r['model'] == model_name][0]

        avg_y_probas = np.array(results['y_probas_test']).mean(axis=0)
        avg_y_true = np.array(results['y_true_test']).mean(axis=0)

        data = pd.DataFrame({
            'y_proba': avg_y_probas,
            'y_true': avg_y_true
        })

        data['bin'] = pd.qcut(
            data['y_proba'],
            q=n_bins,
            duplicates='drop'
        )

        calibration = data.groupby('bin', observed=False).agg(
            mean_pred=('y_proba', 'mean'),
            mean_true=('y_true', 'mean'),
            std_true=('y_true', 'std'),
            count=('y_true', 'size')
        ).reset_index()

        calibration['sem_true'] = (
            calibration['std_true'] /
            np.sqrt(calibration['count'])
        )

        # Plot calibration points with error bars
        ax.errorbar(
            calibration['mean_pred'],
            calibration['mean_true'],
            yerr=calibration['sem_true'],
            fmt='o',
            markersize=6,
            linewidth=2,
            capsize=4,
            color=colors[i],
            markerfacecolor=colors[i],
            markeredgecolor='black'
        )

        # Connect points
        ax.plot(
            calibration['mean_pred'],
            calibration['mean_true'],
            linewidth=2,
            color=colors[i]
        )

        # Perfect calibration reference line
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle='--',
            color='gray',
            linewidth=1.5
        )

        # Formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(model_name)
        # ax.set_ylabel('Observed Frequency')
        ax.grid(True, linestyle=':', alpha=0.6)

    fig.supylabel('Observed Frequency')
    fig.supxlabel('Predicted Probability')

    # Remove unused axes if fewer than 4 models
    for j in range(len(model_names), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # save figure
    if save is not None:
        fig.savefig(save)