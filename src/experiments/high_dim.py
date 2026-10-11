"""EdNet MoLA-vs-NeuralCDM evaluation pipeline (see
notebooks/high-dim-experiment.ipynb for the full protocol writeup).

Learners are split 70/30 into a parameter-estimation set and an
evaluation set; the parameter-estimation set is further split 80/20 into
`param_fit` (used to fit each candidate MoLA model) and `m_select` (used
only to pick MoLA's number of archetypes `M` by held-out predictive
log-likelihood) -- so, of all eligible learners: 56% param_fit, 14%
m_select, 30% evaluation. Once `M` is selected, both models' FINAL
population parameters are fit on the full 70% (`param_fit` + `m_select`
recombined) -- selecting `M` on a held-out slice and then training the
chosen model on all the data that slice came from is standard practice
(the same pattern as k-fold cross-validation followed by a refit on the
full training set): `m_select` only ever informs which `M` to use, it
never contributes to any model's fitted parameter values until after
that choice is fixed, so there is no leakage into the parameters
themselves. For each evaluation learner, responses are split
chronologically 60/40 into a profiling period (used to infer that
learner's proficiency, with population parameters frozen) and a
prediction period (used only to score predictions).

Reuses `src.mola.MoLA` as-is (no MoLA-side model code needed -- `.fit`,
`.predict_proba`, `.predict_item_proba`, `.score` already cover
population fitting, learner profiling, and prediction) and
`src.cdm.neural_cdm.NeuralCDM` for the NCDM side. Scoring reuses
`src.simulations.metrics`'s `masked_nll`/`brier_score`/`masked_auc`
and `src.simulations.run`'s `split_holdout`, matching the pattern already
established there (infer a posterior/embedding from a learner's visible
responses, then score predictions on a disjoint held-out subset) rather
than introducing a second one.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.mola import MoLA
from src.cdm.neural_cdm import NeuralCDM
from src.simulations.metrics import masked_nll, brier_score, masked_auc
from src.simulations.run import split_holdout


def split_learners(
    responses_df: pd.DataFrame,
    param_frac: float = 0.7,
    m_select_frac: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Learner-level 3-way split: `param_fit` / `m_select` / `evaluation`.

    `param_frac` of all eligible learners (unique `user_id` in
    `responses_df`) go to parameter estimation; the rest go to
    `evaluation`. The parameter-estimation group is further split so that
    `m_select_frac` of *it* goes to `m_select` and the remainder to
    `param_fit` -- e.g. the defaults give 70% * 80% = 56% param_fit,
    70% * 20% = 14% m_select, 30% evaluation, matching
    notebooks/high-dim-experiment.ipynb's design diagram.

    Returns `(param_fit_ids, m_select_ids, evaluation_ids)`, three
    disjoint arrays of `user_id` whose union is every learner in
    `responses_df`.
    """
    rng = np.random.default_rng(seed)
    learner_ids = responses_df["user_id"].unique()
    rng.shuffle(learner_ids)

    n_param_est = int(round(param_frac * len(learner_ids)))
    param_est_ids, evaluation_ids = learner_ids[:n_param_est], learner_ids[n_param_est:]

    n_m_select = int(round(m_select_frac * len(param_est_ids)))
    m_select_ids, param_fit_ids = param_est_ids[:n_m_select], param_est_ids[n_m_select:]

    return param_fit_ids, m_select_ids, evaluation_ids


def build_item_index(questions_df: pd.DataFrame) -> dict:
    """Maps each `question_id` in `questions_df` (as returned by
    `src.utils.ednet.load_ednet_q_matrix`) to its row in the aligned
    Q-matrix -- the shared indexing every function below assumes for
    `item_id_to_row`."""
    return {qid: row for row, qid in enumerate(questions_df.index)}


def _build_response_matrix(
    df: pd.DataFrame, user_ids: np.ndarray, item_id_to_row: dict, n_items: int
) -> tuple[np.ndarray, dict]:
    """Pivots long-form `(user_id, question_id, correct)` rows for
    `user_ids` into a dense `(len(user_ids), n_items)` array (`NaN`
    for unobserved learner-item pairs), the shape MoLA's `.fit`/
    `.predict_proba`/`.predict_item_proba` expect. Returns the matrix and
    the `user_id -> row` mapping used to build it (0-indexed in the order
    of `user_ids`).
    """
    user_id_to_row = {uid: row for row, uid in enumerate(user_ids)}
    df = df[df["user_id"].isin(user_id_to_row)]
    X = np.full((len(user_ids), n_items), np.nan)
    rows = df["user_id"].map(user_id_to_row).to_numpy()
    cols = df["question_id"].map(item_id_to_row).to_numpy()
    X[rows, cols] = df["correct"].to_numpy(dtype=float)
    return X, user_id_to_row


def select_mola_m(
    param_fit_df: pd.DataFrame,
    m_select_df: pd.DataFrame,
    Q: np.ndarray,
    item_id_to_row: dict,
    m_grid: list,
    holdout_frac: float = 0.2,
    seed: int = 0,
    **mola_kwargs,
) -> tuple[int, dict]:
    """Picks MoLA's number of archetypes `M` from `m_grid` by held-out
    predictive log-likelihood on `m_select` learners.

    For each candidate `M`: fit `MoLA(Q, n_components=M)` on
    `param_fit_df`'s full response matrix; infer `m_select` learners'
    archetype responsibilities from a random per-response subset of
    *their own* responses (`split_holdout`), then score predicted
    probabilities against the complementary held-out subset -- the same
    infer-from-visible / score-on-held-out pattern
    `src.simulations.run._score_fit` uses, not a fresh one. Returns
    `(best_m, nll_by_m)`.
    """
    rng = np.random.default_rng(seed)
    n_items = Q.shape[0]

    param_ids = param_fit_df["user_id"].unique()
    X_param, _ = _build_response_matrix(param_fit_df, param_ids, item_id_to_row, n_items)

    select_ids = m_select_df["user_id"].unique()
    X_select, _ = _build_response_matrix(m_select_df, select_ids, item_id_to_row, n_items)
    X_select_train, hold_mask = split_holdout(X_select, holdout_frac, rng)

    nll_by_m = {}
    for m in m_grid:
        model = MoLA(Q=Q, n_components=m, random_state=int(rng.integers(0, 2**31 - 1)), **mola_kwargs)
        model.fit(X_param)
        y_true = X_select[hold_mask]
        y_prob = model.predict_item_proba(X_select_train)[hold_mask]
        nll_by_m[m] = masked_nll(y_true, y_prob)

    best_m = min(nll_by_m, key=nll_by_m.get)
    return best_m, nll_by_m


def fit_population_models(
    population_df: pd.DataFrame,
    Q: np.ndarray,
    item_id_to_row: dict,
    selected_m: int,
    neural_cdm_kwargs: dict | None = None,
    seed: int = 0,
    **mola_kwargs,
) -> tuple[MoLA, NeuralCDM, dict]:
    """Fits both models' FINAL population parameters on `population_df` --
    intended to be the full parameter-estimation set (`param_fit` +
    `m_select` recombined, all 70% of eligible learners), now that `M` has
    already been chosen (`select_mola_m`) and `m_select` is free to
    contribute to the final fit itself (see the module docstring for why
    that isn't leakage).

    Returns `(mola_model, neural_cdm_model, population_user_id_to_row)` --
    the id-to-row mapping is needed by callers that want a
    population-estimation learner's own fitted NeuralCDM embedding (e.g.
    for sanity checks via `NeuralCDM.population_learner_embedding`).
    """
    n_items = Q.shape[0]
    population_ids = population_df["user_id"].unique()

    t0 = time.perf_counter()
    X_param, user_id_to_row = _build_response_matrix(population_df, population_ids, item_id_to_row, n_items)
    mola_model = MoLA(Q=Q, n_components=selected_m, **mola_kwargs)
    mola_model.fit(X_param)
    mola_model.train_time_sec = time.perf_counter() - t0

    # MoLA lazily populates its inference-time `a`/`b` cache on its first
    # ever `predict_item_proba` call, and `get_log_likelihood` takes a
    # numerically different (though similarly valid) code path once that
    # cache exists vs. before -- confirmed directly (identical inputs give
    # slightly different output pre- vs. post-population). Left alone,
    # this pipeline calls `predict_item_proba` once per evaluation learner
    # on this same `mola_model`, so the very first evaluation learner
    # would be scored via a different numerical path than every other one.
    # Populating the cache once here, before any evaluation learner is
    # scored, makes every learner go through the identical path.
    mola_model.update_ab_params(use_norm=True)

    n_skills = Q.shape[1]
    nc_kwargs = dict(neural_cdm_kwargs or {})
    nc_kwargs.setdefault("random_state", seed)
    neural_cdm_model = NeuralCDM(n_skills, n_items, len(population_ids), **nc_kwargs)
    nc_train_df = pd.DataFrame(
        {
            "user_row": population_df["user_id"].map(user_id_to_row).to_numpy(),
            "item_row": population_df["question_id"].map(item_id_to_row).to_numpy(),
            "correct": population_df["correct"].to_numpy(dtype=float),
        }
    )
    neural_cdm_model.fit(nc_train_df, Q)

    return mola_model, neural_cdm_model, user_id_to_row


def evaluate_learner(
    learner_df: pd.DataFrame,
    mola_model: MoLA,
    neural_cdm_model: NeuralCDM,
    Q: np.ndarray,
    item_id_to_row: dict,
    profile_frac: float = 0.6,
    mask: str = "chronological",
    seed: int = 0,
) -> dict:
    """Scores both models on one evaluation learner: split their responses
    `profile_frac`/`1 - profile_frac` (mechanism controlled by `mask`),
    infer a representation (MoLA's archetype posterior; NeuralCDM's
    fitted embedding row) from the profiling period with population
    parameters frozen, and score predictions on the prediction period.

    `mask="chronological"` (default) sorts by `timestamp` first, so
    profiling responses strictly precede prediction responses in time --
    this measures *forecasting*: predicting a learner's future responses
    from their earlier trajectory. `mask="random"` shuffles the learner's
    responses (seeded, independent of timestamp) before the same split --
    this measures *tracking*: reconstructing a learner's current
    proficiency from an arbitrary partial subset of their responses,
    interleaved in time with the ones being predicted rather than
    strictly preceding them. Both masks use the exact same profiling/
    prediction mechanics below; only which responses land in which set
    differs, so a tracking-vs-forecasting performance gap is attributable
    to the temporal-ordering question itself, not a confound.
    """
    if mask == "chronological":
        learner_df = learner_df.sort_values("timestamp").reset_index(drop=True)
    elif mask == "random":
        learner_df = learner_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        raise ValueError(f"mask must be 'chronological' or 'random', got {mask!r}")
    n_profile = int(np.floor(profile_frac * len(learner_df)))
    profile_df = learner_df.iloc[:n_profile]
    predict_df = learner_df.iloc[n_profile:]

    profile_rows = profile_df["question_id"].map(item_id_to_row).to_numpy()
    predict_rows = predict_df["question_id"].map(item_id_to_row).to_numpy()
    y_true = predict_df["correct"].to_numpy(dtype=float)

    # MoLA
    n_items = Q.shape[0]
    X_profile = np.full((1, n_items), np.nan)
    X_profile[0, profile_rows] = profile_df["correct"].to_numpy(dtype=float)
    mola_probs = mola_model.predict_item_proba(X_profile, item_idxs=list(predict_rows))[0]

    # NeuralCDM
    profile_nc_df = pd.DataFrame({"item_row": profile_rows, "correct": profile_df["correct"].to_numpy(dtype=float)})
    stu_emb = neural_cdm_model.profile_learner(profile_nc_df, Q, seed=seed)
    nc_probs = neural_cdm_model.predict_item_proba(stu_emb, predict_rows, Q)

    return {
        "mask": mask,
        "n_profile": len(profile_df),
        "n_predict": len(predict_df),
        "mola_nll": masked_nll(y_true, mola_probs),
        "mola_brier": brier_score(y_true, mola_probs),
        "mola_auc": masked_auc(y_true, mola_probs),
        "neural_cdm_nll": masked_nll(y_true, nc_probs),
        "neural_cdm_brier": brier_score(y_true, nc_probs),
        "neural_cdm_auc": masked_auc(y_true, nc_probs),
    }


def run_evaluation(
    responses_df: pd.DataFrame,
    Q: np.ndarray,
    questions_df: pd.DataFrame,
    m_grid: list,
    param_frac: float = 0.7,
    m_select_frac: float = 0.2,
    profile_frac: float = 0.6,
    masks: tuple[str, ...] = ("chronological", "random"),
    seed: int = 0,
    mola_kwargs: dict | None = None,
    neural_cdm_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Runs the full protocol end to end: split learners, select MoLA's
    `M`, fit both models' population parameters once, then evaluate every
    evaluation-set learner under each mask in `masks` (see
    `evaluate_learner` -- `"chronological"` measures forecasting,
    `"random"` measures tracking; population fitting doesn't depend on
    the mask at all, so it happens once regardless of how many masks are
    scored). Returns a per-(learner, mask) results DataFrame (one row per
    evaluation learner per mask -- `groupby("mask")` to compare tracking
    vs. forecasting) with NLL/Brier/AUC for both models, plus the
    selected `M` and both models' population-fit timings as constant
    columns (so this frame alone documents the run that produced it).
    """
    mola_kwargs = dict(mola_kwargs or {})
    item_id_to_row = build_item_index(questions_df)

    param_fit_ids, m_select_ids, evaluation_ids = split_learners(
        responses_df, param_frac=param_frac, m_select_frac=m_select_frac, seed=seed
    )
    param_fit_df = responses_df[responses_df["user_id"].isin(param_fit_ids)]
    m_select_df = responses_df[responses_df["user_id"].isin(m_select_ids)]
    evaluation_df = responses_df[responses_df["user_id"].isin(evaluation_ids)]

    selected_m, m_select_nll = select_mola_m(
        param_fit_df, m_select_df, Q, item_id_to_row, m_grid, seed=seed, **mola_kwargs
    )

    # M is chosen; m_select is now free to contribute to the final fit
    # itself (see module docstring) -- both models train on the full 70%.
    population_df = pd.concat([param_fit_df, m_select_df], ignore_index=True)
    mola_model, neural_cdm_model, _ = fit_population_models(
        population_df, Q, item_id_to_row, selected_m,
        neural_cdm_kwargs=neural_cdm_kwargs, seed=seed, **mola_kwargs
    )

    records = []
    for user_id, learner_df in evaluation_df.groupby("user_id"):
        for mask in masks:
            result = evaluate_learner(
                learner_df, mola_model, neural_cdm_model, Q, item_id_to_row,
                profile_frac, mask=mask, seed=seed,
            )
            result["user_id"] = user_id
            records.append(result)

    results = pd.DataFrame.from_records(records)
    results["selected_m"] = selected_m
    results["mola_train_time_sec"] = getattr(mola_model, "train_time_sec", float("nan"))
    results["neural_cdm_train_time_sec"] = neural_cdm_model.train_time_sec
    results.attrs["m_select_nll"] = m_select_nll
    return results
