"""EdNet-KT1 data acquisition and loading, for the real-data MoLA-vs-
NeuralCDM evaluation (see notebooks/high-dim-experiment.ipynb and
src/experiments/high_dim.py).

EdNet (https://github.com/riiid/ednet) ships several variants of
increasing complexity (KT1-KT4). This uses **KT1** specifically: simple
`(learner, item, correct)` triples plus a Q-matrix are all this evaluation
needs -- KT1 doesn't carry the action-sequence/study-mode/lecture-watching
detail the richer KT2-KT4 variants add, and is far smaller (KT1: 1.2GB
compressed / 784,309 per-user files; KT2 alone is 555.8MB / 297,444 files
just for its own additional detail).

Neither dataset is bundled with this repo (both are multi-GB) --
``download_ednet_kt1`` fetches and caches the raw archives locally on
first use; nothing here extracts the ~800K per-user files to disk
individually (reads members directly out of the zip instead, since only a
sampled subset of users is ever needed for one run).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import gdown
import numpy as np
import pandas as pd

# EdNet's public archives are hosted on Google Drive (riiid/ednet's own
# short links -- https://bit.ly/ednet_kt1 and https://bit.ly/ednet-content
# -- just redirect to these file views). Plain `requests` can't fetch them
# directly: for a file this large, Drive serves an HTML "can't scan this
# file for viruses" interstitial instead of the raw bytes (this is exactly
# what broke here the first time -- the cached ``.zip`` turned out to be a
# Google Drive viewer HTML page, not a zip). `gdown` handles that
# interstitial/confirmation-token dance; a plain URL can't.
KT1_FILE_ID = "1AmGcOs5U31wIIqvthn9ARqJMrMTFTcaw"  # EdNet-KT1.zip, ~1.2GB
CONTENT_FILE_ID = "117aYJAWG3GU48suS66NPaB82HwFj6xWS"  # EdNet-Contents.zip

# Confirmed against a real downloaded per-user file (KT1/u<id>.csv has no
# bundle_id -- that column only exists in questions.csv).
KT1_RESPONSE_COLUMNS = ["timestamp", "solving_id", "question_id", "user_answer", "elapsed_time"]


def _download_to_cache(file_id: str, dest: Path) -> Path:
    """Download Google Drive file `file_id` to `dest` if not already
    cached (and not left over as a bad file from a previous failed
    attempt -- validated as an actual zip before being trusted); return
    `dest`.
    """
    if dest.exists():
        if zipfile.is_zipfile(dest):
            return dest
        dest.unlink()  # stale/corrupt download (e.g. an HTML error page saved by mistake) -- redo it

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    gdown.download(id=file_id, output=str(tmp), quiet=False)

    if not zipfile.is_zipfile(tmp):
        raise ValueError(
            f"downloaded file for Google Drive id={file_id!r} is not a valid zip "
            f"(Drive may have served an HTML page instead of the file -- check "
            f"{tmp} directly, and that the file id above is still current)"
        )
    tmp.rename(dest)
    return dest


def download_ednet_kt1(cache_dir: str | Path = "data/ednet") -> tuple[Path, Path]:
    """Download (if not already cached) the EdNet-KT1 response archive and
    the EdNet-Content archive (which contains ``questions.csv``, needed
    for correct answers and skill tags). Returns
    ``(kt1_zip_path, content_zip_path)``. Neither is extracted -- see
    `load_ednet_responses`/`load_ednet_q_matrix`, which read members
    directly out of the zip.
    """
    cache_dir = Path(cache_dir)
    kt1_zip = _download_to_cache(KT1_FILE_ID, cache_dir / "ednet_kt1.zip")
    content_zip = _download_to_cache(CONTENT_FILE_ID, cache_dir / "ednet_content.zip")
    return kt1_zip, content_zip


def _find_member(zf: zipfile.ZipFile, filename: str) -> str:
    """Find a member of `zf` whose basename is `filename`, regardless of
    which subdirectory the archive unpacks it into (unconfirmed at
    write-time, since the archives haven't been downloaded yet -- this
    searches rather than assuming one exact internal path)."""
    matches = [n for n in zf.namelist() if n.endswith(filename)]
    if not matches:
        raise FileNotFoundError(f"{filename!r} not found in archive (looked at {len(zf.namelist())} members)")
    return matches[0]


def load_ednet_responses(
    cache_dir: str | Path = "data/ednet",
    max_users: int | None = None,
    min_responses: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """Load eligible learners' responses from the cached EdNet-KT1 archive.

    Iterates per-user CSV members directly out of the zip (never extracts
    all ~800K of them to disk), keeping only users with at least
    `min_responses` rows. If `max_users` is given, randomly samples that
    many *candidate* member names first (seeded) before reading them, so
    a small run doesn't have to scan the whole archive -- the final
    eligible count can still be below `max_users` if some sampled users
    don't meet `min_responses`.

    Returns a long-form DataFrame: `user_id, timestamp, solving_id,
    question_id, user_answer, elapsed_time` (see `KT1_RESPONSE_COLUMNS`),
    sorted within each user by `timestamp` (needed for the chronological
    profile/prediction split downstream) -- NOT yet joined against
    `correct_answer` (see `load_ednet_q_matrix`, which returns the
    correctness join alongside the Q-matrix since both come from the same
    `questions.csv`; use `derive_correctness` to do that join).
    """
    kt1_zip, _ = download_ednet_kt1(cache_dir)
    rng = np.random.default_rng(seed)

    with zipfile.ZipFile(kt1_zip) as zf:
        members = [n for n in zf.namelist() if n.endswith(".csv")]
        if max_users is not None and max_users < len(members):
            members = list(rng.choice(members, size=max_users, replace=False))

        frames = []
        for member in members:
            user_id = Path(member).stem  # "u12345.csv" -> "u12345"
            with zf.open(member) as f:
                df = pd.read_csv(f)
            if len(df) < min_responses:
                continue
            df = df.sort_values("timestamp").reset_index(drop=True)
            df.insert(0, "user_id", user_id)
            frames.append(df)

    if not frames:
        raise ValueError(
            f"no users met min_responses={min_responses} out of {len(members)} sampled -- "
            f"try a larger max_users"
        )
    return pd.concat(frames, ignore_index=True)


def _parse_q_matrix(questions_df: pd.DataFrame) -> tuple[np.ndarray, dict[int, int]]:
    """Pure tag-parsing step of `load_ednet_q_matrix`, factored out so it
    can be unit-tested on a small hand-built `questions_df` without a
    real EdNet download. `questions_df` must have a `tags` column
    (semicolon-separated integer skill codes, possibly `NaN`).

    Returns `(Q, skill_code_to_column)` -- see `load_ednet_q_matrix`.
    """
    tag_lists = questions_df["tags"].fillna("").apply(
        lambda s: [int(t) for t in str(s).split(";") if t.strip().isdigit()]
    )
    skill_codes = sorted({code for tags in tag_lists for code in tags})
    skill_code_to_column = {code: col for col, code in enumerate(skill_codes)}

    Q = np.zeros((len(questions_df), len(skill_codes)), dtype=float)
    for row, tags in enumerate(tag_lists):
        for code in tags:
            Q[row, skill_code_to_column[code]] = 1.0

    return Q, skill_code_to_column


def load_ednet_q_matrix(
    cache_dir: str | Path = "data/ednet",
    question_ids: pd.Series | np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame, dict[int, int]]:
    """Load the Q-matrix and correct-answer key from EdNet-Content's
    `questions.csv`.

    Returns `(Q, questions_df, skill_code_to_column)`:
    - `Q`: `(n_items, n_skills)` binary array, restricted to
      `question_ids` if given (else every question in the file) --
      `Q[i, k] = 1` iff item `i`'s `tags` (semicolon-separated skill
      codes) includes the skill at column `k`.
    - `questions_df`: the raw table restricted the same way, indexed by
      `question_id`, with `correct_answer` -- join this against
      `load_ednet_responses`'s `user_answer` to derive a binary
      `correct` column (`user_answer == correct_answer`).
    - `skill_code_to_column`: maps each raw EdNet skill/tag code to its
      column index in `Q`, for reporting real skill identities later.
    """
    _, content_zip = download_ednet_kt1(cache_dir)
    with zipfile.ZipFile(content_zip) as zf:
        member = _find_member(zf, "questions.csv")
        with zf.open(member) as f:
            questions_df = pd.read_csv(f)
    questions_df = questions_df.set_index("question_id")

    if question_ids is not None:
        questions_df = questions_df.loc[questions_df.index.intersection(question_ids)]

    Q, skill_code_to_column = _parse_q_matrix(questions_df)

    # Drop items with no tagged skill (a missing `tags` field parses to an
    # all-zero Q row): with the row-normalized Q used at fit/inference time,
    # such a row makes the per-item normalization in `compute_ab_params`
    # collapse to exactly 0, producing an infinite log-likelihood term for
    # any learner who answered that item.
    has_skill = Q.sum(axis=1) > 0
    Q = Q[has_skill]
    questions_df = questions_df.loc[has_skill]

    return Q, questions_df, skill_code_to_column


def derive_correctness(responses_df: pd.DataFrame, questions_df: pd.DataFrame) -> pd.DataFrame:
    """Joins `load_ednet_responses`'s output against `questions_df` (from
    `load_ednet_q_matrix`) to add a binary `correct` column
    (`user_answer == correct_answer`), and restricts to responses whose
    `question_id` is present in `questions_df` (i.e. has a known correct
    answer and skill tags). Returns a new DataFrame; does not mutate
    either input.
    """
    correct_answer = questions_df["correct_answer"]
    merged = responses_df.merge(
        correct_answer.rename("__correct_answer"), left_on="question_id", right_index=True, how="inner"
    )
    merged["correct"] = (merged["user_answer"] == merged["__correct_answer"]).astype(float)
    return merged.drop(columns="__correct_answer")


def filter_items_by_support(responses_df: pd.DataFrame, min_item_responses: int = 10) -> pd.DataFrame:
    """Restricts `responses_df` to items answered at least
    `min_item_responses` times overall.

    EdNet's item catalog is far larger than what any modest learner
    sample covers with any depth -- e.g. 200 sampled KT1 users (60
    meeting the usual `min_responses=30` threshold) touch over 7,000
    distinct items, most of them seen by only one or two learners.
    Fitting MoLA (or NeuralCDM) with that many near-unobserved items
    leaves their item/skill parameters driven almost entirely by the
    prior with a handful of extreme observations, which can push the
    E-/M-step's `theta`/`mu` estimates toward exactly 0 or 1 and produce
    `NaN`s downstream (confirmed against a real smoke-scale download --
    not a hypothetical). Restricting to reasonably well-observed items
    before fitting is standard practice for long-tailed real response
    data and avoids this instability. Apply this **before** building
    `question_ids` for `load_ednet_q_matrix`, so the Q-matrix and
    `responses_df` stay restricted to the same item set.
    """
    support = responses_df["question_id"].value_counts()
    keep = support[support >= min_item_responses].index
    return responses_df[responses_df["question_id"].isin(keep)]


def _run_filter_pipeline(
    responses_raw: pd.DataFrame, min_responses: int, min_item_responses: int, cache_dir: str | Path
) -> pd.DataFrame:
    """Shared body of the item-support -> Q-matrix (drops untagged items)
    -> eligibility-recheck sequence, applied twice by
    `sample_eligible_learners` (once on the oversampled raw pool, once
    again on the exact final subsample, since trimming users can itself
    drop items/learners back under threshold)."""
    responses_raw = filter_items_by_support(responses_raw, min_item_responses=min_item_responses)
    counts = responses_raw["user_id"].value_counts()
    eligible = counts[counts >= min_responses].index
    responses_raw = responses_raw[responses_raw["user_id"].isin(eligible)]

    Q, questions_df, _ = load_ednet_q_matrix(cache_dir=cache_dir, question_ids=responses_raw["question_id"].unique())
    responses_df = derive_correctness(responses_raw, questions_df)

    counts = responses_df["user_id"].value_counts()
    eligible = counts[counts >= min_responses].index
    return responses_df[responses_df["user_id"].isin(eligible)]


def sample_eligible_learners(
    n_learners: int,
    *,
    min_responses: int = 30,
    min_item_responses: int = 30,
    cache_dir: str | Path = "data/ednet",
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Returns exactly `n_learners` eligible learners' responses (plus
    their restricted `Q`/`questions_df`), by oversampling raw KT1 users
    and retrying with a larger raw pool as needed, rather than sampling
    a fixed raw `max_users` and hoping enough survive filtering.

    The eligible fraction of a raw sample isn't knowable in advance and
    is volatile below a critical mass of raw users -- e.g. at
    `min_item_responses=30`, 100 raw users yields 0 eligible learners
    (no item clears the support bar at all), 200 yields 5, but the rate
    stabilizes around 26-34% for raw >= 500. So a fixed `max_users` can
    "sometimes" yield too few (or zero) eligible learners depending on
    where the requested scale happens to fall relative to that cliff.
    This instead starts from a raw guess comfortably past the cliff and
    past a conservative rate estimate, and grows it until enough eligible
    learners are found.

    Once enough eligible learners exist, subsamples down to exactly
    `n_learners` (seeded). Each learner's own response count is
    unaffected by removing *other* learners, so `min_responses` continues
    to hold exactly for every returned learner with no further check
    needed. `min_item_responses`, by contrast, is an aggregate-across-
    learners count, so it was enforced on the larger oversampled pool the
    subsample is drawn from, not re-verified on the exact final subset --
    confirmed empirically that re-verifying it here is actively
    counterproductive: a larger raw pool pulls in more long-tail items
    that sit just barely above the support threshold in aggregate (e.g.
    at raw=4096, the full pool has 4916 items, but trimming to 200
    learners leaves a *median* item support of 8, far below 30 -- versus
    raw=1000's smaller, less diluted pool leaving a much healthier
    profile after the same trim), so aggressively oversampling to make a
    strict post-trim recheck pass makes the result worse, not better.
    `min_item_responses` is a soft data-quality knob rather than a
    correctness requirement in any case -- MoLA handles low/zero-response
    items cleanly via its prior (see `src/mola/train.py`'s `update_theta`/
    `update_mu`), so a handful of items landing under the nominal
    threshold after this final trim isn't a correctness issue. Q and
    questions_df are restricted to items actually present (at least one
    response) in the returned data, for accurate reporting.

    Raises `ValueError` if `n_learners` isn't reachable within a small,
    fixed number of retries (the raw pool is capped by how many distinct
    KT1 users exist at all).
    """
    raw = max(1000, int(np.ceil(n_learners / 0.25)))
    rng = np.random.default_rng(seed)
    last_n = 0

    for _ in range(6):
        responses_raw = load_ednet_responses(cache_dir=cache_dir, max_users=raw, min_responses=min_responses, seed=seed)
        responses_df = _run_filter_pipeline(responses_raw, min_responses, min_item_responses, cache_dir)
        eligible_ids = responses_df["user_id"].unique()
        last_n = len(eligible_ids)

        if last_n >= n_learners:
            chosen = rng.choice(sorted(eligible_ids), size=n_learners, replace=False)
            responses_df = responses_df[responses_df["user_id"].isin(chosen)]
            Q, questions_df, _ = load_ednet_q_matrix(
                cache_dir=cache_dir, question_ids=responses_df["question_id"].unique()
            )
            return responses_df, Q, questions_df

        raw = int(np.ceil(raw * 1.6))

    raise ValueError(
        f"could not reach {n_learners} eligible learners (min_responses={min_responses}, "
        f"min_item_responses={min_item_responses}) after 6 attempts -- last attempt reached {last_n}"
    )
