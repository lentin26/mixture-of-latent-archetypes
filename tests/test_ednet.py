"""Tests for EdNet-KT1 data loading (``src/utils/ednet.py``).

No real EdNet download is used -- ``load_ednet_responses`` is exercised
against a small zip built on the fly in a tmp path (by monkeypatching
``download_ednet_kt1``'s return value), and ``_parse_q_matrix`` is tested
directly on a hand-built ``questions_df``.

Run from the repo root:

    pytest tests/test_ednet.py -q
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.ednet import (
    _download_to_cache,
    _parse_q_matrix,
    derive_correctness,
    filter_items_by_support,
    load_ednet_q_matrix,
    load_ednet_responses,
    sample_eligible_learners,
)


def test_parse_q_matrix_marks_tagged_skills():
    questions_df = pd.DataFrame(
        {"tags": ["1;3", "2", "", None, "3;1"]},
        index=pd.Index([10, 11, 12, 13, 14], name="question_id"),
    )
    Q, skill_code_to_column = _parse_q_matrix(questions_df)

    assert set(skill_code_to_column) == {1, 2, 3}
    assert Q.shape == (5, 3)

    col1, col2, col3 = skill_code_to_column[1], skill_code_to_column[2], skill_code_to_column[3]
    np.testing.assert_array_equal(Q[0, [col1, col3]], [1.0, 1.0])
    assert Q[0].sum() == 2  # only tags 1 and 3 set for row 0
    assert Q[1, col2] == 1.0
    assert Q[2].sum() == 0  # empty tags -> no skills
    assert Q[3].sum() == 0  # NaN tags -> no skills
    np.testing.assert_array_equal(Q[4], Q[0])  # "3;1" same skill set as "1;3"


def test_parse_q_matrix_ignores_non_numeric_tags():
    questions_df = pd.DataFrame({"tags": ["1;abc;2"]}, index=pd.Index([0], name="question_id"))
    Q, skill_code_to_column = _parse_q_matrix(questions_df)
    assert set(skill_code_to_column) == {1, 2}
    assert Q[0].sum() == 2


def _build_kt1_zip(path: Path, user_responses: dict[str, pd.DataFrame]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for user_id, df in user_responses.items():
            zf.writestr(f"{user_id}.csv", df.to_csv(index=False))
    return path


def _build_content_zip(path: Path, questions_df: pd.DataFrame) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("questions.csv", questions_df.to_csv(index=False))
    return path


def test_load_ednet_q_matrix_drops_untagged_items(tmp_path, monkeypatch):
    # A question with no `tags` (real EdNet has this for ~6% of its
    # catalog) parses to an all-zero Q row -- which, left in, makes
    # per-item inference-time normalization collapse to exactly 0 and
    # produce an infinite log-likelihood term (see src/mola/train.py's
    # compute_ab_params). It must be dropped, not just parsed as empty.
    content_zip = _build_content_zip(
        tmp_path / "content.zip",
        pd.DataFrame(
            {
                "question_id": ["q1", "q2", "q3"],
                "correct_answer": ["a", "b", "a"],
                "tags": ["1;2", None, "3"],
            }
        ),
    )
    monkeypatch.setattr(
        "src.utils.ednet.download_ednet_kt1",
        lambda cache_dir="data/ednet": (tmp_path / "kt1.zip", content_zip),
    )

    Q, questions_df, skill_code_to_column = load_ednet_q_matrix(cache_dir=str(tmp_path))

    assert list(questions_df.index) == ["q1", "q3"]
    assert Q.shape == (2, len(skill_code_to_column))
    assert (Q.sum(axis=1) > 0).all()


def test_load_ednet_responses_filters_by_min_responses(tmp_path, monkeypatch):
    zip_path = _build_kt1_zip(
        tmp_path / "kt1.zip",
        {
            "u1": pd.DataFrame({"timestamp": [3, 1, 2], "question_id": ["q1", "q2", "q3"]}),  # 3 rows
            "u2": pd.DataFrame({"timestamp": [1], "question_id": ["q1"]}),  # 1 row -- filtered out
        },
    )
    monkeypatch.setattr(
        "src.utils.ednet.download_ednet_kt1",
        lambda cache_dir="data/ednet": (zip_path, tmp_path / "content.zip"),
    )

    result = load_ednet_responses(cache_dir=str(tmp_path), min_responses=2)

    assert set(result["user_id"]) == {"u1"}
    assert len(result) == 3


def test_load_ednet_responses_sorts_within_user_by_timestamp(tmp_path, monkeypatch):
    zip_path = _build_kt1_zip(
        tmp_path / "kt1.zip",
        {"u1": pd.DataFrame({"timestamp": [3, 1, 2], "question_id": ["qA", "qB", "qC"]})},
    )
    monkeypatch.setattr(
        "src.utils.ednet.download_ednet_kt1",
        lambda cache_dir="data/ednet": (zip_path, tmp_path / "content.zip"),
    )

    result = load_ednet_responses(cache_dir=str(tmp_path), min_responses=1)

    assert result["timestamp"].tolist() == [1, 2, 3]
    assert result["question_id"].tolist() == ["qB", "qC", "qA"]


def test_load_ednet_responses_max_users_is_seeded_and_deterministic(tmp_path, monkeypatch):
    responses = {
        f"u{i}": pd.DataFrame({"timestamp": range(5), "question_id": [f"q{i}"] * 5}) for i in range(10)
    }
    zip_path = _build_kt1_zip(tmp_path / "kt1.zip", responses)
    monkeypatch.setattr(
        "src.utils.ednet.download_ednet_kt1",
        lambda cache_dir="data/ednet": (zip_path, tmp_path / "content.zip"),
    )

    result_a = load_ednet_responses(cache_dir=str(tmp_path), max_users=3, min_responses=1, seed=0)
    result_b = load_ednet_responses(cache_dir=str(tmp_path), max_users=3, min_responses=1, seed=0)

    assert set(result_a["user_id"]) == set(result_b["user_id"])
    assert result_a["user_id"].nunique() == 3


def test_load_ednet_responses_raises_when_nobody_meets_threshold(tmp_path, monkeypatch):
    zip_path = _build_kt1_zip(
        tmp_path / "kt1.zip",
        {"u1": pd.DataFrame({"timestamp": [1], "question_id": ["q1"]})},
    )
    monkeypatch.setattr(
        "src.utils.ednet.download_ednet_kt1",
        lambda cache_dir="data/ednet": (zip_path, tmp_path / "content.zip"),
    )

    with pytest.raises(ValueError, match="min_responses"):
        load_ednet_responses(cache_dir=str(tmp_path), min_responses=30)


def _write_zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("member.csv", "a,b\n1,2\n")
    return path


def test_download_to_cache_skips_download_when_already_a_valid_zip(tmp_path, monkeypatch):
    dest = _write_zip(tmp_path / "cached.zip")
    calls = []
    monkeypatch.setattr("src.utils.ednet.gdown.download", lambda **kwargs: calls.append(kwargs))

    result = _download_to_cache("some-file-id", dest)

    assert result == dest
    assert calls == []  # never re-downloaded


def test_download_to_cache_redownloads_a_stale_non_zip_file(tmp_path, monkeypatch):
    dest = tmp_path / "cached.zip"
    dest.write_text("<!DOCTYPE html>not actually a zip")  # simulates a Drive interstitial saved by mistake

    def fake_download(id, output, quiet):
        _write_zip(Path(output))

    monkeypatch.setattr("src.utils.ednet.gdown.download", fake_download)

    result = _download_to_cache("some-file-id", dest)

    assert zipfile.is_zipfile(result)


def test_download_to_cache_raises_when_download_is_not_a_zip(tmp_path, monkeypatch):
    dest = tmp_path / "cached.zip"

    def fake_download(id, output, quiet):
        Path(output).write_text("<!DOCTYPE html>still not a zip")

    monkeypatch.setattr("src.utils.ednet.gdown.download", fake_download)

    with pytest.raises(ValueError, match="not a valid zip"):
        _download_to_cache("some-file-id", dest)
    assert not dest.exists()  # never renamed into place


def test_filter_items_by_support_drops_rare_items():
    responses_df = pd.DataFrame(
        {
            "question_id": ["popular"] * 5 + ["rare"] * 2,
            "user_id": ["u1", "u2", "u3", "u4", "u5", "u1", "u2"],
        }
    )

    result = filter_items_by_support(responses_df, min_item_responses=3)

    assert set(result["question_id"]) == {"popular"}
    assert len(result) == 5


def test_filter_items_by_support_keeps_items_at_exactly_the_threshold():
    responses_df = pd.DataFrame({"question_id": ["q1", "q1", "q2"], "user_id": ["u1", "u2", "u1"]})

    result = filter_items_by_support(responses_df, min_item_responses=2)

    assert set(result["question_id"]) == {"q1"}


def test_derive_correctness_matches_against_correct_answer():
    responses_df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "question_id": [1, 2, 1],
            "user_answer": ["a", "b", "c"],
        }
    )
    questions_df = pd.DataFrame({"correct_answer": ["a", "b"]}, index=pd.Index([1, 2], name="question_id"))

    result = derive_correctness(responses_df, questions_df)

    assert result["correct"].tolist() == [1.0, 1.0, 0.0]


def test_derive_correctness_drops_responses_with_unknown_question_id():
    responses_df = pd.DataFrame({"user_id": ["u1", "u1"], "question_id": [1, 99], "user_answer": ["a", "a"]})
    questions_df = pd.DataFrame({"correct_answer": ["a"]}, index=pd.Index([1], name="question_id"))

    result = derive_correctness(responses_df, questions_df)

    assert result["question_id"].tolist() == [1]
    assert len(result) == 1


def _build_sample_eligible_learners_fixture(tmp_path):
    # 6 users: q1-q3 answered by all 6 (always safe); q4 answered by
    # only 3 of the 6, so it's exactly at a min_item_responses=3
    # threshold in the full pool but at risk of dropping below it once
    # subsampled down to fewer users -- the scenario the final
    # re-verification pass in sample_eligible_learners exists for.
    responses = {}
    for i in range(1, 7):
        rows = {
            "timestamp": [1, 2, 3],
            "solving_id": [1, 2, 3],
            "question_id": ["q1", "q2", "q3"],
            "user_answer": ["a", "a", "a"],
            "elapsed_time": [100, 100, 100],
        }
        if i <= 3:
            for col, val in zip(["timestamp", "solving_id", "question_id", "user_answer", "elapsed_time"],
                                 [4, 4, "q4", "a", 100]):
                rows[col] = rows[col] + [val]
        responses[f"u{i}"] = pd.DataFrame(rows)

    kt1_zip = _build_kt1_zip(tmp_path / "kt1.zip", responses)
    content_zip = _build_content_zip(
        tmp_path / "content.zip",
        pd.DataFrame(
            {
                "question_id": ["q1", "q2", "q3", "q4"],
                "correct_answer": ["a", "a", "a", "a"],
                "tags": ["1", "2", "3", "4"],
            }
        ),
    )
    return kt1_zip, content_zip


def test_sample_eligible_learners_returns_exact_count(tmp_path, monkeypatch):
    kt1_zip, content_zip = _build_sample_eligible_learners_fixture(tmp_path)
    monkeypatch.setattr(
        "src.utils.ednet.download_ednet_kt1", lambda cache_dir="data/ednet": (kt1_zip, content_zip)
    )

    responses_df, Q, questions_df = sample_eligible_learners(
        n_learners=6, min_responses=3, min_item_responses=3, cache_dir=str(tmp_path), seed=0
    )

    assert responses_df["user_id"].nunique() == 6


def test_sample_eligible_learners_preserves_min_responses_after_trim(tmp_path, monkeypatch):
    kt1_zip, content_zip = _build_sample_eligible_learners_fixture(tmp_path)
    monkeypatch.setattr(
        "src.utils.ednet.download_ednet_kt1", lambda cache_dir="data/ednet": (kt1_zip, content_zip)
    )

    # Trimming from 6 eligible learners down to 3 can (and, for q4, does)
    # push some *item's* aggregate support below min_item_responses --
    # not re-enforced on the final subset (see sample_eligible_learners'
    # docstring: re-verifying it turned out to be counterproductive).
    # What must still hold exactly is min_responses per *learner*, since
    # removing other learners can't change a kept learner's own count.
    responses_df, Q, questions_df = sample_eligible_learners(
        n_learners=3, min_responses=3, min_item_responses=3, cache_dir=str(tmp_path), seed=0
    )

    assert responses_df["user_id"].nunique() == 3
    assert (responses_df.groupby("user_id").size() >= 3).all()
    # q1-q3 are answered by every user, so they must survive regardless
    # of which 3 users were chosen.
    assert {"q1", "q2", "q3"}.issubset(set(responses_df["question_id"]))
    # Q/questions_df are restricted to items actually present in the
    # returned data.
    assert set(questions_df.index) == set(responses_df["question_id"])


def test_sample_eligible_learners_raises_when_unreachable(tmp_path, monkeypatch):
    kt1_zip, content_zip = _build_sample_eligible_learners_fixture(tmp_path)
    monkeypatch.setattr(
        "src.utils.ednet.download_ednet_kt1", lambda cache_dir="data/ednet": (kt1_zip, content_zip)
    )

    with pytest.raises(ValueError, match="could not reach"):
        sample_eligible_learners(n_learners=1000, min_responses=3, min_item_responses=3, cache_dir=str(tmp_path), seed=0)
