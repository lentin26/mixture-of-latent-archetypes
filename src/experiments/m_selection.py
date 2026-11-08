"""Compares rules for choosing the number of archetypes against a known truth.

Run from the repository root so that ``import src...`` resolves:

    python -m src.experiments.m_selection

    # a quick pass; the full design takes roughly 90 minutes
    python -m src.experiments.m_selection --seeds 3

Three rules are scored on the same fits:

``argmin``
    The candidate with the lowest held-out negative log-likelihood. Standard,
    citation-backed, and the rule the reported experiments use.
``elbow``
    The smallest candidate within ``--tol`` nats of that minimum. Reported for
    continuity with the earlier analysis; the tolerance is a free constant
    chosen by hand, which is the reason for looking at the third rule.
``icl``
    The integrated completed likelihood of Biernacki, Celeux and Govaert:
    BIC plus twice the classification entropy
    ``-sum_n sum_m gamma_nm log gamma_nm``. The entropy term grows as learners
    are split across archetypes rather than claimed by one, which is what an
    unnecessary component produces -- an overfitted mixture either empties the
    extra component or duplicates an existing one, and both show up as fuzzier
    allocations. Nothing here is tuned: the ``log N`` penalty comes from BIC's
    Laplace approximation and the entropy from integrating the completed
    likelihood.

Two definitional choices do enter the ICL, neither of them a dial to turn.
``N`` is the number of learners rather than the number of responses, since the
learners are the independent units of a mixture; and the parameter count is
MoLA's own ``get_param_count``, which counts ``M`` mixture weights rather than
the ``M - 1`` that are free. Both are held fixed across candidates, so they
shift every candidate's ICL by nearly the same amount and leave the argmin
essentially unaffected.

The design mirrors the earlier elbow-vs-argmin analysis exactly -- the same
true archetype counts, separation levels, seeds, candidate grids and skill
counts -- so the ``argmin`` and ``elbow`` columns are directly comparable to
what that analysis reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from dataclasses import replace

from src.simulations.factors import SEPARATION_LEVELS, SimulationCondition
from src.simulations.run import default_m_grid, run_m_sweep

TRUE_MS = (2, 4, 8, 16)
N_LEARNERS = 10_000
DEFAULT_TOL = 0.002
CACHE = Path("data/m_selection_rules.json")


def icl(model, X) -> float:
    """Integrated completed likelihood on ``X``; lower is better.

    Both terms are evaluated on ``X``, and the caller is expected to pass the
    matrix the model was fit to. Scoring a holdout-fitted model on the full
    matrix would not do: a larger ``M`` overfits the training portion more, so
    its full-matrix likelihood suffers more, which biases the criterion toward
    smaller ``M`` on top of the penalty it already applies. The sweep
    therefore fits a separate full-matrix model for this rule, which costs an
    extra fit per candidate but lets each rule be computed the way it is meant
    to be -- the held-out rules need a holdout, and this one does not.

    ``N`` is the learner count rather than the response count, the learners
    being the independent units of the mixture.
    """
    post = np.asarray(model.predict_proba(X))
    entropy = -(post * np.log(np.clip(post, 1e-300, None))).sum()
    return float(
        2.0 * model.get_avg_nll_from_data(X)[0]
        + model.get_param_count() * np.log(post.shape[0])
        + 2.0 * entropy
    )


def _select(curve: list[tuple[int, float, float, float]], tol: float) -> dict:
    """Apply the four rules to one dataset's sweep.

    ``curve`` is ``(m, holdout_nll, holdout_nll_se, icl)`` per candidate, in
    ascending ``m``.

    ``one_se`` is the one-standard-error rule: the smallest candidate whose
    held-out NLL is within one standard error of the best. It has the same
    shape as ``elbow`` -- smallest model within a tolerance of the minimum --
    but takes the tolerance from the sampling error of the held-out estimate
    rather than from a chosen constant, which is how the rule is normally
    stated for cross-validated model selection and how the gap statistic
    selects the number of clusters.
    """
    ms = [m for m, _, _, _ in curve]
    nlls = [n for _, n, _, _ in curve]
    icls = [c for _, _, _, c in curve]
    best_ix = int(np.argmin(nlls))
    best = nlls[best_ix]
    se = curve[best_ix][2]
    return {
        "argmin": ms[best_ix],
        "elbow": next(m for m, n, _, _ in curve if n <= best + tol),
        "one_se": next(m for m, n, _, _ in curve if n <= best + se),
        "icl": ms[int(np.argmin(icls))],
    }


def run(seeds: int = 25, tol: float = DEFAULT_TOL) -> pd.DataFrame:
    rows = []
    total = len(TRUE_MS) * len(SEPARATION_LEVELS) * seeds
    done = 0
    for true_m in TRUE_MS:
        grid = default_m_grid(true_m, n_points=7)
        n_skills = max(20, 2 * true_m)
        for sep in SEPARATION_LEVELS:
            for seed in range(seeds):
                cond = SimulationCondition(
                    n_archetypes=true_m,
                    n_skills=n_skills,
                    n_learners=N_LEARNERS,
                    archetype_separation=sep,
                    initialization="k-means",
                )
                # Held-out rules read the sweep that masks a fifth of the
                # responses; ICL reads a second sweep fit on everything, since
                # it is an in-sample criterion. holdout_frac=0 still masks one
                # response, which split_holdout enforces as a floor.
                held_out = {
                    int(r.metrics["n_components_fit"]): (
                        float(r.metrics["holdout_nll"]),
                        float(r.metrics["holdout_nll_se"]),
                    )
                    for r in run_m_sweep(cond, grid, seed=seed)
                }
                in_sample = {
                    int(r.metrics["n_components_fit"]): icl(r.model, r.data.X)
                    for r in run_m_sweep(replace(cond, holdout_frac=0.0), grid, seed=seed)
                }
                curve = sorted((m, *held_out[m], in_sample[m]) for m in grid)
                picked = _select(curve, tol)
                row = {"true_m": true_m, "sep": sep, "seed": seed}
                for rule, m_hat in picked.items():
                    row[f"m_{rule}"] = m_hat
                    row[f"err_{rule}"] = m_hat - true_m
                    row[f"abs_{rule}"] = abs(m_hat - true_m)
                    row[f"exact_{rule}"] = m_hat == true_m
                rows.append(row)
                done += 1
                print(f"  [{done}/{total}] true_m={true_m} sep={sep} seed={seed} -> {picked}", flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rules = ("argmin", "elbow", "one_se", "icl")
    agg = {}
    for rule in rules:
        agg[f"exact_{rule}"] = (f"exact_{rule}", "mean")
        agg[f"signed_{rule}"] = (f"err_{rule}", "mean")
        agg[f"abs_{rule}"] = (f"abs_{rule}", "mean")
    return df.groupby("sep").agg(**agg).loc[list(SEPARATION_LEVELS)]


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    """ICL against each other rule on absolute selection error, by separation.

    Paired within dataset and tested with a Wilcoxon signed-rank test, matching
    how the earlier elbow-vs-argmin comparison was tested.
    """
    out = []
    for sep in SEPARATION_LEVELS:
        sub = df[df["sep"] == sep]
        for other in ("argmin", "elbow", "one_se"):
            a, b = sub["abs_icl"].to_numpy(), sub[f"abs_{other}"].to_numpy()
            p = float("nan") if np.all(a == b) else float(wilcoxon(a, b).pvalue)
            out.append(
                {
                    "sep": sep,
                    "contrast": f"icl vs {other}",
                    "mean_abs_icl": a.mean(),
                    f"mean_abs_{other}": b.mean(),
                    "p": p,
                }
            )
    return pd.DataFrame(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--cache", default=str(CACHE))
    parser.add_argument("--refresh", action="store_true", help="ignore any cached rows")
    args = parser.parse_args(argv)

    cache = Path(args.cache)
    if cache.exists() and not args.refresh:
        df = pd.DataFrame(json.loads(cache.read_text()))
        print(f"loaded {len(df)} cached rows from {cache}")
    else:
        df = run(seeds=args.seeds, tol=args.tol)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(df.to_dict("records")))
        print(f"\nwrote {len(df)} rows to {cache}")

    print(f"\n{'=' * 78}\nSelection rules by separation ({df.seed.nunique()} seeds per true M)\n{'=' * 78}")
    print(summarize(df).round(3).to_string())
    print(f"\n{'=' * 78}\nICL against the others, paired Wilcoxon on absolute error\n{'=' * 78}")
    print(paired_tests(df).round(4).to_string(index=False))
    print(f"\n{'=' * 78}\nExact-match rate by true M\n{'=' * 78}")
    print(
        df.groupby("true_m")[["exact_argmin", "exact_elbow", "exact_one_se", "exact_icl"]]
        .mean()
        .round(3)
        .to_string()
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
