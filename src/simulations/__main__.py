"""Command-line entry point for the MoLA simulation study.

Run from the repository root so that ``import src...`` resolves:

    python -m src.simulations --design ofat --replications 5 --out results/sim-results.csv

    # fast smoke test: baseline condition, one replication
    python -m src.simulations --design baseline --replications 1 --summarize

    # restrict the one-factor-at-a-time sweep to a few factors
    python -m src.simulations --design ofat --factors n_archetypes n_learners --limit 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.simulations.factors import BASELINE, FACTOR_LEVELS, ofat_design
from src.simulations.run import run_design, summarize_results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.simulations",
        description="Fit MoLA across simulation conditions and score parameter/predictive recovery.",
    )
    parser.add_argument(
        "--design",
        choices=("ofat", "baseline"),
        default="ofat",
        help="'ofat' sweeps one factor at a time around the baseline; "
        "'baseline' runs only the single baseline condition. (default: ofat)",
    )
    parser.add_argument(
        "--factors",
        nargs="+",
        metavar="FACTOR",
        choices=sorted(FACTOR_LEVELS),
        help="Restrict an 'ofat' design to these factors "
        f"(choices: {', '.join(sorted(FACTOR_LEVELS))}).",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=5,
        help="Monte Carlo replications per condition (default: 5).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Base RNG seed; replication seeds are derived from it (default: 0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run at most this many conditions (handy for a quick check).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the per-run results table to this CSV path. Unless "
        "--summary-out is given, the aggregated table is written alongside it "
        "as '<stem>_summary.csv'.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Write the mean/std summary (grouped by condition_id) to this CSV path.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Also print the mean/std summary grouped by condition_id.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.design == "baseline":
        conditions = [BASELINE]
    else:
        conditions = ofat_design(factors=args.factors)

    if args.limit is not None:
        conditions = conditions[: args.limit]

    n_runs = len(conditions) * args.replications
    print(
        f"Running {len(conditions)} condition(s) x {args.replications} replication(s) "
        f"= {n_runs} fit(s)...",
        file=sys.stderr,
    )

    results = run_design(
        conditions=conditions,
        n_replications=args.replications,
        base_seed=args.base_seed,
    )

    summary = summarize_results(results)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.out, index=False)
        print(f"Wrote {len(results)} per-run rows to {args.out}", file=sys.stderr)

    summary_out = args.summary_out
    if summary_out is None and args.out is not None:
        summary_out = args.out.with_name(f"{args.out.stem}_summary{args.out.suffix}")
    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_out, index=False)
        print(f"Wrote {len(summary)} summary rows to {summary_out}", file=sys.stderr)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        if args.summarize:
            print(summary.to_string(index=False))
        else:
            cols = [
                "condition_id",
                "seed",
                "mu_rmse",
                "theta_rmse",
                "assignment_ari",
                "holdout_auc",
            ]
            cols = [c for c in cols if c in results.columns]
            print(results[cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
