"""Simulation analysis utilities for MoLA.

The full 11-factor cross is ~8e4 cells; start from :func:`ofat_design` or a
restricted :func:`factorial_design`.
"""

from src.simulations.dgp import SimulatedDataset, generate_dataset
from src.simulations.factors import (
    BASELINE,
    FACTOR_LEVELS,
    IMBALANCE_LEVELS,
    INITIALIZATION_LEVELS,
    ITEM_COVERAGE_LEVELS,
    MODEL_SPEC_LEVELS,
    N_ARCHETYPES,
    N_LEARNERS,
    N_SKILLS,
    RESPONSES_PER_LEARNER,
    RESPONSE_FN_LEVELS,
    SAMPLING_LEVELS,
    SEPARATION_LEVELS,
    SimulationCondition,
    factorial_design,
    ofat_design,
    replicate_seeds,
)
from src.simulations.metrics import align_components
from src.simulations.run import (
    MoLAWithInit,
    SimulationResult,
    run_condition,
    run_design,
    summarize_results,
)

__all__ = [
    "BASELINE",
    "FACTOR_LEVELS",
    "IMBALANCE_LEVELS",
    "INITIALIZATION_LEVELS",
    "ITEM_COVERAGE_LEVELS",
    "MODEL_SPEC_LEVELS",
    "N_ARCHETYPES",
    "N_LEARNERS",
    "N_SKILLS",
    "RESPONSES_PER_LEARNER",
    "RESPONSE_FN_LEVELS",
    "SAMPLING_LEVELS",
    "SEPARATION_LEVELS",
    "MoLAWithInit",
    "SimulatedDataset",
    "SimulationCondition",
    "SimulationResult",
    "align_components",
    "factorial_design",
    "generate_dataset",
    "ofat_design",
    "replicate_seeds",
    "run_condition",
    "run_design",
    "summarize_results",
]
