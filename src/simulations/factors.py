"""Factorial design for MoLA simulation experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from itertools import product
from typing import Iterable, Iterator, Literal, Sequence

N_ARCHETYPES = (2, 4, 8, 16)
N_LEARNERS = (500, 2_000, 10_000, 50_000)
RESPONSES_PER_LEARNER = (10, 30, 100, 300)
N_SKILLS = (5, 20, 100)

Separation = Literal["low", "medium", "high"]
Imbalance = Literal["balanced", "moderately_imbalanced", "highly_imbalanced"]
Initialization = Literal["random", "k-means", "informed"]
Sampling = Literal["iid", "sparse"]
ItemCoverage = Literal["balanced", "uneven"]
ModelSpecification = Literal["mola_correct", "misspecified"]
ResponseFunction = Literal["compensatory", "noncompensatory"]

SEPARATION_LEVELS: tuple[Separation, ...] = ("low", "medium", "high")
IMBALANCE_LEVELS: tuple[Imbalance, ...] = (
    "balanced",
    "moderately_imbalanced",
    "highly_imbalanced",
)
INITIALIZATION_LEVELS: tuple[Initialization, ...] = ("random", "k-means", "informed")
SAMPLING_LEVELS: tuple[Sampling, ...] = ("iid", "sparse")
ITEM_COVERAGE_LEVELS: tuple[ItemCoverage, ...] = ("balanced", "uneven")
MODEL_SPEC_LEVELS: tuple[ModelSpecification, ...] = ("mola_correct", "misspecified")
RESPONSE_FN_LEVELS: tuple[ResponseFunction, ...] = ("compensatory", "noncompensatory")

# Maps categorical separation to half-width around 0.5 on the proficiency scale.
SEPARATION_DELTA: dict[Separation, float] = {
    "low": 0.10,
    "medium": 0.25,
    "high": 0.42,
}

# Zipf exponents for mixture weights (0 => uniform).
IMBALANCE_ZIPF: dict[Imbalance, float] = {
    "balanced": 0.0,
    "moderately_imbalanced": 0.7,
    "highly_imbalanced": 1.8,
}

Q_FLIP_PROB = 0.10


@dataclass(frozen=True)
class SimulationCondition:
    """One cell of the simulation design."""

    n_archetypes: int = 4
    n_learners: int = 2_000
    responses_per_learner: int = 30
    archetype_separation: Separation = "medium"
    archetype_imbalance: Imbalance = "balanced"
    initialization: Initialization = "random"
    sampling: Sampling = "iid"
    item_coverage: ItemCoverage = "balanced"
    model_specification: ModelSpecification = "mola_correct"
    response_function: ResponseFunction = "compensatory"
    n_skills: int = 20
    n_items: int | None = None
    seed: int = 0
    n_iter: int = 50
    holdout_frac: float = 0.2
    fit_kwargs: dict = field(default_factory=dict)

    def resolved_n_items(self) -> int:
        if self.n_items is not None:
            return self.n_items
        return int(max(2 * self.responses_per_learner, 3 * self.n_skills, 40))

    def condition_id(self) -> str:
        return (
            f"M{self.n_archetypes}"
            f"_N{self.n_learners}"
            f"_R{self.responses_per_learner}"
            f"_K{self.n_skills}"
            f"_sep-{self.archetype_separation}"
            f"_imb-{self.archetype_imbalance}"
            f"_init-{self.initialization}"
            f"_samp-{self.sampling}"
            f"_cov-{self.item_coverage}"
            f"_spec-{self.model_specification}"
            f"_resp-{self.response_function}"
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["n_items"] = self.resolved_n_items()
        payload["condition_id"] = self.condition_id()
        return payload


BASELINE = SimulationCondition()

FACTOR_LEVELS: dict[str, Sequence] = {
    "n_archetypes": N_ARCHETYPES,
    "n_learners": N_LEARNERS,
    "responses_per_learner": RESPONSES_PER_LEARNER,
    "archetype_separation": SEPARATION_LEVELS,
    "archetype_imbalance": IMBALANCE_LEVELS,
    "initialization": INITIALIZATION_LEVELS,
    "sampling": SAMPLING_LEVELS,
    "item_coverage": ITEM_COVERAGE_LEVELS,
    "model_specification": MODEL_SPEC_LEVELS,
    "response_function": RESPONSE_FN_LEVELS,
    "n_skills": N_SKILLS,
}


def ofat_design(
    baseline: SimulationCondition = BASELINE,
    factors: Iterable[str] | None = None,
) -> list[SimulationCondition]:
    """One-factor-at-a-time grid around a baseline (includes the baseline once)."""
    names = list(factors) if factors is not None else list(FACTOR_LEVELS)
    seen: set[str] = set()
    conditions: list[SimulationCondition] = []

    def _add(condition: SimulationCondition) -> None:
        key = condition.condition_id()
        if key not in seen:
            seen.add(key)
            conditions.append(condition)

    _add(baseline)
    for name in names:
        if name not in FACTOR_LEVELS:
            raise KeyError(f"Unknown factor {name!r}. Valid: {sorted(FACTOR_LEVELS)}")
        for level in FACTOR_LEVELS[name]:
            _add(replace(baseline, **{name: level}))
    return conditions


def factorial_design(
    factors: dict[str, Sequence] | None = None,
    baseline: SimulationCondition = BASELINE,
) -> Iterator[SimulationCondition]:
    """Cartesian product over the given factor levels; omitted factors stay at baseline.

    Passing no `factors` yields the full 11-way design (~8e4 cells) and is
    intended only for subsetting, e.g. ``{"n_archetypes": N_ARCHETYPES, "n_learners": N_LEARNERS}``.
    """
    spec = factors if factors is not None else FACTOR_LEVELS
    names = list(spec)
    for unknown in names:
        if unknown not in FACTOR_LEVELS:
            raise KeyError(f"Unknown factor {unknown!r}.")
    for combo in product(*(spec[name] for name in names)):
        yield replace(baseline, **dict(zip(names, combo)))


def replicate_seeds(base_seed: int, n_replications: int) -> list[int]:
    return [base_seed + 10_003 * r for r in range(n_replications)]
