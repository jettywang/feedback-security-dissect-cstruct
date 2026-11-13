"""Deterministic, budget-bounded corpus generator / fuzzing loop.

The generator walks the four stages in order and keeps each individual case partially
legal by mutating one structured slot at a time. It is deliberately *not* coverage
guided and does not feed random bytes to ``read()``: the binary stream is a fixed,
capped buffer, while the structured definition carries the fuzzing entropy. This is
what makes definition-parse x binary-parse interaction bugs reachable.

The whole campaign always terminates after a fixed number of iterations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from .case_model import Case, mutate, seed_case
from .oracle import TrialResult, Verdict, run_case

DEFAULT_ITERATIONS = 120
STAGES = (1, 2, 3, 4)


@dataclass
class Campaign:
    iterations: int
    results: list[TrialResult]

    def by_verdict(self, verdict: Verdict) -> list[TrialResult]:
        return [r for r in self.results if r.verdict is verdict]

    def summary(self) -> dict[str, int]:
        counts = {v.value: 0 for v in Verdict}
        for r in self.results:
            counts[r.verdict.value] += 1
        return counts


def generate_campaign(iterations: int = DEFAULT_ITERATIONS, seed: int = 1337) -> Campaign:
    """Run a fixed-length campaign, cycling through the four associated stages."""
    rng = random.Random(seed)
    results: list[TrialResult] = []

    for i in range(iterations):
        stage = STAGES[i % len(STAGES)]
        case = seed_case(stage, rng)
        if i >= len(STAGES):
            # First pass keeps one clean legal baseline per stage; later cases mutate.
            for _ in range(rng.randrange(1, 4)):
                case = mutate(case, rng)

        case = replace(case, seed=seed + i)
        results.append(run_case(case))

    return Campaign(iterations, results)


def fuzz_one(case: Case) -> TrialResult:
    """Public helper used by the reducer and by ad-hoc experiments."""
    return run_case(case)
