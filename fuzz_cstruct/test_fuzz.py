"""Contract tests for the fuzzing harness itself.

These test the fuzzer machinery and its classification guarantees, not dissect.cstruct.
Run with: ``python -m pytest fuzz_cstruct/test_fuzz.py`` or directly as a script.
"""

from __future__ import annotations

import time

from fuzz_cstruct.case_model import Case, mutate, payload_for, seed_case
from fuzz_cstruct.fuzzer import generate_campaign
from fuzz_cstruct.guards import (
    DoSBudgetExceeded,
    StreamBudgetExceeded,
    CappedStream,
    recursion_ceiling,
    time_budget,
)
from fuzz_cstruct.oracle import Verdict, run_case
from fuzz_cstruct.reducer import minimize
from fuzz_cstruct.samples import MINIMAL_SAMPLES


def test_guards_time_budget_terminates():
    start = time.monotonic()
    try:
        with time_budget(0.2):
            while True:
                pass
    except DoSBudgetExceeded:
        pass
    assert time.monotonic() - start < 1.0


def test_capped_stream_bounds_reads():
    stream = CappedStream(b"abc", read_cap=10)
    assert stream.read(2) == b"ab"
    try:
        stream.read(1_000_000)
    except StreamBudgetExceeded:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("read cap not enforced")
    assert stream.max_requested >= 1_000_000


def test_recursion_ceiling_restored():
    import sys

    original = sys.getrecursionlimit()
    with recursion_ceiling(120):
        assert sys.getrecursionlimit() == 120
    assert sys.getrecursionlimit() == original


def test_seed_cases_are_legal():
    import random

    rng = random.Random(0)
    for stage in (1, 2, 3, 4):
        result = run_case(seed_case(stage, rng))
        assert result.verdict in (Verdict.OK, Verdict.ILLEGAL), (stage, result.label())


def test_minimal_samples_have_expected_verdicts():
    assert run_case(MINIMAL_SAMPLES["illegal"]).verdict is Verdict.ILLEGAL
    assert run_case(MINIMAL_SAMPLES["diverge"]).verdict is Verdict.DIVERGE
    assert run_case(MINIMAL_SAMPLES["dos"]).verdict is Verdict.DOS


def test_divergence_is_efofamily_but_different_semantics():
    result = run_case(MINIMAL_SAMPLES["diverge"])
    assert result.compiled.exception == "EOFError"
    assert result.interpreted.exception == "EOFError"
    assert result.compiled.signature != result.interpreted.signature


def test_dos_is_caught_in_both_modes_under_budget():
    result = run_case(MINIMAL_SAMPLES["dos"])
    assert result.compiled.kind == "dos"
    assert result.interpreted.kind == "dos"


def test_mutations_keep_renderable_definitions():
    import random

    rng = random.Random(42)
    case = seed_case(3, rng)
    for _ in range(25):
        case = mutate(case, rng)
        rendered = case.render()
        assert "struct" in rendered and rendered.count("{") == rendered.count("}")
        assert len(payload_for(case)) == case.payload_len


def test_reducer_preserves_verdict_and_shrinks():
    bloated = Case(
        stage=3,
        aliases=("typedef uint32 t1;", "typedef uint16 t2;"),
        fields=(
            ("uint32", "n", None),
            ("uint16", "extra", None),
            ("uint8", "data", "nope"),
        ),
        seed=3,
    )
    original_verdict = run_case(bloated).verdict
    reduced = minimize(bloated)
    assert run_case(reduced).verdict is original_verdict
    assert len(reduced.render()) < len(bloated.render())


def test_campaign_is_bounded_and_exercises_all_verdicts():
    campaign = generate_campaign(iterations=120, seed=1337)
    assert len(campaign.results) == 120
    counts = campaign.summary()
    # The seeded campaign deterministically reaches every classification bucket.
    for verdict in Verdict:
        assert counts[verdict.value] > 0, counts


def _run_all():
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"PASS {name}")


if __name__ == "__main__":
    _run_all()
    print("all fuzz harness contract tests passed")
