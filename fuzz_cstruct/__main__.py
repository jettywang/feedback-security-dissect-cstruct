"""Command line entry point for the bounded dissect.cstruct fuzzing demo.

Examples:
    python -m fuzz_cstruct                 # fixed 120-iteration campaign
    python -m fuzz_cstruct -n 40           # shorter, still bounded campaign
    python -m fuzz_cstruct --findings      # only print non-OK cases
"""

from __future__ import annotations

import argparse
import sys

from .fuzzer import DEFAULT_ITERATIONS, generate_campaign
from .oracle import Verdict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded differential fuzzer for dissect.cstruct")
    parser.add_argument("-n", "--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--findings", action="store_true", help="hide OK/legal cases")
    args = parser.parse_args(argv)

    campaign = generate_campaign(args.iterations, args.seed)

    shown = 0
    for index, result in enumerate(campaign.results):
        if args.findings and result.verdict in (Verdict.OK, Verdict.ILLEGAL):
            continue
        shown += 1
        case = result.case
        print(f"#{index:03d} stage{case.stage} {result.label()}")
        for note in case.history:
            print(f"        - {note}")

    counts = campaign.summary()
    print("\n=== summary ===")
    for verdict in Verdict:
        print(f"{verdict.value:8s}: {counts[verdict.value]}")
    print(f"shown    : {shown} / {args.iterations}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
