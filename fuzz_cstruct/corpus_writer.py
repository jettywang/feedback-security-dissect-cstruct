"""Persist the seed corpus and reduced findings to ``fuzz_cstruct/corpus``.

Every corpus entry is a readable text file with three sections: the rendered C-style
definition, the hex payload offered to the reader, and the expected verdict. Keeping
the corpus human-readable means a triager can replay or hand-edit any entry without
the generator.
"""

from __future__ import annotations

from pathlib import Path

from .case_model import payload_for
from .fuzzer import generate_campaign
from .oracle import Verdict, run_case
from .reducer import minimize
from .samples import MINIMAL_SAMPLES

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"


def _serialize(case, verdict: Verdict, reason: str) -> str:
    payload = payload_for(case)
    lines = [
        f"# verdict: {verdict.value}",
        f"# reason: {reason}",
        f"# stage: {case.stage}",
        f"# payload_hex: {payload.hex()}",
        "# --- definition ---",
        case.render(),
    ]
    return "\n".join(lines) + "\n"


def write_minimal_samples() -> list[Path]:
    paths = []
    for name, case in MINIMAL_SAMPLES.items():
        result = run_case(case)
        path = CORPUS_DIR / f"minimal_{name}.txt"
        path.write_text(_serialize(case, result.verdict, result.reason))
        paths.append(path)
    return paths


def write_campaign_findings(iterations: int = 120, seed: int = 1337) -> list[Path]:
    """Run the bounded campaign, minimize each non-OK finding, and write unique ones."""
    campaign = generate_campaign(iterations, seed)
    seen_signatures: set[tuple] = set()
    paths: list[Path] = []

    for index, result in enumerate(campaign.results):
        if result.verdict is Verdict.OK:
            continue
        signature = (result.verdict, result.compiled.signature, result.interpreted.signature)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        reduced = minimize(result.case)
        reduced_result = run_case(reduced)
        path = CORPUS_DIR / f"finding_{index:03d}_{reduced_result.verdict.value}.txt"
        path.write_text(_serialize(reduced, reduced_result.verdict, reduced_result.reason))
        paths.append(path)

    return paths


def main(iterations: int = 120) -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    minimal = write_minimal_samples()
    findings = write_campaign_findings(iterations)
    print(f"wrote {len(minimal)} minimal samples and {len(findings)} unique reduced findings")
    print(f"corpus directory: {CORPUS_DIR}")


if __name__ == "__main__":
    main()
