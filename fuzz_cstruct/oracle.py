"""Differential oracle for ``dissect.cstruct``.

The same structured case is executed twice on fresh ``cstruct`` instances:

* ``compiled=True``  - the default path that generates and compiles Python source,
  transparently falling back to the interpreted reader when codegen is unsupported;
* ``compiled=False`` - the pure interpreted reader.

Each phase (definition ``load``, then binary ``read``) is observed independently. The
oracle never decides "this is a bug" from the presence of an exception alone. Instead
it compares the two executions and classifies every outcome into one of four buckets:

* ``ok``       - both modes parse and read successfully with an equal value digest;
* ``illegal``  - both modes reject the input with the same exception family at the
                 same phase: the input is simply invalid, not a library bug;
* ``diverge``  - the two execution modes disagree (different phase, exception family,
                 or value digest): the definition-parse x binary-parse combination
                 exposed an implementation inconsistency;
* ``dos``      - a guard tripped (time, read cap, memory, recursion), which is a real
                 resource-exhaustion risk regardless of which mode triggered it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from dissect.cstruct import cstruct

from .case_model import Case, payload_for
from .guards import (
    DOS_EXCEPTIONS,
    CappedStream,
    recursion_ceiling,
    time_budget,
)

# Wall-clock and read budgets. Kept deliberately small so the demo always terminates.
DEFAULT_TIME_BUDGET = 2.0
DEFAULT_READ_CAP = 1 * 1024 * 1024  # 1 MiB: far above any legitimate demo payload
DEFAULT_RECURSION = 300


class Verdict(str, Enum):
    OK = "ok"
    ILLEGAL = "illegal"
    DIVERGE = "diverge"
    DOS = "dos"


@dataclass(frozen=True)
class PhaseResult:
    """Outcome of one phase in one mode."""

    phase: str  # "load" or "read"
    kind: str  # "ok" | "error" | "dos"
    exception: str | None
    message: str | None
    digest: str | None

    @property
    def family(self) -> str:
        """Coarse exception family used for cross-mode comparison.

        Messages carry concrete numbers (array sizes, offsets) that legitimately
        differ, so comparisons use the exception class plus the phase, not the text.
        """
        return self.exception or "ok"

    @property
    def signature(self) -> tuple[str, str, str | None]:
        """Semantic signature of a failure: ``(phase, exception, normalized-detail)``.

        The normalized detail keeps semantics a caller may branch on while dropping
        incidental values. In particular, ``EOFError`` messages embed the number of
        expected bytes: the compiled reader inlines nested static structs into one bulk
        ``unpack`` (it reports the whole block), while the interpreted reader reads field
        by field (it reports the first missing field). Both raise ``EOFError`` at read
        time, but the expected byte count differs - a genuine behavioral divergence that
        must not be flattened into an "illegal" verdict, so the expected-size token is
        retained.
        """
        normalized = None
        if self.kind == "error" and self.message is not None:
            normalized = _normalize_message(self.exception, self.message)
        return (self.phase, self.exception or "ok", normalized)


def _normalize_message(exception: str | None, message: str) -> str:
    """Keep error semantics that matter, discard incidental concrete integers."""
    if exception == "EOFError":
        if "expected" in message:
            return f"EOFError(expected {message.split('expected', 1)[1].strip()})"
        return "EOFError"
    return exception or "error"


@dataclass(frozen=True)
class TrialResult:
    case: Case
    compiled: PhaseResult
    interpreted: PhaseResult
    verdict: Verdict
    reason: str

    def label(self) -> str:
        return f"[{self.verdict.value.upper():7s}] {self.reason}"


def _safe_digest(obj) -> str:
    """Best-effort structural digest that does not allocate huge containers."""
    try:
        return f"{type(obj).__name__}:{obj!r}"[:200]
    except BaseException as exc:  # repr itself can recurse / blow up
        return f"<unreprable {type(obj).__name__}:{type(exc).__name__}>"


def _run_mode(case: Case, compiled: bool) -> PhaseResult:
    """Execute one full load+read trial in a single mode, inside all guards."""
    definition = case.render()
    data = payload_for(case)
    phase = "load"
    try:
        with recursion_ceiling(DEFAULT_RECURSION), time_budget(DEFAULT_TIME_BUDGET):
            cs = cstruct()
            try:
                cs.load(definition, compiled=compiled)
            except DOS_EXCEPTIONS:
                raise
            except Exception as exc:
                return PhaseResult("load", "error", type(exc).__name__, str(exc)[:120], None)

            phase = "read"
            root = cs.resolve(case.struct_name)
            stream = CappedStream(data, DEFAULT_READ_CAP)
            try:
                value = root.read(stream)
            except DOS_EXCEPTIONS:
                raise
            except Exception as exc:
                return PhaseResult("read", "error", type(exc).__name__, str(exc)[:120], None)

            return PhaseResult("read", "ok", None, None, _safe_digest(value))
    except DOS_EXCEPTIONS as exc:
        return PhaseResult(phase, "dos", type(exc).__name__, str(exc)[:120], None)
    except RecursionError as exc:
        return PhaseResult(phase, "dos", "RecursionError", str(exc)[:120], None)


def _classify(case: Case, a: PhaseResult, b: PhaseResult) -> TrialResult:
    dos_side = next((r for r in (a, b) if r.kind == "dos"), None)
    if dos_side is not None:
        mode = "compiled" if a.kind == "dos" else "interpreted"
        return TrialResult(
            case,
            a,
            b,
            Verdict.DOS,
            f"{dos_side.exception} at {dos_side.phase} ({mode}): {dos_side.message}",
        )

    if a.kind == "ok" and b.kind == "ok":
        if a.digest == b.digest:
            return TrialResult(case, a, b, Verdict.OK, "equal values in both modes")
        return TrialResult(
            case, a, b, Verdict.DIVERGE, f"value mismatch: {a.digest!r} != {b.digest!r}"
        )

    if a.kind == "error" and b.kind == "error":
        if a.signature == b.signature:
            return TrialResult(
                case,
                a,
                b,
                Verdict.ILLEGAL,
                f"both reject at {a.phase} with {a.family}: input is invalid",
            )
        return TrialResult(
            case,
            a,
            b,
            Verdict.DIVERGE,
            f"error semantics differ: compiled {a.signature} vs interpreted {b.signature}",
        )

    # One side succeeded, the other raised.
    good, bad = (a, b) if a.kind == "ok" else (b, a)
    good_mode = "compiled" if a.kind == "ok" else "interpreted"
    return TrialResult(
        case,
        a,
        b,
        Verdict.DIVERGE,
        f"{good_mode} succeeded ({good.digest!r}), other raised {bad.family} at {bad.phase}",
    )


def run_case(case: Case) -> TrialResult:
    """Execute a case in both modes and return the differential verdict."""
    compiled = _run_mode(case, True)
    interpreted = _run_mode(case, False)
    return _classify(case, compiled, interpreted)
