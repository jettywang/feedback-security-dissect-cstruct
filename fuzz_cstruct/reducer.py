"""Delta-debugging style reducer that shrinks a case while preserving its verdict.

Reduction operates on the structured :class:`~fuzz_cstruct.case_model.Case`, not on
raw bytes, so every candidate remains renderable. A candidate is accepted only when
the oracle returns the same :class:`~fuzz_cstruct.oracle.Verdict` *and* the rendered
definition got shorter. The reducer itself is bounded: each rule is tried a fixed
number of times, guaranteeing termination.

Two rules additionally protect the *root cause* of a failure: an alias or forward
declaration whose name is still referenced by a surviving field is never dropped, even
when dropping it would keep the same verdict for a different reason. This stops the
reducer from turning a meaningful minimal case into an unrelated trivial rejection.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .case_model import Case
from .oracle import run_case

# Simpler expressions tried in order; the first one preserving the verdict wins.
_EXPR_SIMPLIFICATIONS = [
    "1 << 1000000",
    "1 << n",
    "n",
    "1 / 0",
    "nope",
    "1",
]

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _rendered_len(case: Case) -> int:
    return len(case.render())


def _referenced_names(case: Case) -> set[str]:
    """All type and expression identifiers still used by the surviving fields."""
    names: set[str] = set()
    for type_name, _field_name, length in case.fields:
        names.update(_IDENT.findall(type_name))
        if length is not None:
            names.update(_IDENT.findall(length))
    return names


def _declared_name(line: str) -> str | None:
    """Extract the name introduced by a typedef / forward-declaration line."""
    match = _IDENT.findall(line)
    if not match:
        return None
    if line.lstrip().startswith("typedef"):
        # typedef <...> NAME; -> the last identifier is the alias name.
        return match[-1]
    if line.lstrip().startswith("struct"):
        # struct NAME; -> the identifier after the keyword.
        return match[1] if len(match) > 1 else None
    return None


def minimize(case: Case, max_passes: int = 4) -> Case:
    """Shrink ``case`` while preserving the oracle verdict. Always terminates."""
    target_verdict = run_case(case).verdict
    current = case
    baseline = _rendered_len(current)

    for _ in range(max_passes):
        improved = False
        referenced = _referenced_names(current)

        # Rule 1: drop alias lines one at a time, keeping referenced aliases.
        # Snapshot the tuple because a successful drop shrinks it mid-loop.
        for idx, alias in enumerate(tuple(current.aliases)):
            if _declared_name(alias) in referenced:
                continue
            candidate = replace(
                current,
                aliases=current.aliases[:idx] + current.aliases[idx + 1 :],
                history=(*current.history, f"reduce: drop alias #{idx}"),
            )
            if _rendered_len(candidate) < baseline and run_case(candidate).verdict is target_verdict:
                current, baseline, improved = candidate, _rendered_len(candidate), True

        # Rule 2: drop forward declarations one at a time, keeping referenced ones.
        referenced = _referenced_names(current)
        for idx, decl in enumerate(tuple(current.forward_decls)):
            if _declared_name(decl) in referenced:
                continue
            candidate = replace(
                current,
                forward_decls=current.forward_decls[:idx] + current.forward_decls[idx + 1 :],
                history=(*current.history, f"reduce: drop forward decl #{idx}"),
            )
            if _rendered_len(candidate) < baseline and run_case(candidate).verdict is target_verdict:
                current, baseline, improved = candidate, _rendered_len(candidate), True

        # Rule 3: simplify the array-length expression.
        simplified_fields = list(current.fields)
        for fidx, (type_name, field_name, length) in enumerate(simplified_fields):
            if length is None:
                continue
            for simpler in _EXPR_SIMPLIFICATIONS:
                if simpler == length:
                    continue
                trial_fields = list(simplified_fields)
                trial_fields[fidx] = (type_name, field_name, simpler)
                candidate = replace(
                    current,
                    fields=tuple(trial_fields),
                    history=(*current.history, f"reduce: expression {length!r} -> {simpler!r}"),
                )
                if _rendered_len(candidate) < baseline and run_case(candidate).verdict is target_verdict:
                    current, baseline, improved = candidate, _rendered_len(candidate), True
                    simplified_fields = list(current.fields)
                    break

        # Rule 4: remove non-essential fields, always keeping the first length field.
        # Iterate a snapshot of indices from the end, since drops shrink the tuple.
        for fidx in range(len(current.fields) - 1, 0, -1):
            if fidx >= len(current.fields):
                continue
            trial_fields = current.fields[:fidx] + current.fields[fidx + 1 :]
            candidate = replace(
                current,
                fields=trial_fields,
                history=(*current.history, f"reduce: drop field #{fidx}"),
            )
            if _rendered_len(candidate) < baseline and run_case(candidate).verdict is target_verdict:
                current, baseline, improved = candidate, _rendered_len(candidate), True

        if not improved:
            break

    return current
