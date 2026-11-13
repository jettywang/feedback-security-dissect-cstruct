"""Structured representation of a fuzzing case and grammar-aware mutators.

A case is *not* a blob. It is a small, partially legal C-style definition assembled
from four stages that map directly onto the way ``dissect.cstruct`` consumes input:

1. alias resolution      - ``typedef`` / type-name aliases;
2. forward type reference - named structs pre-registered before their body;
3. array length expression - the expression inside ``field[...]``;
4. compile fallback      - a field the compiler cannot inline (LEB128 / custom type),
                           forcing the interpreted reader.

Mutators replace a single structured slot with another value of the same syntactic
kind, so the result stays lexable/parseable at the grammar level even when it becomes
semantically invalid (undefined name, cyclic dependency, extreme shift, ...). Pure
byte/character flipping is deliberately avoided: it mostly produces lexer noise and
can never reach the definition-parse x binary-parse interaction we are after.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

# Names known to the default cstruct instance, reused so most type slots stay legal.
KNOWN_TYPES = ["uint8", "uint16", "uint32", "uint64", "int32", "char", "wchar"]
# Names that are intentionally *not* defined, to exercise unresolved references.
GHOST_NAMES = ["nope", "ghost", "unborn", "missing_len"]
# Operators legal inside cstruct array-length expressions.
BIN_OPS = ["+", "-", "*", "/", "%", "<<", ">>", "|", "&", "^"]
# Expressions that stay syntactically valid but stress resources or semantics.
EXPR_LIBRARY = [
    "1",
    "2",
    "4",
    "n",
    "n + 1",
    "n - 100",
    "n * 2",
    "1 << n",
    "1 >> n",
    "n % 3",
    "1 / 0",
    "sizeof(S)",
    "n | 3",
    "nope",
    "n + ghost",
    "1 << 1000000",
    "EOF",
]


@dataclass(frozen=True)
class Case:
    """A structured, partially legal definition.

    Attributes:
        stage: Which of the four stages this case primarily exercises (1..4).
        aliases: ``typedef`` lines emitted first, so alias resolution runs first.
        forward_decls: Named struct headers pre-registered before any body.
        struct_name: Name of the root structure to read back.
        fields: ``(type, name, length_expr_or_None)`` tuples for the root struct.
        fallback_type: Optional field type forcing the compiler fallback (stage 4).
        cyclic: When true, the root struct references itself by value.
        seed: RNG seed making the generated payload reproducible.
        payload_len: Number of bytes offered to the reader.
        history: Human-readable record of the mutations that produced the case.
    """

    stage: int
    aliases: tuple[str, ...] = ()
    forward_decls: tuple[str, ...] = ()
    extra_defs: tuple[str, ...] = ()
    struct_name: str = "S"
    fields: tuple[tuple[str, str, str | None], ...] = (("uint32", "n", None),)
    fallback_type: str | None = None
    cyclic: bool = False
    seed: int = 0
    payload_len: int = 64
    payload_prefix: bytes = b""
    history: tuple[str, ...] = field(default_factory=tuple)

    def render(self) -> str:
        """Render the case back into a C-style definition string."""
        lines: list[str] = ["#define SEED 1"]
        lines.extend(self.aliases)
        lines.extend(self.forward_decls)
        lines.extend(self.extra_defs)

        body: list[str] = []
        for type_name, field_name, length in self.fields:
            decl = f"    {type_name} {field_name}"
            if length is not None:
                decl += f"[{length}]"
            body.append(decl + ";")

        if self.fallback_type is not None:
            body.append(f"    {self.fallback_type} dyn;")
        if self.cyclic:
            body.append(f"    {self.struct_name} next;")

        lines.append(f"struct {self.struct_name} {{")
        lines.extend(body)
        lines.append("};")
        return "\n".join(lines)

    def with_(self, note: str, **changes) -> "Case":
        """Return a mutated copy, appending a human-readable mutation note."""
        return replace(self, history=(*self.history, note), **changes)


def seed_case(stage: int, rng: random.Random) -> Case:
    """Build a baseline legal case for the given stage.

    The payload prefix pins leading length fields to small values so the legal
    baselines stay deterministic and never accidentally trip the resource guards.
    """
    small_len = (2).to_bytes(4, "little")
    if stage == 1:
        return Case(
            stage=1,
            aliases=("typedef uint32 mylen;",),
            fields=(("mylen", "n", None), ("uint8", "data", "n")),
            payload_prefix=small_len,
            payload_len=8,
            history=("seed: legal typedef alias driving an array length",),
        )
    if stage == 2:
        return Case(
            stage=2,
            forward_decls=("struct node;",),
            struct_name="node",
            fields=(("uint32", "x", None), ("uint16", "y", None)),
            payload_len=8,
            history=("seed: forward-declared named struct, filled in later",),
        )
    if stage == 3:
        return Case(
            stage=3,
            fields=(("uint32", "n", None), ("uint8", "data", "1 << n")),
            payload_prefix=(1).to_bytes(4, "little"),
            payload_len=8,
            history=("seed: legal shift-based dynamic array length",),
        )
    return Case(
        stage=4,
        fields=(("uint16", "a", None),),
        fallback_type="uleb128",
        payload_len=8,
        history=("seed: LEB128 field forces the interpreted compiler fallback",),
    )


# --------------------------------------------------------------------------- #
# Mutators, grouped by stage. Each keeps the surrounding grammar intact.
# --------------------------------------------------------------------------- #


def mutate_aliases(case: Case, rng: random.Random) -> Case:
    choice = rng.randrange(4)
    if choice == 0:
        # Alias a name that is never defined -> unresolved type at use site.
        ghost = rng.choice(GHOST_NAMES)
        return case.with_(f"alias to undefined type {ghost!r}", aliases=(f"typedef {ghost} mylen;",))
    if choice == 1:
        # Two aliases pointing at each other -> cyclic alias resolution.
        return case.with_(
            "cyclic alias pair A<->B",
            aliases=("typedef A B;", "typedef B A;"),
        )
    if choice == 2:
        # Alias chain ending in a ghost, still valid grammar.
        return case.with_(
            "alias chain ending at undefined type",
            aliases=("typedef uint32 t1;", "typedef t1 t2;", "typedef ghost t3;"),
        )
    # Re-point a legal alias at a different built-in width.
    return case.with_("repoint alias at another builtin", aliases=("typedef uint64 mylen;",))


def mutate_forward_refs(case: Case, rng: random.Random) -> Case:
    choice = rng.randrange(3)
    if choice == 0:
        # Reference a struct name that is never completed.
        return case.with_(
            "field uses never-completed forward struct",
            forward_decls=("struct phantom;",),
            struct_name="holder",
            fields=(("uint32", "x", None), ("phantom", "p", None)),
            cyclic=False,
        )
    if choice == 1:
        # Self reference by value -> infinite nesting, grammar is legal.
        return case.with_("self-referential field by value", cyclic=True, struct_name="node")
    # Mutual reference through a forward header.
    return case.with_(
        "mutual value reference between two structs",
        forward_decls=("struct B;",),
        struct_name="A",
        fields=(("uint32", "x", None), ("B", "b", None)),
        cyclic=True,
    )


def mutate_expression(case: Case, rng: random.Random) -> Case:
    expr = rng.choice(EXPR_LIBRARY)
    base_fields = (("uint32", "n", None), ("uint8", "data", expr))
    return case.with_(f"swap array length expression to {expr!r}", fields=base_fields)


def mutate_fallback(case: Case, rng: random.Random) -> Case:
    choice = rng.randrange(3)
    if choice == 0:
        return case.with_("use signed LEB128 fallback type", fallback_type="ileb128")
    if choice == 1:
        # Fallback type feeding an EOF-terminated array.
        return case.with_(
            "LEB128 fallback plus EOF array",
            fallback_type=None,
            fields=(("uint16", "a", None), ("uleb128", "rest", "EOF")),
        )
    return case.with_("nested dynamic struct that blocks inlining", fallback_type="uleb128")


def mutate_nested_packing(case: Case, rng: random.Random) -> Case:
    """Emit a nested *static* struct read over a truncated stream.

    The compiled reader merges the nested static struct into a single bulk unpack,
    while the interpreted reader reads it field by field. A short stream therefore
    produces the same ``EOFError`` family but a different expected byte count - a real
    cross-mode semantic difference, only reachable by combining a specific definition
    with a specific (short) binary payload rather than random bytes.
    """
    return case.with_(
        "nested static struct over truncated stream (packed vs field-wise EOF)",
        stage=4,
        struct_name="outer",
        aliases=(),
        forward_decls=(),
        extra_defs=("struct inner { uint32 a; uint32 b; };",),
        fallback_type=None,
        cyclic=False,
        fields=(
            ("inner", "x", None),
            ("uint32", "c", None),
        ),
        payload_len=5,
    )


STAGE_MUTATORS = {
    1: mutate_aliases,
    2: mutate_forward_refs,
    3: mutate_expression,
    4: mutate_fallback,
}


def mutate(case: Case, rng: random.Random) -> Case:
    """Apply one stage-aware mutation, occasionally crossing to another stage.

    A small fraction of mutations injects the nested-packing differential case so the
    campaign can exercise genuine compiled-vs-interpreted divergence.
    """
    if rng.random() < 0.15:
        return mutate_nested_packing(case, rng)
    stage = case.stage if rng.random() < 0.8 else rng.randrange(1, 5)
    return STAGE_MUTATORS[stage](replace(case, stage=stage), rng)


def payload_for(case: Case) -> bytes:
    """Deterministic, intentionally small payload.

    The stream itself is never the fuzz target: a fixed short buffer plus
    :class:`~fuzz_cstruct.guards.CappedStream` is enough, because any oversized read
    driven by the *definition* trips the read cap rather than consuming real memory.

    ``payload_prefix`` pins leading bytes (e.g. a length field) so hand-written minimal
    samples trigger a specific count deterministically; the remaining bytes come from a
    seedable RNG and the whole buffer never exceeds ``payload_len``.
    """
    rng = random.Random(case.seed & 0xFFFF)
    prefix = case.payload_prefix
    filler = bytes(rng.randrange(256) for _ in range(max(0, case.payload_len - len(prefix))))
    return (prefix + filler)[: case.payload_len]
