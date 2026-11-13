"""Three readable minimal samples, one per non-OK failure class.

Each sample is a hand-verified, minimal :class:`~fuzz_cstruct.case_model.Case`. They
double as living documentation of how the oracle distinguishes:

* :data:`SAMPLE_ILLEGAL` - an input that is simply invalid; both execution modes reject
  it identically, so it is *not* a library bug;
* :data:`SAMPLE_DIVERGE` - a legal definition whose binary read fails differently in
  compiled vs interpreted mode (same exception family, different semantics);
* :data:`SAMPLE_DOS` - a small, grammar-legal definition that drives an unbounded
  allocation via an attacker-controlled shift, caught by the resource guards.
"""

from __future__ import annotations

from .case_model import Case

# 1) Illegal input: the array length names a field that does not exist. The grammar is
#    fine (it lexes and parses), but resolution fails identically in both modes.
SAMPLE_ILLEGAL = Case(
    stage=3,
    struct_name="S",
    fields=(
        ("uint32", "n", None),
        ("uint8", "data", "nope"),
    ),
    payload_prefix=(1).to_bytes(4, "little"),
    payload_len=8,
    history=("minimal illegal: undefined name 'nope' in array length",),
)

# 2) Mode divergence: a nested static struct inlined into one bulk unpack by the
#    compiler, but read field-by-field by the interpreter. A 5-byte truncated stream
#    yields EOFError(expected 12/8) vs EOFError(expected 4).
SAMPLE_DIVERGE = Case(
    stage=4,
    struct_name="outer",
    extra_defs=("struct inner { uint32 a; uint32 b; };",),
    fields=(
        ("inner", "x", None),
        ("uint32", "c", None),
    ),
    payload_len=5,
    history=(
        "minimal divergence: packed bulk read vs field-wise read on truncated stream",
    ),
)

# 3) Real DoS risk: n is read from the (small) stream and 1 << n requests a gigantic
#    array. Guards convert the resource attempt into a bounded, reproducible verdict.
SAMPLE_DOS = Case(
    stage=3,
    struct_name="S",
    fields=(
        ("uint32", "n", None),
        ("uint8", "data", "1 << n"),
    ),
    # Pin n = 60 -> 1 << 60 bytes requested, far beyond the 1 MiB read cap.
    payload_prefix=(60).to_bytes(4, "little"),
    payload_len=8,
    history=("minimal dos: attacker-controlled 1 << n array length",),
)

MINIMAL_SAMPLES = {
    "illegal": SAMPLE_ILLEGAL,
    "diverge": SAMPLE_DIVERGE,
    "dos": SAMPLE_DOS,
}
