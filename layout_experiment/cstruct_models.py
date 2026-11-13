"""The same logical definition modelled three ways inside dissect.cstruct.

Every model is built for an *explicit* target pointer width (``uint32`` for
32-bit firmware, ``uint64`` for 64-bit firmware). The host's default pointer
type is never relied upon - see ``repro_misalignment.py`` for what happens
when it is.

Policies
--------
packed:   loaded with ``align=False`` (the cstruct default). Fields get no
          padding, so the struct is the raw sum of field widths.
natural:  loaded with ``align=True``. cstruct applies the target ABI's
          natural alignment, including struct tail padding.
explicit: still ``align=False`` (packed), but the definition itself contains
          ``char padNN[n]`` reserve fields, so the bytes land exactly where
          the natural ABI would put them without relying on the aligner.
"""
from __future__ import annotations

from dataclasses import dataclass

from dissect.cstruct import cstruct

from abi_layout import ABI, ABI_ILP32, ABI_LP64

POLICIES = ("packed", "natural", "explicit")


def _definition(abi: ABI, policy: str) -> str:
    ptr = "uint32" if abi.pointer_size == 4 else "uint64"
    align = policy == "natural"
    # The explicit model is a packed struct that carries its own reserve bytes.
    if policy == "explicit" and abi.pointer_size == 8:
        rec = f"""
        struct Rec {{
            uint32 magic;
            char   pad04[4];   /* align 'next' to 8 */
            {ptr}  next;
            union {{ uint32 type; struct Inner info; uint32 raw; }};
            uint16 count;
            char   pad16[2];   /* tail pad to struct alignment 8 */
        }};
        """
    elif policy == "explicit":  # ILP32: only the 2-byte tail pad is needed
        rec = f"""
        struct Rec {{
            uint32 magic;
            {ptr}  next;
            union {{ uint32 type; struct Inner info; uint32 raw; }};
            uint16 count;
            char   pad0e[2];   /* tail pad to struct alignment 4 */
        }};
        """
    else:
        rec = f"""
        struct Rec {{
            uint32 magic;
            {ptr}  next;
            union {{ uint32 type; struct Inner info; uint32 raw; }};
            uint16 count;
        }};
        """
    return f"""
    struct Inner {{
        uint16 id;
        uint16 flags;
    }};
    {rec}
    struct Block {{
        Rec item[2];
    }};
    """, align


@dataclass(frozen=True)
class Model:
    abi: ABI
    policy: str
    cs: cstruct

    @property
    def rec(self):
        return self.cs.Rec

    @property
    def block(self):
        return self.cs.Block


def build_model(abi: ABI, policy: str, *, compiled: bool) -> Model:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    ptr_name = "uint32" if abi.pointer_size == 4 else "uint64"
    # Pin the pointer type to the *target*, never inherit the host default.
    cs = cstruct(pointer=ptr_name)
    definition, align = _definition(abi, policy)
    cs.load(definition, compiled=compiled, align=align)
    return Model(abi=abi, policy=policy, cs=cs)


def all_models(*, compiled: bool):
    """The full 2 ABIs x 3 policies matrix."""
    for abi in (ABI_ILP32, ABI_LP64):
        for policy in POLICIES:
            yield build_model(abi, policy, compiled=compiled)


def cstruct_field_offsets(model: Model) -> dict[str, tuple[int, int]]:
    """Return ``{field_name: (offset, size)}`` for the top-level Rec fields.

    The anonymous union is exposed under its compiler-generated name
    ``__anonymous_0__``; its members (type/info/raw) all share that offset.
    """
    out = {}
    for field in model.rec.__fields__:
        out[field.name or "__anonymous_0__"] = (field.offset, field.type.__size__)
    return out
