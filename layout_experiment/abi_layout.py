"""Target-ABI C layout engine (independent of host ctypes / dissect.cstruct).

This module computes struct/union field offsets, padding and total size for a
chosen *target* ABI. It deliberately does NOT use :mod:`ctypes` or any host
platform default: the pointer width and alignment rules are passed in
explicitly so a 64-bit analysis host can reason about 32-bit firmware and
vice versa.

Three layout policies are supported for the same logical definition:

* ``packed``         - ``__attribute__((packacked))`` style: every field is
                       placed at the running offset, no padding anywhere.
* ``natural``        - the target ABI's default alignment rules (SysV-ish:
                       each scalar aligns to its own width, capped at the
                       pointer width; structs tail-pad to their alignment).
* ``explicit``       - a packed struct that carries its own ``uint8 padN[n]``
                       fields, reproducing the natural layout byte for byte.

The engine only knows the handful of scalar types used by the experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ABI:
    """A target ABI described by pointer width and the natural policy."""

    name: str
    pointer_size: int
    max_natural_align: int

    @property
    def pointer_align(self) -> int:
        return self.pointer_size


ABI_ILP32 = ABI("ILP32 (32-bit, e.g. ARM/x86 firmware)", 4, 4)
ABI_LP64 = ABI("LP64 (64-bit, e.g. AArch64/x86_64)", 8, 8)


@dataclass
class Field:
    """A leaf or aggregate field inside a layout node."""

    name: str
    size: int
    alignment: int
    kind: str = "scalar"            # scalar | struct | union | pad | array
    children: list["Field"] = field(default_factory=list)
    count: int = 1
    offset: int | None = None

    @property
    def is_anonymous(self) -> bool:
        return self.name is None or self.name.startswith("__anonymous")


@dataclass
class LayoutResult:
    node: Field
    size: int
    alignment: int
    policy: str
    abi: ABI

    def walk(self):
        """Yield ``(dotted_path, Field, absolute_offset)`` for every leaf."""

        def _walk(node: Field, base: int, path: str):
            here = path if path else node.name or "<root>"
            if node.kind in ("struct", "union"):
                for child in node.children:
                    child_path = f"{here}.{child.name or child.kind}"
                    yield from _walk(child, base + (child.offset or 0), child_path)
            elif node.kind == "array":
                element = node.children[0]
                stride = element.size
                for index in range(node.count):
                    elem_path = f"{here}[{index}]"
                    yield from _walk(element, base + index * stride, elem_path)
            else:
                yield here, node, base

        yield from _walk(self.node, 0, "")


def _scalar(name: str, size: int, abi: ABI) -> Field:
    # Natural alignment of a scalar is its width, capped at the ABI max.
    return Field(name=name, size=size, alignment=min(size, abi.max_natural_align))


def _find(node: Field, name: str) -> Field:
    """Depth-first lookup of a named node (used for natural-layout probes)."""
    if node.name == name:
        return node
    for child in node.children:
        found = _find(child, name)
        if found is not None:
            return found
    raise KeyError(name)


def build_definition(abi: ABI, policy: str = "natural") -> Field:
    """Return the shared logical C definition used by every policy.

    C source (target-independent, ``uintN`` widths come from <stdint.h>)::

        struct Inner {            // natural size 4, align 4 (both ABIs)
            uint16 id;            // 0
            uint16 flags;         // 2
        };

        struct Rec {
            uint32 magic;         // 0
            void  *next;          // pointer: 4 or 8 bytes
            union {               // anonymous, size 4, align 4
                uint32        type;
                struct Inner  info;
                uint32        raw;
            };
            uint16 count;
        };

        struct Block { Rec item[2]; };
    """
    inner = Field(
        name="Inner",
        size=0,
        alignment=2,
        kind="struct",
        children=[
            _scalar("id", 2, abi),
            _scalar("flags", 2, abi),
        ],
    )
    anon_union = Field(
        name="__anonymous_0__",
        size=0,
        alignment=4,
        kind="union",
        children=[
            _scalar("type", 4, abi),
            Field(name="info", size=4, alignment=4, kind="struct",
                  children=[_scalar("id", 2, abi), _scalar("flags", 2, abi)]),
            _scalar("raw", 4, abi),
        ],
    )
    rec_children = [
        _scalar("magic", 4, abi),
        _scalar("next", abi.pointer_size, abi),
        anon_union,
        _scalar("count", 2, abi),
    ]
    # The explicit model spells out every reserve byte the natural ABI would
    # insert, including the struct's tail padding, as named char reserve fields.
    # Derive the tail amount from a natural probe so it follows the real ABI
    # rather than a hand-computed constant.
    if policy == "explicit":
        probe = build_definition(abi, "natural")
        _layout(probe, abi, "natural")
        natural_rec = _find(probe, "Rec")
        last_child = max(natural_rec.children,
                         key=lambda child: (child.offset or 0) + child.size)
        tail_pad = natural_rec.size - ((last_child.offset or 0) + last_child.size)
        rec_children.append(Field(name=f"pad_tail[{tail_pad}]", size=tail_pad,
                                  alignment=1, kind="pad"))
    rec = Field(
        name="Rec",
        size=0,
        alignment=0,
        kind="struct",
        children=rec_children,
    )
    block = Field(
        name="Block",
        size=0,
        alignment=0,
        kind="struct",
        children=[
            Field(name="item", size=0, alignment=0, kind="array", count=2,
                  children=[rec]),
        ],
    )
    return block


def _layout(node: Field, abi: ABI, policy: str, offset: int = 0) -> int:
    """Compute offsets/sizes in place. Returns the node's laid-out size.

    ``natural`` inserts both leading alignment padding and automatic tail
    padding. ``explicit`` inserts leading alignment padding but never tail
    padding - the definition carries its own named reserve fields for that,
    so every reserve byte stays a real, addressable field byte. ``packed``
    inserts no padding at all.
    """
    if policy not in ("packed", "natural", "explicit"):
        raise ValueError(f"unknown policy: {policy}")
    insert_leading = policy in ("natural", "explicit")
    auto_tail = policy == "natural"
    if node.kind == "union":
        node.alignment = min(
            max((c.alignment for c in node.children), default=1),
            abi.max_natural_align,
        )
        body = 0
        for child in node.children:
            child_size = _layout(child, abi, policy, 0)
            body = max(body, child_size)
            child.offset = 0  # every union member starts at offset zero
        if auto_tail:  # unions tail-pad to their own alignment
            body += -body & (node.alignment - 1)
        node.size = body
        return body

    if node.kind == "struct":
        cursor = 0
        struct_align = 1
        new_children: list[Field] = []
        for child in node.children:
            if insert_leading:
                pad = -cursor & (child.alignment - 1)
                if pad:
                    pad_field = Field(
                        name=f"pad{cursor:02x}", size=pad, alignment=1, kind="pad"
                    )
                    pad_field.offset = cursor
                    new_children.append(pad_field)
                    cursor += pad
            child.offset = cursor
            new_children.append(child)
            child_size = _layout(child, abi, policy, cursor)
            cursor += child_size
            struct_align = max(struct_align, child.alignment)
        node.children = new_children
        # A struct's alignment is its strictest member's, but it can never
        # exceed the target ABI's maximum scalar alignment (e.g. an all-u16
        # struct stays 4-aligned even inside a 64-bit LP64 program).
        struct_align = min(struct_align, abi.max_natural_align)
        node.alignment = struct_align
        if auto_tail:  # struct tail padding up to struct alignment
            cursor += -cursor & (struct_align - 1)
        node.size = cursor
        return cursor

    if node.kind == "array":
        element = node.children[0]
        element_size = _layout(element, abi, policy, 0)
        node.alignment = min(element.alignment, abi.max_natural_align)
        if auto_tail:
            # Each element (including the last) occupies a stride rounded up to
            # the element alignment, so element N+1 starts on a valid boundary.
            stride = element_size + (-element_size & (node.alignment - 1))
        else:
            stride = element_size
        node.size = stride * node.count
        element.offset = 0
        return node.size

    return node.size  # scalar / explicit pad leaf


def layout(abi: ABI, policy: str) -> LayoutResult:
    root = build_definition(abi, policy)
    size = _layout(root, abi, policy)
    return LayoutResult(node=root, size=size, alignment=root.alignment,
                        policy=policy, abi=abi)


def byte_map(result: LayoutResult) -> list[tuple[int, str]]:
    """Assign every byte of the layout to a human-readable owner label.

    Returns a list of ``(byte_offset, label)`` that doubles as a hand-checkable
    byte diagram and as the source of truth for byte-level tests.
    """
    owners: dict[int, str] = {}

    def _paint(node: Field, base: int, path: str):
        here = path if path else node.name
        if node.kind in ("struct", "union"):
            for child in node.children:
                _paint(child, base + (child.offset or 0),
                       f"{here}.{child.name or child.kind}")
        elif node.kind == "array":
            element = node.children[0]
            stride = node.size // node.count
            for index in range(node.count):
                elem_path = f"{here}[{index}]"
                # Tail padding of each element is part of its array stride;
                # paint those bytes in the element's name so no byte is lost.
                _paint(element, base + index * stride, elem_path)
                painted_end = base + index * stride + element.size
                stride_end = base + index * stride + stride
                for pad_byte in range(painted_end, stride_end):
                    owners[pad_byte] = f"{elem_path} (tail padding)"
        else:
            label = here if node.kind != "pad" else f"{here} (padding)"
            for byte_index in range(node.size):
                owners[base + byte_index] = label

    _paint(result.node, 0, "")
    # Any byte never claimed by a child is implicit struct/union tail padding.
    for offset in range(result.size):
        owners.setdefault(offset, "implicit tail padding")
    return [(offset, owners[offset]) for offset in range(result.size)]


def render_diagram(result: LayoutResult) -> str:
    """Render an ASCII byte diagram grouped in 16-byte rows."""
    rows = []
    mapping = dict(byte_map(result))
    for start in range(0, result.size, 16):
        cells = []
        for offset in range(start, min(start + 16, result.size)):
            label = mapping[offset]
            short = label.replace("Block.item", "item").replace(
                "__anonymous_0__", "anon")
            cells.append(f"{offset:02x}:{short}")
        rows.append("  " + " | ".join(cells))
    return "\n".join(rows)
