"""Hand-checkable byte buffers and field-level read/write verification.

A buffer is generated directly from the independent layout engine: every leaf
field gets a distinct, offset-derived marker value and every padding byte gets
the sentinel ``0xCC``. The same buffer is then parsed by the dissect.cstruct
models, proving that each field is read from the offset the byte diagram says
it occupies.
"""
from __future__ import annotations

from io import BytesIO

from abi_layout import LayoutResult, byte_map

PAD_SENTINEL = 0xCC


def padding_offsets(result: LayoutResult) -> set[int]:
    """Absolute byte offsets owned by padding (explicit pad fields or tail)."""
    pads: set[int] = set()

    def _collect(node, base: int):
        if node.kind in ("struct", "union"):
            for child in node.children:
                child_base = base + (child.offset or 0)
                if child.kind == "pad":
                    pads.update(range(child_base, child_base + child.size))
                _collect(child, child_base)
            # Bytes beyond the last child are implicit tail padding.
            if node.children:
                last = max(node.children, key=lambda c: (c.offset or 0) + c.size)
                tail_start = base + (last.offset or 0) + last.size
                pads.update(range(tail_start, base + node.size))
        elif node.kind == "array":
            element = node.children[0]
            stride = node.size // node.count
            for index in range(node.count):
                elem_base = base + index * stride
                _collect(element, elem_base)
                pads.update(range(elem_base + element.size, elem_base + stride))

    _collect(result.node, 0)
    return {offset for offset in pads if offset < result.size}


def build_buffer(result: LayoutResult) -> bytearray:
    """Lay out one distinct marker byte per absolute offset.

    Every byte of a field gets marker ``0x10 + (offset & 0x0f)``; padding
    bytes keep the ``0xCC`` sentinel. Overlapping union members therefore read
    the same underlying bytes, and expected integer values are reconstructed
    from the buffer with :func:`expected_int` rather than assumed per path.
    """
    buf = bytearray([PAD_SENTINEL]) * result.size
    for path, node, absolute in result.walk():
        if node.kind == "pad":
            continue
        for index in range(node.size):
            buf[absolute + index] = 0x10 + ((absolute + index) & 0x0F)
    return buf


def expected_int(buf: bytes, result: LayoutResult, path: str) -> int:
    """Reconstruct the little-endian integer a leaf path must evaluate to."""
    for walked_path, node, absolute in result.walk():
        if walked_path == path:
            return int.from_bytes(buf[absolute:absolute + node.size], "little")
    raise KeyError(path)


def parse_block(model, buf: bytes):
    return model.block(BytesIO(buf))


def check_round_trip(result: LayoutResult, model, buf: bytes) -> None:
    """Assert write-back semantics for the model.

    * packed / explicit: every byte is a real field byte (explicit ``padNN``
      reserve fields are themselves named fields) -> exact reproduction.
    * natural: implicit padding is not backed by a field, so cstruct emits zero
      bytes there, while every field byte reproduces the input exactly.
    """
    obj = parse_block(model, bytes(buf))
    out = obj.dumps()
    implicit_pads = padding_offsets(result) if result.policy == "natural" else set()
    assert len(out) == len(buf)
    for offset, (in_byte, out_byte) in enumerate(zip(buf, out)):
        if offset in implicit_pads:
            assert out_byte == 0, f"implicit pad @{offset:#x} not zeroed: {out_byte:#x}"
        else:
            assert out_byte == in_byte, (
                f"field byte @{offset:#x} changed: {in_byte:#x} -> {out_byte:#x}"
            )


def leaf_values(obj, index: int) -> dict[str, int]:
    """Extract scalar leaf values from ``Block.item[index]`` as plain ints."""
    rec = obj.item[index]
    anon = rec.__anonymous_0__
    return {
        "magic": int(rec.magic),
        "next": int(rec.next),
        "type": int(anon.type),
        "raw": int(anon.raw),
        "info.id": int(anon.info.id),
        "info.flags": int(anon.info.flags),
        "count": int(rec.count),
    }
