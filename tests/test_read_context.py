from __future__ import annotations

from io import BytesIO

import pytest

from dissect.cstruct import ReadContext
from dissect.cstruct.cstruct import cstruct
from dissect.cstruct.exception import NullPointerDereference, PointerBoundsError, PointerDepthError
from tests.util import verify_compiled

CDEF = """
struct header {
    uint32  *data;
    char    *name;
};

struct node {
    uint32  value;
    node    *next;
};
"""

# Two 4-byte headers, followed by the data they point to
#   0x00: header1 (data=0x08, name=0x0c)
#   0x04: header2 (data=0x10, name=0x14)
#   0x08: uint32 0xbeef
#   0x0c: "ab\0\0"
#   0x10: uint32 0xdeadbeef
#   0x14: "xy\0\0"
SHARED_BUF = (
    b"\x08\x00\x0c\x00"
    b"\x10\x00\x14\x00"
    b"\xef\xbe\x00\x00"
    b"ab\x00\x00"
    b"\xef\xbe\xad\xde"
    b"xy\x00\x00"
)


@pytest.fixture
def cs_loaded(cs: cstruct, compiled: bool) -> cstruct:
    cs.pointer = cs.uint16
    cs.load(CDEF, compiled=compiled)
    assert verify_compiled(cs.header, compiled)
    assert verify_compiled(cs.node, compiled)
    return cs


def test_read_context_shared_stream(cs_loaded: cstruct, compiled: bool) -> None:
    """Two structures share one stream and one read context, with interleaved dereferences."""
    stream = BytesIO(SHARED_BUF)
    read_context = ReadContext(0, len(SHARED_BUF))

    obj1 = cs_loaded.header.read(stream, read_context=read_context)
    obj2 = cs_loaded.header.read(stream, read_context=read_context)
    assert stream.tell() == 8

    # Interleaved dereferences of both structures through the same context
    assert obj1.data.dereference() == 0xBEEF
    assert obj2.name.dereference() == b"xy"
    assert obj1.name.dereference() == b"ab"
    assert obj2.data.dereference() == 0xDEADBEEF

    # Every dereference restored the stream position
    assert stream.tell() == 8

    # The accessed intervals were recorded in dereference order
    assert read_context.accessed == [(8, 12), (20, 23), (12, 15), (16, 20)]


def test_read_context_out_of_bounds(cs_loaded: cstruct, compiled: bool) -> None:
    """Targets beyond the end, before the start, and reads overflowing the end are rejected."""
    read_context = ReadContext(4, len(SHARED_BUF))

    # Target before the start of the allowed range
    stream = BytesIO(b"\x02\x00\x04\x00")
    obj = cs_loaded.header.read(stream, read_context=read_context)
    with pytest.raises(PointerBoundsError, match="outside the allowed range"):
        obj.data.dereference()
    assert stream.tell() == 4

    stream = BytesIO(SHARED_BUF)
    obj = cs_loaded.header.read(stream, read_context=read_context)

    # Target inside the range, but the read length overflows the end
    with pytest.raises(PointerBoundsError, match="outside the allowed range"):
        (obj.data + 0x10).dereference()  # 0x18 + 4 bytes > end

    # Pointer arithmetic landing outside of the range is checked at the target
    with pytest.raises(PointerBoundsError):
        (obj.data + 0x100).dereference()
    with pytest.raises(PointerBoundsError):
        (obj.data - 0x100).dereference()

    # Failed dereferences leave the stream position untouched
    assert stream.tell() == 4
    assert read_context.accessed == []


def test_read_context_unterminated_string(cs_loaded: cstruct, compiled: bool) -> None:
    """A string pointer without a terminator inside the bounds fails closed."""
    buf = b"\x04\x00\x04\x00" + b"abcdef"
    stream = BytesIO(buf)
    read_context = ReadContext(0, len(buf))

    obj = cs_loaded.header.read(stream, read_context=read_context)
    with pytest.raises(EOFError):
        obj.name.dereference()

    # The stream position is restored and the partially read interval is recorded
    assert stream.tell() == 4
    assert read_context.accessed == [(4, 10)]


def test_read_context_nested(cs_loaded: cstruct, compiled: bool) -> None:
    """Nested pointers inherit the range constraint of the parent context."""
    # node* at 0 -> node at 4: value=1, next=0x40 (out of bounds)
    buf = b"\x04\x00\x00\x00" + (1).to_bytes(4, "little") + b"\x40\x00"
    stream = BytesIO(buf)
    read_context = ReadContext(0, len(buf))

    node_ptr = cs_loaded.node.__cs__._make_pointer(cs_loaded.node).read(stream, read_context=read_context)
    obj = node_ptr.dereference()
    assert obj.value == 1
    assert obj.next == 0x40

    with pytest.raises(PointerBoundsError):
        obj.next.dereference()

    assert stream.tell() == 2
    assert read_context.accessed == [(4, 10)]


def test_read_context_depth(cs_loaded: cstruct, compiled: bool) -> None:
    """Cyclic references stop at the explicit depth policy, aliases may share data."""
    # node* at 0 -> node at 4: value=0x2a, next=0x04 (self-referential cycle)
    buf = b"\x04\x00\x00\x00" + (0x2A).to_bytes(4, "little") + b"\x04\x00"
    stream = BytesIO(buf)
    read_context = ReadContext(0, len(buf), max_depth=2)

    pointer_type = cs_loaded.node.__cs__._make_pointer(cs_loaded.node)
    root = pointer_type.read(stream, read_context=read_context)

    node = root.dereference()
    assert node.value == 0x2A
    node = node.next.dereference()
    assert node.value == 0x2A
    with pytest.raises(PointerDepthError, match="maximum depth"):
        node.next.dereference()

    # Aliased pointers to the same node remain legal, regardless of the chain depth
    alias = pointer_type.read(BytesIO(buf), read_context=read_context)
    assert alias.dereference().value == 0x2A
    assert alias.dereference().next.dereference().value == 0x2A

    assert stream.tell() == 2


def test_read_context_null_pointer(cs_loaded: cstruct, compiled: bool) -> None:
    """Null pointers keep raising NullPointerDereference when constrained."""
    obj = cs_loaded.header.read(BytesIO(b"\x00\x00\x00\x00"), read_context=ReadContext(0, 4))
    with pytest.raises(NullPointerDereference):
        obj.data.dereference()


def test_read_context_default_unconstrained(cs_loaded: cstruct, compiled: bool) -> None:
    """Callers that don't opt in to a read context keep the original unconstrained behavior."""
    stream = BytesIO(SHARED_BUF)
    obj1 = cs_loaded.header.read(stream)
    obj2 = cs_loaded.header.read(stream)

    assert obj1.data.dereference() == 0xBEEF
    assert obj2.name.dereference() == b"xy"
    assert stream.tell() == 8

    # An out-of-bounds pointer without a constraint fails with a plain EOFError,
    # and the stream position is restored afterwards
    buf = b"\x04\x00\x00\x00" + (1).to_bytes(4, "little") + b"\x40\x00"
    stream = BytesIO(buf)
    node_ptr = cs_loaded.node.__cs__._make_pointer(cs_loaded.node).read(stream)
    obj = node_ptr.dereference()
    with pytest.raises(EOFError):
        obj.next.dereference()
    assert stream.tell() == 2


def test_read_context_reads_api(cs_loaded: cstruct, compiled: bool) -> None:
    """The read context can be supplied through reads() as well."""
    obj = cs_loaded.header.reads(SHARED_BUF[:4], read_context=ReadContext(0, 4))
    # The buffer is only 4 bytes, so every pointer target is out of bounds
    with pytest.raises(PointerBoundsError):
        obj.data.dereference()

    obj = cs_loaded.header.reads(SHARED_BUF, read_context=ReadContext(0, len(SHARED_BUF)))
    assert obj.data.dereference() == 0xBEEF


def test_read_context_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        ReadContext(-1)
    with pytest.raises(ValueError, match="smaller"):
        ReadContext(10, 5)
    with pytest.raises(ValueError, match="at least 1"):
        ReadContext(0, 10, max_depth=0)
