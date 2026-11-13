from __future__ import annotations

import io

import pytest

from dissect.cstruct.cstruct import cstruct
from dissect.cstruct.types.base import _is_eof
from tests.util import ShortReadStream, verify_compiled


# A fixed structure, a length-prefixed array and a trailing fixed field after the dynamic array, so a
# misaligned field boundary is detected immediately.
STRUCT_DEF = """
struct entry {
    uint32  id;
    uint16  flags;
    uint32  count;
    int32   values[count];
    char    tag[4];
};
"""


def _make_cs(compiled: bool) -> cstruct:
    cs = cstruct()
    cs.load(STRUCT_DEF, compiled=compiled)
    assert verify_compiled(cs.entry, compiled)
    return cs


def _sample() -> bytes:
    cs = cstruct()
    cs.load(STRUCT_DEF)
    obj = cs.entry(id=0x44332211, flags=0x6655, count=3, values=[-1, 2, -3], tag=b"TAG!")
    return obj.dumps()


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
@pytest.mark.parametrize("max_chunk", [1, 2, 3])
def test_short_read_matches_normal_parse(compiled: bool, max_chunk: int) -> None:
    """A stream returning 1-3 bytes per read must parse to the exact same object as a normal stream."""
    cs = _make_cs(compiled)
    data = _sample()

    expected = cs.entry(io.BytesIO(data))
    parsed = cs.entry(ShortReadStream(data, max_chunk))

    assert parsed == expected
    assert parsed.id == expected.id
    assert parsed.flags == expected.flags
    assert parsed.count == expected.count
    assert parsed.values == expected.values
    assert parsed.tag == expected.tag


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
@pytest.mark.parametrize("max_chunk", [1, 2, 3])
def test_two_records_keep_their_boundary(compiled: bool, max_chunk: int) -> None:
    """The second of two concatenated records must start at the correct byte boundary."""
    cs = _make_cs(compiled)
    first = cs.entry(id=1, flags=2, count=2, values=[10, 20], tag=b"AAAA").dumps()
    second = cs.entry(id=3, flags=4, count=1, values=[30], tag=b"BBBB").dumps()

    stream = ShortReadStream(first + second, max_chunk)
    parsed_first = cs.entry(stream)
    parsed_second = cs.entry(stream)

    assert parsed_first.id == 1
    assert parsed_first.values == [10, 20]
    assert parsed_first.tag == b"AAAA"

    assert parsed_second.id == 3
    assert parsed_second.flags == 4
    assert parsed_second.values == [30]
    assert parsed_second.tag == b"BBBB"

    # Everything was consumed, no bytes were dropped or replayed between the records
    assert stream.read() == b""


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
@pytest.mark.parametrize("max_chunk", [1, 2, 3])
def test_every_truncation_raises_eof(compiled: bool, max_chunk: int) -> None:
    """Truncating the input at every position must raise an EOFError, never parse shifted fields."""
    cs = _make_cs(compiled)
    data = _sample()

    for cut in range(len(data)):
        truncated = data[:cut]
        with pytest.raises(EOFError) as normal:
            cs.entry(io.BytesIO(truncated))

        with pytest.raises(EOFError) as short:
            cs.entry(ShortReadStream(truncated, max_chunk))

        # The short-read stream reports the same amount of consumed bytes as the normal stream
        assert str(short.value) == str(normal.value)


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
@pytest.mark.parametrize("max_chunk", [1, 2, 3])
def test_short_read_primitive_and_arrays(compiled: bool, max_chunk: int) -> None:
    """Short reads must also work for standalone primitives, fixed arrays and EOF arrays."""
    cs = cstruct()
    cs.load("struct inner { uint16 a; uint16 b; };", compiled=compiled)

    assert cs.uint32(ShortReadStream(b"\x01\x02\x03\x04", max_chunk)) == 0x04030201
    assert cs.uint16[3](ShortReadStream(b"\x01\x00\x02\x00\x03\x00", max_chunk)) == [1, 2, 3]
    assert cs.uint24[2](ShortReadStream(b"\x01\x00\x00\x02\x00\x00", max_chunk)) == [1, 2]
    assert cs.char[4](ShortReadStream(b"abcd", max_chunk)) == b"abcd"
    assert cs.wchar[2](ShortReadStream("ab".encode("utf-16-le"), max_chunk)) == "ab"

    nested_data = b"\x01\x00\x02\x00\x03\x00\x04\x00"
    objects = cs.inner[2](ShortReadStream(nested_data, max_chunk))
    assert [(obj.a, obj.b) for obj in objects] == [(1, 2), (3, 4)]

    # An EOF array consumes the whole stream even when the chunks are tiny
    assert cs.uint16["EOF"](ShortReadStream(b"\x01\x00\x02\x00", max_chunk)) == [1, 2]

    # A null-terminated array stops exactly at its zero terminator
    assert cs.uint16[None](ShortReadStream(b"\x01\x00\x02\x00\x00\x00", max_chunk)) == [1, 2]


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "interpreted"])
def test_null_terminated_short_read(compiled: bool) -> None:
    """Null-terminated char and wchar arrays keep their terminator boundary on short reads."""
    cs = cstruct()

    fh = ShortReadStream(b"abc\x00rest", 2)
    assert cs.char[None](fh) == b"abc"
    assert fh.read() == b"rest"

    fh = ShortReadStream("abc".encode("utf-16-le") + b"\x00\x00rest", 2)
    assert cs.wchar[None](fh) == "abc"
    assert fh.read() == b"rest"


def test_non_seekable_short_read_supported_for_sequential_types() -> None:
    """Plain sequential structures must work on a non-seekable short-read stream."""
    cs = cstruct()
    cs.load(STRUCT_DEF)
    data = _sample()

    parsed = cs.entry(ShortReadStream(data, 2, seekable_stream=False))
    assert parsed == cs.entry(io.BytesIO(data))


def test_non_seekable_features_are_rejected() -> None:
    """Features that inherently need seeking must raise instead of silently dropping bytes."""
    # Aligned structures need to reposition the stream, on both read paths
    for compiled in (True, False):
        aligned = cstruct()
        aligned.load("struct aligned { uint8 a; uint32 b; };", compiled=compiled, align=True)
        with pytest.raises(io.UnsupportedOperation):
            aligned.aligned(ShortReadStream(b"\x00" * 8, 2, seekable_stream=False))

    # A dynamic union has to rewind to capture its bytes
    dynamic_union = cstruct()
    dynamic_union.load("union dyn { char a[]; uint32 b; };")
    with pytest.raises(io.UnsupportedOperation):
        dynamic_union.dyn(ShortReadStream(b"abc", 2, seekable_stream=False))

    # Pointer dereferencing seeks to the pointed-to address
    cs = cstruct()
    cs.pointer = cs.uint32
    pointer_type = cs._make_pointer(cs.uint32)
    pointer = pointer_type.__new__(
        pointer_type, 4, ShortReadStream(b"\x00" * 16, 2, seekable_stream=False), context=None, endian="<"
    )
    with pytest.raises(io.UnsupportedOperation):
        pointer.dereference()

    # EOF detection on a generic element type must not swallow the probe byte
    with pytest.raises(io.UnsupportedOperation):
        _is_eof(ShortReadStream(b"\x01", 1, seekable_stream=False))
