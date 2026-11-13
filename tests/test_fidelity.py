from __future__ import annotations

import pytest

from dissect.cstruct import (
    FidelityResult,
    LayoutPreservationError,
    cstruct,
    fidelity_edit,
)
from dissect.cstruct.types.structure import get_fidelity_buffer
from tests.util import NonSeekableStream, verify_compiled


@pytest.fixture
def aligned_cs(cs: cstruct, compiled: bool) -> cstruct:
    cs.load(
        """
        struct record {
            uint8   tag;     // 0x00
            uint32  value;   // 0x04 (3 bytes of padding at 0x01-0x03)
            uint16  n;       // 0x08
        };                          // 2 bytes of tail padding at 0x0a-0x0b
        """,
        compiled=compiled,
        align=True,
    )
    assert verify_compiled(cs.record, compiled)
    return cs


def test_fidelity_edit_preserves_nonzero_padding(aligned_cs: cstruct) -> None:
    # Padding carries non-zero "evidence" bytes that normal reserialization would wipe
    original = bytes.fromhex("AA 11 22 33 BB CC CC CC DD DD EE EE")
    obj = aligned_cs.record(original)

    result = obj.fidelity_edit({"value": 0x11223344})

    assert isinstance(result, FidelityResult)
    # Only the four bytes of `value` (0x04-0x07) change; everything else is byte-identical
    assert result.data == bytes.fromhex("AA 11 22 33 44 33 22 11 DD DD EE EE")
    assert result.changed_offsets == [0x04, 0x05, 0x06, 0x07]
    assert {c.region for c in result.changes} == {"field"}
    assert {c.path for c in result.changes} == {"value"}

    # The padding bytes are explicitly reported as padding and left untouched
    padding = [r for r in result.regions if r.kind == "padding"]
    assert [(r.start, r.size) for r in padding] == [(1, 3), (10, 2)]
    assert result.data[1:4] == original[1:4]
    assert result.data[10:12] == original[10:12]


def test_normal_dumps_zeroes_padding_but_fidelity_does_not(aligned_cs: cstruct) -> None:
    original = bytes.fromhex("AA 11 22 33 BB CC CC CC DD DD EE EE")
    obj = aligned_cs.record(original)
    obj.value = 0x11223344

    # The whole point: dumps rebuilds the object and erases padding, fidelity editing does not
    assert obj.dumps() == bytes.fromhex("AA 00 00 00 44 33 22 11 DD DD 00 00")
    assert obj.fidelity_edit().data == bytes.fromhex("AA 11 22 33 44 33 22 11 DD DD EE EE")


def test_fidelity_byte_by_byte_diff(aligned_cs: cstruct) -> None:
    original = bytes.fromhex("AA 11 22 33 BB CC CC CC DD DD EE EE")
    result = aligned_cs.record(original).fidelity_edit({"value": 0x00000001})

    assert [(c.offset, c.old, c.new, c.path) for c in result.changes] == [
        (4, 0xBB, 0x01, "value"),
        (5, 0xCC, 0x00, "value"),
        (6, 0xCC, 0x00, "value"),
        (7, 0xCC, 0x00, "value"),
    ]
    assert "value" in result.diff()


def test_fidelity_edit_nested_structure_and_fixed_array(cs: cstruct, compiled: bool) -> None:
    cs.load(
        """
        struct inner {
            uint16 a;
            uint16 b;
        };
        struct outer {
            uint8         hdr;
            struct inner  in;
            uint8         arr[4];
        };
        """,
        compiled=compiled,
    )
    original = bytes.fromhex("11 22 22 33 33 AA BB CC DD")
    obj = cs.outer(original)

    result = obj.fidelity_edit({"in.b": 0xABCD})
    assert result.data == bytes.fromhex("11 22 22 CD AB AA BB CC DD")
    assert [(c.offset, c.path) for c in result.changes] == [(3, "in.b"), (4, "in.b")]

    # A same-length fixed array is fine; growing it is rejected (would move the trailing bytes)
    assert obj.fidelity_edit({"arr": b"\x01\x02\x03\x04"}).data[-4:] == b"\x01\x02\x03\x04"

    with pytest.raises(LayoutPreservationError, match="fixed array"):
        obj.fidelity_edit({"arr": b"\x01\x02\x03\x04\x05"})

    with pytest.raises(LayoutPreservationError, match="fixed array"):
        obj.fidelity_edit({"arr": b"\x01\x02"})


def test_fidelity_edit_array_of_structures(cs: cstruct, compiled: bool) -> None:
    cs.load(
        """
        struct pair { uint8 x; uint8 y; };
        struct holder { uint8 z; struct pair p[2]; };
        """,
        compiled=compiled,
    )
    obj = cs.holder(bytes.fromhex("09 01 02 03 04"))
    result = obj.fidelity_edit({"p": [cs.pair(0x10, 0x20), cs.pair(0x03, 0x04)]})

    assert result.data == bytes.fromhex("09 10 20 03 04")
    assert [c.offset for c in result.changes] == [1, 2]
    assert [c.path for c in result.changes] == ["p[0].x", "p[0].y"]


def test_fidelity_edit_union_preserves_inactive_bytes(cs: cstruct, compiled: bool) -> None:
    cs.load(
        """
        union u {
            uint32  x;
            uint16  y[2];
        };
        struct s {
            uint8   a;
            union u u;
        };
        """,
        compiled=compiled,
    )
    original = bytes.fromhex("11 78 56 34 12")
    obj = cs.s(original)

    result = obj.fidelity_edit({"u.x": 0xAABBCCDD})
    assert result.data == bytes.fromhex("11 DD CC BB AA")
    union_region = [r for r in result.regions if r.kind == "union"]
    assert [(r.start, r.size, r.path) for r in union_region] == [(1, 4, "u")]
    assert all(c.path == "u" and c.region == "union" for c in result.changes)


def test_fidelity_edit_anonymous_union(cs: cstruct, compiled: bool) -> None:
    cs.load(
        """
        struct withanon {
            uint8 tag;
            union {
                uint32 a;
                uint8  b[4];
            };
        };
        """,
        compiled=compiled,
    )
    original = bytes.fromhex("22 01 02 03 04")
    result = cs.withanon(original).fidelity_edit({"a": 0xAABBCCDD})
    assert result.data == bytes.fromhex("22 DD CC BB AA")
    assert [r for r in result.regions if r.kind == "union"]


def test_fidelity_edit_bitfield_group(cs: cstruct, compiled: bool) -> None:
    cs.load("struct bits { uint8 lo : 3; uint8 hi : 5; uint8 full; };", compiled=compiled)
    original = bytes.fromhex("A6 77")
    obj = cs.bits(original)

    result = obj.fidelity_edit({"lo": 1})
    # 0xA6 -> 0xA1 (only low 3 bits change), the unrelated `full` byte is preserved exactly
    assert result.data == bytes.fromhex("A1 77")
    assert [c.offset for c in result.changes] == [0]
    assert result.changes[0].region == "bitfield"
    assert result.data[1] == 0x77


def test_fidelity_edit_rejects_dynamic_layout(cs: cstruct, compiled: bool) -> None:
    cs.load(
        """
        struct dyn {
            uint16 count;
            uint8  data[count];
        };
        """,
        compiled=compiled,
    )
    obj = cs.dyn(bytes.fromhex("02 00 11 22"))
    with pytest.raises(LayoutPreservationError, match="dynamically sized"):
        obj.fidelity_edit({"count": 3})


def test_fidelity_edit_rejects_nested_dynamic_array(cs: cstruct, compiled: bool) -> None:
    cs.load(
        """
        struct wrapper {
            uint32 n;
            uint8  data[n];
        };
        struct root {
            struct wrapper w;
        };
        """,
        compiled=compiled,
    )
    obj = cs.root(bytes.fromhex("01 00 00 00 AA"))
    with pytest.raises(LayoutPreservationError, match="dynamically sized"):
        obj.fidelity_edit({"w.n": 2})


def test_fidelity_edit_requires_exact_length_original(aligned_cs: cstruct) -> None:
    obj = aligned_cs.record(bytes.fromhex("AA 11 22 33 BB CC CC CC DD DD EE EE"))
    with pytest.raises(LayoutPreservationError, match="exact-length"):
        fidelity_edit(obj, {"tag": 1}, original=b"\x00\x01")


def test_fidelity_edit_needs_original_for_non_seekable_stream(cs: cstruct) -> None:
    # The compiled packed reader can consume a non-seekable stream but cannot snapshot its bytes, so the
    # original record must be supplied explicitly. (The interpreted reader requires tell() up front.)
    cs.load("struct packed_rec { uint8 tag; uint32 value; };", compiled=True)
    original = bytes.fromhex("AA BB CC CC CC")

    obj = cs.packed_rec(NonSeekableStream(original))
    assert get_fidelity_buffer(obj) is None

    with pytest.raises(LayoutPreservationError, match="original"):
        obj.fidelity_edit({"value": 1})

    # Passing the original record explicitly restores the in-place guarantee
    result = fidelity_edit(obj, {"value": 0x11223344}, original=original)
    assert result.data == bytes.fromhex("AA 44 33 22 11")


def test_fidelity_edit_accepts_bytearray_and_memoryview(aligned_cs: cstruct) -> None:
    original = bytes.fromhex("AA 11 22 33 BB CC CC CC DD DD EE EE")
    obj = aligned_cs.record(original)

    assert obj.fidelity_edit({"tag": 1}, original=bytearray(original)).data[0] == 1
    assert obj.fidelity_edit({"tag": 2}, original=memoryview(original)).data[0] == 2


def test_fidelity_edit_does_not_mutate_input(aligned_cs: cstruct) -> None:
    original = bytearray.fromhex("AA 11 22 33 BB CC CC CC DD DD EE EE")
    snapshot = bytes(original)
    aligned_cs.record(bytes(original)).fidelity_edit({"value": 0x11223344}, original=original)
    assert bytes(original) == snapshot


def test_fidelity_edit_idempotent_no_changes(aligned_cs: cstruct) -> None:
    original = bytes.fromhex("AA 11 22 33 BB CC CC CC DD DD EE EE")
    result = aligned_cs.record(original).fidelity_edit()
    assert result.data == original
    assert result.changes == []


def test_fidelity_edit_module_level_function(aligned_cs: cstruct) -> None:
    original = bytes.fromhex("AA 11 22 33 BB CC CC CC DD DD EE EE")
    obj = aligned_cs.record(original)
    assert fidelity_edit(obj, {"n": 1}).data == bytes.fromhex("AA 11 22 33 BB CC CC CC 01 00 EE EE")


def test_fidelity_edit_nested_change_through_union_proxy(cs: cstruct, compiled: bool) -> None:
    cs.load(
        """
        struct point { uint16 x; uint16 y; };
        struct box {
            uint8  kind;
            union {
                struct point p;
                uint32 raw;
            };
        };
        """,
        compiled=compiled,
    )
    original = bytes.fromhex("07 10 00 20 00")
    box = cs.box(original)

    # The nested structure is wrapped in a union proxy; editing through it rebuilds the union
    box.p.x = 0x0102
    result = box.fidelity_edit()

    assert result.data == bytes.fromhex("07 02 01 20 00")
    assert all(change.region == "union" for change in result.changes)
    # The untouched member byte of the union and the leading field are preserved
    assert result.data[0] == original[0]
    assert result.data[3:5] == original[3:5]
