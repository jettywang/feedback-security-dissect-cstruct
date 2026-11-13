from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest

from dissect.cstruct import cstruct
from dissect.cstruct.schema import instance_schema

if TYPE_CHECKING:
    from dissect.cstruct.types.base import BaseType

RECORD_V1 = """
struct header {
    uint8   magic[4];
    uint16  count;
};

struct entry {
    uint16  id;
    uint8   flags;
};

struct record {
    header          head;
    entry           entries[2];
    union {
        uint32      raw;
        struct {
            uint16  kind;
            uint16  length;
        } typed;
    } payload;
    uint8           body[6];
};
"""

RECORD_V2 = """
struct header {
    uint8   magic[4];
    uint32  count;
};

struct entry {
    uint16  id;
    uint16  flags;
};

struct record {
    header          head;
    entry           entries[2];
    union {
        uint32      raw;
        struct {
            uint16  kind;
            uint16  length;
        } typed;
    } payload;
    uint8           body[12];
};
"""

# Hand-built records with distinct values so a wrong field interpretation is visible.
RECORD_V1_BYTES = (
    b"SEC1"  # magic
    b"\x02\x00"  # count
    b"\x11\x00\x01"  # entries[0]: id=0x11, flags=1
    b"\x22\x00\x02"  # entries[1]: id=0x22, flags=2
    b"\x34\x12\x06\x00"  # payload: kind=0x1234, length=6
    b"ABCDEF"  # body
)

RECORD_V2_BYTES = (
    b"SEC2"  # magic
    b"\x00\x01\x00\x00"  # count=256
    b"\x11\x00\x01\x00"  # entries[0]: id=0x11, flags=1
    b"\x22\x00\x02\x00"  # entries[1]: id=0x22, flags=2
    b"\x34\x12\x0c\x00"  # payload: kind=0x1234, length=12
    b"ABCDEFGHIJKL"  # body
)


def _define(definition: str) -> cstruct:
    return cstruct().load(definition, compiled=False)


def _snapshot_values(obj: object) -> list[int]:
    """Return the field values flattened, including array and union values, without enum helpers."""
    values: list[int] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("__dynamic_sizes__", "_buf", "_endian"):
                    continue
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif hasattr(value, "__dict__") and hasattr(value.__class__, "__fields__"):
            walk({k: v for k, v in value.__dict__.items() if k not in ("__dynamic_sizes__", "_buf", "_endian")})
        elif isinstance(value, int):
            values.append(int(value))

    walk(obj)
    return values


def test_snapshot_is_immutable() -> None:
    cs = _define(RECORD_V1)
    schema = cs.snapshot()

    assert schema.revision == cs.revision
    assert schema.digest == cs.snapshot().digest
    assert schema.resolve("record") is schema.record

    record = schema.record(RECORD_V1_BYTES)
    assert record.head.count == 2
    assert record.entries[1].id == 0x22
    assert record.payload.typed.length == 6

    # Changing the live definition after the snapshot was taken must not change the snapshot.
    cs.entry.add_field("extra", cs.uint8)
    schema.record(RECORD_V1_BYTES)

    assert [field.name for field in schema.entry.__fields__] == ["id", "flags"]
    assert [field.name for field in cs.entry.__fields__] == ["id", "flags", "extra"]


def test_snapshot_reads_are_pinned_and_bound() -> None:
    cs = _define(RECORD_V1)
    snapshot = cs.snapshot()
    record = snapshot.record(RECORD_V1_BYTES)

    assert instance_schema(record) is snapshot
    assert type(record).__cs__ is snapshot

    # A snapshot class cannot be structurally edited, so two parses cannot drift apart.
    with pytest.raises(TypeError, match="immutable"):
        snapshot.entry.add_field("extra", snapshot.uint8)


def test_default_parse_is_latest_and_old_instance_dumps_stays_stable() -> None:
    cs = _define(RECORD_V1)
    old = cs.record(RECORD_V1_BYTES)

    # The default documented behaviour is to use the latest definitions for new parses.
    cs.entry.add_field("extra", cs.uint8)
    new = cs.record(RECORD_V1_BYTES + b"\x07")

    assert new.entries[0].extra == 0x22
    assert old.dumps() == RECORD_V1_BYTES
    assert len(new) == len(RECORD_V1_BYTES) + 2

    # The old object still exposes the version it was parsed with.
    assert [field.name for field in old.__class__.__fields__] == ["head", "entries", "payload", "body"]
    assert [field.name for field in old.entries[0].__class__.__fields__] == ["id", "flags"]


def test_old_instance_with_nested_union_dumps() -> None:
    cs = _define(RECORD_V1)
    old = cs.record(RECORD_V1_BYTES)

    # Mutate both the union member and the nested array element type after parsing.
    cs.record.__fields__[2].type.add_field("extra", cs.uint8)
    cs.entry.add_field("extra", cs.uint8)

    assert old.dumps() == RECORD_V1_BYTES
    assert old.payload.typed.length == 6
    assert [field.name for field in old.payload.__class__.__fields__] == ["raw", "typed"]


def test_pinned_definition_used_for_new_parses() -> None:
    cs = _define(RECORD_V1)
    pinned = cs.pin()

    # Change the nested definitions after pinning. New parses through this cstruct instance must still use the
    # pinned version until the caller explicitly unpins.
    cs.entry.add_field("extra", cs.uint8)

    old = cs.record(RECORD_V1_BYTES)
    assert old.head.count == 2
    assert old.entries[1].id == 0x22
    assert old.payload.typed.length == 6
    assert old.dumps() == RECORD_V1_BYTES
    assert instance_schema(old) is pinned
    assert [field.name for field in old.entries[0].__class__.__fields__] == ["id", "flags"]

    # Explicitly unpinning returns to the documented default of the latest definitions.
    cs.unpin()
    new = cs.record(RECORD_V1_BYTES + b"	")
    assert new.entries[0].extra == 0x22


def test_recompile_forces_all_snapshot_structures() -> None:
    cs = _define(RECORD_V1)
    schema = cs.snapshot()

    # Compilation is lazy by default. Explicit recompilation makes the chosen boundary eager.
    schema.recompile()
    for name in ("header", "entry", "record"):
        assert schema.resolve(name).__compiled__
        assert hasattr(schema.resolve(name)._read, "__source__")

    # Recompilation doesn't change the values of the chosen version.
    record = schema.record(RECORD_V1_BYTES)
    assert record.dumps() == RECORD_V1_BYTES


def test_interleaved_versions_of_same_record() -> None:
    v1 = _define(RECORD_V1)
    v2 = _define(RECORD_V2)
    pinned_v1 = v1.snapshot()
    pinned_v2 = v2.snapshot()

    objects = []
    for _ in range(3):
        one = pinned_v1.record(RECORD_V1_BYTES)
        two = pinned_v2.record(RECORD_V2_BYTES)

        assert (one.head.count, one.entries[1].flags, one.payload.typed.length, len(one.body)) == (2, 2, 6, 6)
        assert (two.head.count, two.entries[1].flags, two.payload.typed.length, len(two.body)) == (256, 2, 12, 12)
        assert one.dumps() == RECORD_V1_BYTES
        assert two.dumps() == RECORD_V2_BYTES
        objects.append((one, two))

    # All earlier instances remain stable after both definitions changed.
    v1.entry.add_field("new_field", v1.uint16)
    v2.body = v2.resolve("uint8")  # no-op, just exercise attribute access
    for one, two in objects:
        assert one.dumps() == RECORD_V1_BYTES
        assert two.dumps() == RECORD_V2_BYTES
        assert instance_schema(one) is pinned_v1
        assert instance_schema(two) is pinned_v2


def test_nested_change_is_a_new_version() -> None:
    cs = _define(RECORD_V1)
    first = cs.snapshot()
    first_record = first.record(RECORD_V1_BYTES)

    # A nested field change produces a new definition version, even though the outer class object stays the same.
    cs.entry.add_field("extra", cs.uint8)

    second = cs.snapshot()
    assert second.revision > first.revision
    assert second.digest != first.digest
    assert first_record.dumps() == RECORD_V1_BYTES
    assert [field.name for field in second.entry.__fields__] == ["id", "flags", "extra"]

    # The old version still parses exactly as before.
    assert first.record(RECORD_V1_BYTES).dumps() == RECORD_V1_BYTES
    assert first.record(RECORD_V1_BYTES).entries[0].id == 0x11


def test_stream_parse_keeps_its_version() -> None:
    cs = _define(RECORD_V1)
    schema = cs.snapshot()
    stream = io.BytesIO(RECORD_V1_BYTES)

    record = schema.record(stream)
    assert stream.tell() == len(RECORD_V1_BYTES)
    assert record.dumps() == RECORD_V1_BYTES
    assert instance_schema(record) is schema


def test_mid_parse_definition_change_fails_closed() -> None:
    """A definition change during a parse must not let outer and nested fields use different versions."""
    from typing import BinaryIO

    from dissect.cstruct.exception import Error
    from dissect.cstruct.types.structure import StructureMetaType

    cs = _define(
        """
        struct nested { uint8 a; };
        struct outer { nested n; uint32 tail; };
        """
    )
    original = StructureMetaType._read
    changed = False

    def read_with_change(
        cls: type[BaseType], stream: BinaryIO, *, context: dict | None = None, endian: str
    ) -> BaseType:
        # Mutate the definition in the middle of the outer read: after "nested" was read, before "tail".
        nonlocal changed
        if not changed:
            changed = True
            cs.outer.add_field("extra", cs.uint8)
        return original(cls, stream, context=context, endian=endian)

    # Patch only the nested class read to change the definition once entered.
    with pytest.MonkeyPatch.context() as monkey:
        monkey.setattr(cs.nested, "_read", classmethod(read_with_change))
        with pytest.raises(Error, match="Definitions changed during an active parse"):
            cs.outer(b"\x01\x02\x03\x04\x05")

    # The failed parse didn't silently produce a mixed object. Once the parse is over, the (unchanged) definition
    # is still usable.
    assert cs.outer(b"\x01\x02\x03\x04\x05").dumps() == b"\x01\x02\x03\x04\x05"


def test_top_level_array_of_records_keeps_version() -> None:
    cs = _define(RECORD_V1)
    schema = cs.snapshot()

    records = schema.record[2](RECORD_V1_BYTES + RECORD_V1_BYTES)
    assert [record.head.count for record in records] == [2, 2]
    assert records[0].dumps() == RECORD_V1_BYTES

    # Change the nested element type after the array was parsed.
    cs.entry.add_field("extra", cs.uint8)
    assert records[1].dumps() == RECORD_V1_BYTES
    assert [field.name for field in records[0].entries[0].__class__.__fields__] == ["id", "flags"]

    # A new array through the same snapshot still uses the snapshot's original definitions.
    again = schema.record[2](RECORD_V1_BYTES + RECORD_V1_BYTES)
    assert again[1].dumps() == RECORD_V1_BYTES


def test_nested_array_element_redefinition_does_not_leak() -> None:
    cs = _define(
        """
        struct cell { uint8 a; uint8 b; };
        struct grid { cell rows[2]; uint8 matrix[2][2]; };
        """
    )
    schema = cs.snapshot()
    grid = schema.grid(b"\x01\x02\x03\x04\x05\x06\x07\x08")

    # Replace the element type with a wider one. Existing values and the snapshot keep the two-byte stride.
    cs.add_type("cell", cstruct().load("struct cell { uint32 a; };").cell, replace=True)
    cs.grid.commit()

    assert grid.dumps() == b"\x01\x02\x03\x04\x05\x06\x07\x08"
    assert grid.rows[1].b == 4
    assert grid.matrix[1] == [7, 8]
    assert [field.name for field in grid.rows[0].__class__.__fields__] == ["a", "b"]
    assert schema.grid(b"\x01\x02\x03\x04\x05\x06\x07\x08").dumps() == (b"\x01\x02\x03\x04\x05\x06\x07\x08")


def test_change_between_capture_and_read_fails_closed() -> None:
    """A parse that already captured a version must not silently read new fields with old offsets."""
    from dissect.cstruct.exception import Error

    cs = _define("struct record { uint8 a; uint8 b; };")
    context = cs._parse_context()
    schema = context.__enter__()

    # Change the definition inside the boundary, before any bytes have been read.
    with pytest.raises(Error, match="Definitions changed during an active parse"):
        cs.record.add_field("c", cs.uint8)
    context.__exit__(None, None, None)

    # The failed mutation didn't apply, and parsing still uses the captured version.
    assert [field.name for field in cs.record.__fields__] == ["a", "b"]
    assert schema.record(b"\x01\x02").dumps() == b"\x01\x02"


def test_default_top_level_array_of_records_keeps_version() -> None:
    cs = _define(RECORD_V1)
    records = cs.record[2](RECORD_V1_BYTES * 2)

    assert instance_schema(records) is not None
    assert all(instance_schema(record) is not None for record in records)

    cs.entry.add_field("extra", cs.uint8)
    assert records.dumps() == RECORD_V1_BYTES * 2
    assert records[1].dumps() == RECORD_V1_BYTES
