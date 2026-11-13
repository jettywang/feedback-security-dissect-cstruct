from __future__ import annotations

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dissect.cstruct.bitbuffer import BitBuffer
from dissect.cstruct.exception import LayoutPreservationError
from dissect.cstruct.types.base import BaseArray
from dissect.cstruct.types.structure import (
    Structure,
    StructureMetaType,
    Union,
    UnionMetaType,
    get_fidelity_buffer,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dissect.cstruct.cstruct import Endianness
    from dissect.cstruct.types.structure import Field

# Region ownership labels
FIELD = "field"
PADDING = "padding"
UNION = "union"
BITFIELD = "bitfield"


@dataclass(slots=True)
class ByteChange:
    """A single byte that differs from the original record.

    Args:
        offset: Absolute byte offset within the top-level structure.
        old: The original byte value.
        new: The modified byte value.
        path: The dotted path of the field responsible for this byte (``""`` for the top-level union).
        region: Who owns the byte: ``field``, ``bitfield``, ``union`` or ``padding``.
    """

    offset: int
    old: int
    new: int
    path: str
    region: str


@dataclass(slots=True)
class Region:
    """A contiguous byte span and the field that owns it.

    Args:
        start: Absolute start offset within the top-level structure.
        size: Size of the region in bytes.
        kind: One of ``field``, ``bitfield``, ``union`` or ``padding``.
        path: The dotted path of the owning field (``""`` for the top-level union or tail padding).
    """

    start: int
    size: int
    kind: str
    path: str


@dataclass(slots=True)
class FidelityResult:
    """The outcome of a fidelity edit.

    Args:
        data: The patched record, with every unmodified byte left exactly as it was in the original.
        changes: The byte-level differences caused by the edit. Padding is never part of this list.
        regions: A map describing who owns every byte span of the structure layout.
    """

    data: bytes
    changes: list[ByteChange]
    regions: list[Region]

    @property
    def changed_offsets(self) -> list[int]:
        return [change.offset for change in self.changes]

    def diff(self) -> str:
        """Render a human-readable, byte-by-byte comparison against the original layout."""
        lines = []
        for change in self.changes:
            lines.append(
                f"[{change.offset:>4}] {change.region:<8} {change.path or '<root>'}: "
                f"0x{change.old:02x} -> 0x{change.new:02x}"
            )
        return "\n".join(lines)


class _State:
    """Mutable bookkeeping carried through a single fidelity edit."""

    def __init__(self, buf: bytearray, endian: Endianness):
        self.buf = buf
        self.endian = endian
        self.changes: list[ByteChange] = []
        self.regions: list[Region] = []
        # Lookup of already recorded changes by offset, so a byte touched by multiple fields is reported once
        self._by_offset: dict[int, ByteChange] = {}


def _ensure_static(type_: type, path: str) -> None:
    """Reject any type whose encoded length depends on runtime data, so bytes can never shift."""
    if isinstance(type_, StructureMetaType):
        if type_.__size__ is None or type_.__dynamic__:
            raise LayoutPreservationError(
                f"Cannot perform an in-place fidelity edit: field {path or '<root>'!r} is a dynamically sized "
                "structure; changing it would move the bytes that follow it"
            )
        for field in type_.__fields__:
            _ensure_static(field.type, f"{path}.{field._name}" if path else field._name)
        return

    if issubclass(type_, BaseArray):
        if type_.__dynamic__ or not isinstance(type_.__num_entries__, int):
            raise LayoutPreservationError(
                f"Cannot perform an in-place fidelity edit: field {path!r} is a dynamically sized array"
            )
        _ensure_static(type_.__type__, f"{path}[]")


def _write_value(type_: type, value: Any, endian: Endianness) -> bytes:
    out = io.BytesIO()
    if value is None:
        value = type_.__default__()
    type_._write(out, value, endian=endian)
    return out.getvalue()


def _diff(state: _State, start: int, new_bytes: bytes, path: str, kind: str) -> None:
    existing = state._by_offset
    for index, new in enumerate(new_bytes):
        offset = start + index
        old = state.buf[offset]
        if old == new:
            continue

        if (change := existing.get(offset)) is not None:
            # Multiple fields can share one byte (e.g. bit fields in one storage unit). Record a single
            # change for that byte instead of emitting one per contributing field.
            change.new = new
            if path not in change.path:
                change.path = f"{change.path},{path}"
            continue

        change = ByteChange(offset=offset, old=old, new=new, path=path, region=kind)
        state._by_offset[offset] = change
        state.changes.append(change)
    state.buf[start : start + len(new_bytes)] = new_bytes


def _emit_padding(state: _State, start: int, end: int, path: str) -> None:
    if end > start:
        state.regions.append(Region(start=start, size=end - start, kind=PADDING, path=path))


def _patch_array(field: Field, value: Any, offset: int, path: str, state: _State) -> None:
    array_type = field.type
    expected = array_type.__num_entries__
    if not isinstance(value, (list, bytes, str)) or len(value) != expected:
        raise LayoutPreservationError(
            f"Cannot perform an in-place fidelity edit on fixed array {path!r}: it must keep exactly "
            f"{expected} entries, got {len(value) if hasattr(value, '__len__') else value!r}; "
            "growing or shrinking it would move the surrounding bytes"
        )

    element_type = array_type.__type__
    if isinstance(element_type, StructureMetaType):
        element_size = len(element_type)
        for index, element in enumerate(value):
            _patch_struct_like(element_type, element, offset + index * element_size, f"{path}[{index}]", state)
        state.regions.append(Region(start=offset, size=array_type.__size__, kind=FIELD, path=path))
        return

    new_bytes = _write_value(array_type, value, state.endian)
    state.regions.append(Region(start=offset, size=len(new_bytes), kind=FIELD, path=path))
    _diff(state, offset, new_bytes, path, FIELD)


def _patch_union(union_type: type[Union], value: Union, offset: int, path: str, state: _State) -> None:
    size = union_type.__size__
    new_bytes = _write_value(union_type, value, state.endian)
    if len(new_bytes) != size:
        raise LayoutPreservationError(
            f"Cannot perform an in-place fidelity edit on union {path!r}: encoded size changed from {size} "
            f"to {len(new_bytes)}"
        )

    # The whole union window is owned by the union; overlapping members are reported at union granularity.
    state.regions.append(Region(start=offset, size=size, kind=UNION, path=path))
    _diff(state, offset, new_bytes, path, UNION)


def _patch_bitfield_group(
    cls: type, group: list[Field], obj: Structure, offset: int, path_prefix: str, state: _State
) -> None:
    bit_type = group[0].type
    storage_size = bit_type.__size__
    member_names = ",".join(field._name for field in group)
    group_path = f"{path_prefix}.{{{member_names}}}" if path_prefix else f"{{{member_names}}}"

    out = io.BytesIO()
    buffer = BitBuffer(out, endian=state.endian)
    for field in group:
        member_value = getattr(obj, field._name)
        if member_value is None:
            member_value = field.type.__default__()
        buffer.write(field.type, member_value, field.bits)
    buffer.flush()
    new_bytes = out.getvalue()

    if len(new_bytes) != storage_size:
        raise LayoutPreservationError(
            f"Cannot perform an in-place fidelity edit on bit field group {group_path!r}: storage size changed"
        )

    state.regions.append(Region(start=offset, size=storage_size, kind=BITFIELD, path=group_path))
    _diff(state, offset, new_bytes, group_path, BITFIELD)


def _patch_structure(cls: type, obj: Structure, base: int, path: str, state: _State) -> None:
    cursor = base
    fields = cls.__fields__
    index = 0

    while index < len(fields):
        field = fields[index]
        field_offset = base + (field.offset or 0)
        field_path = f"{path}.{field._name}" if path else field._name

        # Alignment / manual padding that precedes this field is owned by no field and is preserved as-is.
        _emit_padding(state, cursor, field_offset, path)

        if field.bits:
            # Consecutive bit fields share a storage unit (see BitBuffer); the leading field has the offset
            group = [field]
            storage_type = field.type
            storage_offset = field_offset
            index += 1
            while index < len(fields) and fields[index].bits:
                group.append(fields[index])
                index += 1
            _patch_bitfield_group(cls, group, obj, storage_offset, path, state)
            cursor = storage_offset + storage_type.__size__
            continue

        value = getattr(obj, field._name)
        field_type = field.type

        if isinstance(field_type, UnionMetaType):
            _patch_union(field_type, value, field_offset, field_path, state)
        elif isinstance(field_type, StructureMetaType):
            _patch_structure(field_type, value, field_offset, field_path, state)
        elif issubclass(field_type, BaseArray):
            _patch_array(field, value, field_offset, field_path, state)
        else:
            new_bytes = _write_value(field_type, value, state.endian)
            state.regions.append(Region(start=field_offset, size=len(new_bytes), kind=FIELD, path=field_path))
            _diff(state, field_offset, new_bytes, field_path, FIELD)

        cursor = field_offset + len(field_type)
        index += 1

    # Trailing padding added so the structure matches its alignment.
    _emit_padding(state, cursor, base + len(cls), path)


def _patch_struct_like(cls: type, obj: Any, base: int, path: str, state: _State) -> None:
    if isinstance(cls, UnionMetaType):
        _patch_union(cls, obj, base, path, state)
    else:
        _patch_structure(cls, obj, base, path, state)


def _resolve_original(obj: Structure, explicit: bytes | bytearray | memoryview | None) -> bytes:
    if explicit is not None:
        return bytes(explicit)

    captured = get_fidelity_buffer(obj)
    if captured is not None:
        return bytes(captured)

    if isinstance(obj, Union) and getattr(obj, "_buf", None) is not None:
        return bytes(obj._buf)

    raise LayoutPreservationError(
        "No original bytes available for a fidelity edit: pass the original record explicitly as "
        "fidelity_edit(obj, original=...) (the object was not parsed from a seekable byte record)"
    )


def _apply_changes(obj: Structure, changes: Mapping[str, Any] | None) -> None:
    if not changes:
        return

    for dotted_path, value in changes.items():
        target = obj
        parts = dotted_path.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)


def fidelity_edit(
    obj: Structure,
    changes: Mapping[str, Any] | None = None,
    *,
    original: bytes | bytearray | memoryview | None = None,
    endian: Endianness | None = None,
) -> FidelityResult:
    """Apply field changes while preserving every unmodified byte of the original record.

    Unlike :meth:`Structure.dumps`, which re-serializes the whole object and therefore rewrites padding
    and non-active union bytes with zeroes, this function patches a copy of the original bytes in place.
    Only the byte windows owned by the fields that actually changed are rewritten; alignment padding,
    anonymous union storage and every other byte retain their original value.

    A fidelity edit can never move bytes. If the structure (or anything nested in it) is dynamically
    sized, or a fixed array would grow or shrink, :class:`LayoutPreservationError` is raised instead of
    silently rearranging the record.

    Args:
        obj: A structure or union parsed from the original record.
        changes: Optional mapping of dotted field paths to new values, applied before patching.
        original: The original bytes. Required when ``obj`` was not parsed from a seekable record.
        endian: Endianness to encode with. Defaults to the endianness used when parsing.

    Returns:
        A :class:`FidelityResult` containing the patched bytes, the per-byte changes and the layout map.
    """
    cls = obj.__class__
    if not isinstance(cls, StructureMetaType):
        raise TypeError(f"fidelity_edit expects a Structure or Union, got {cls.__name__}")

    _ensure_static(cls, "")
    _apply_changes(obj, changes)

    source = _resolve_original(obj, original)
    expected_size = len(cls)
    if len(source) != expected_size:
        raise LayoutPreservationError(
            f"Original record is {len(source)} bytes but {cls.__name__} is {expected_size} bytes; "
            "an in-place fidelity edit requires an exact-length original record"
        )

    if endian is None:
        endian = getattr(obj, "_endian", None) or cls.__cs__.endian

    state = _State(bytearray(source), endian)
    _patch_struct_like(cls, obj, 0, "", state)

    return FidelityResult(data=bytes(state.buf), changes=state.changes, regions=state.regions)
