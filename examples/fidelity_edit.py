"""Forensic fidelity editing: change one field while preserving every other byte.

Normal reserialization (``obj.dumps()``) rebuilds a whole record and therefore rewrites alignment padding and
inactive union bytes. A fidelity edit instead patches a copy of the original bytes in place and refuses any change
that would move bytes. Run it with ``python examples/fidelity_edit.py``.
"""

from dissect.cstruct import LayoutPreservationError, cstruct

cs = cstruct().load(
    """
    struct record {
        uint8   tag;    // offset 0x00
        uint32  value;  // offset 0x04 (padding at 0x01-0x03)
        uint16  n;      // offset 0x08 (tail padding at 0x0a-0x0b)
    };

    struct chunk {
        uint16 count;
        uint8  data[4];  // fixed array: must keep exactly four entries for an in-place edit
    };
    """,
    align=True,
)

print("=== 1. Change a single integer in a record with non-zero padding ===")
original = bytes.fromhex("AA 11 22 33 BB CC CC CC DD DD EE EE")
record = cs.record(original)
record.value = 0x11223344

reserialized = record.dumps()
fidelity = cs.record(original).fidelity_edit({"value": 0x11223344})

print(f"original bytes : {original.hex(' ')}")
print(f"dumps() bytes  : {reserialized.hex(' ')}  (padding zeroed, evidence altered)")
print(f"fidelity bytes : {fidelity.data.hex(' ')}  (only `value` changed)")
print()
print("Byte-by-byte attribution of the fidelity edit:")
print(fidelity.diff())
print()
print("Regions of the layout (padding is owned by no field and is never rewritten):")
for region in fidelity.regions:
    print(f"  offset {region.start:#04x} size {region.size:<2} {region.kind:<8} {region.path or '<tail/alignment>'}")

print()
print("=== 2. Growing a fixed array must be rejected, never silently rearranged ===")
chunk = cs.chunk(bytes.fromhex("02 00 01 02 03 04"))
try:
    chunk.fidelity_edit({"data": b"\x01\x02\x03\x04\x05"})  # 5 entries into a 4-entry field
except LayoutPreservationError as err:
    print(f"rejected: {err}")
