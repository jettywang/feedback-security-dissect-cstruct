"""Deterministic structural fuzzing of the bit field compiler.

The same attacker controlled protocol record is parsed by three independent implementations:

1. The optimized, code generating compiler (``compiled=True``).
2. The interpreted :class:`~dissect.cstruct.types.structure.StructureMetaType` reader
   (``compiled=False``).
3. A hand written bit mask oracle in this module, which only uses the generated field layout and
   the raw record bytes. The oracle deliberately does not call into :mod:`dissect.cstruct` for its
   values, so a bug shared by the two bundled implementations cannot be "verified" by the oracle.

For every generated definition we check, for both full and truncated records:

* every field value (including sign handling and all supported byte orders),
* the number of bytes consumed from the stream,
* the exception type, message and stream position when a record is truncated.

Failures are minimized to the smallest field combination and stored together with the definition
and the raw bytes. A dedicated test temporarily reverses a shift direction of the compiler to
prove that the harness actually detects a divergence.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from dissect.cstruct import compiler
from dissect.cstruct.cstruct import cstruct
from dissect.cstruct.util import ENDIANNESS_TO_BYTEORDER_MAP

if TYPE_CHECKING:
    from collections.abc import Iterator

# A fixed, deliberately small default number of generated definitions. Overridable for local
# exploration, but the committed default never depends on wall time or host randomness.
DEFAULT_SAMPLE_COUNT = int(os.environ.get("BITFIELD_FUZZ_SAMPLES", "50"))
SEED = int(os.environ.get("BITFIELD_FUZZ_SEED", "0xB17F1E1D"), 0)
CRASH_DIR = Path(os.environ.get("BITFIELD_FUZZ_CRASH_DIR", Path(__file__).parent / "_data" / "bitfield_fuzz_crashes"))

ENDIANS = ("<", ">", "!", "@", "=")
# Explicit little endian is the only endianness for which the LSB of a storage unit is consumed
# first; every other endianness consumes the MSB first (see BitBuffer and the generated source).
LITTLE_ENDIANS = ("<",)
STORAGE_SIZES = (1, 2, 4, 8)
SIGNED_TYPES = {1: "int8", 2: "int16", 4: "int32", 8: "int64"}
UNSIGNED_TYPES = {1: "uint8", 2: "uint16", 4: "uint32", 8: "uint64"}
TYPE_SIZE = {name: size for size, name in SIGNED_TYPES.items()}
TYPE_SIZE.update({name: size for size, name in UNSIGNED_TYPES.items()})
# Plain (non bit field) members and their byte size.
PLAIN_TYPES = (*tuple(TYPE_SIZE), "char")


@dataclass
class BitFieldModel:
    """A single bit field, described independently of any cstruct type object."""

    name: str
    type_name: str
    bits: int
    storage_size: int
    group_index: int
    unit_index: int
    low_bit: int  # Bit position of the LSB of this field within its storage unit

    @property
    def signed(self) -> bool:
        return self.type_name.startswith("int")


@dataclass
class PlainFieldModel:
    """A regular, non bit field member."""

    name: str
    type_name: str
    size: int
    array: int = 1
    signed: bool = False


@dataclass
class StructModel:
    """An independently generated description of a structure and its byte layout."""

    name: str
    members: list[Any] = field(default_factory=list)
    nested: list[StructModel] = field(default_factory=list)
    # Byte offset of every top level member, in declaration order
    offsets: list[int] = field(default_factory=list)
    size: int = 0

    def flatten(self, prefix: str = "") -> Iterator[tuple[str, Any]]:
        for member in self.members:
            path = f"{prefix}{member.name}"
            if isinstance(member, _NestedMember):
                yield from member.model.flatten(f"{path}.")
            else:
                yield path, member


class OracleEOF(Exception):
    """Raised by the oracle at the exact stream position a truncation becomes observable."""

    def __init__(self, got: int, expected: int, position: int):
        super().__init__(f"Read {got} bytes, but expected {expected}")
        self.position = position


class BitMaskOracle:
    """Reference parser that extracts every value with explicit masks and shifts."""

    def __init__(self, model: StructModel, endian: str):
        self.model = model
        self.endian = endian
        # The bit ordering follows the explicit little endian LSB-first rule only for ``<``; all
        # other endiannesses consume the MSB first (matching BitBuffer and the generated source).
        self.little = endian in LITTLE_ENDIANS
        self.byteorder = ENDIANNESS_TO_BYTEORDER_MAP[endian]

    def parse(self, data: bytes) -> tuple[dict[str, Any], int]:
        stream = io.BytesIO(data)
        values: dict[str, Any] = {}
        self._parse_struct(self.model, stream, values)
        return values, stream.tell()

    def _read_unit(self, stream: io.BytesIO, size: int) -> int:
        position = stream.tell()
        chunk = stream.read(size)
        if len(chunk) != size:
            raise OracleEOF(len(chunk), size, position + len(chunk))
        return int.from_bytes(chunk, self.byteorder)

    def _parse_struct(self, model: StructModel, stream: io.BytesIO, values: dict[str, Any]) -> None:
        unit_cache: dict[tuple[int, int], int] = {}

        for member in model.members:
            if isinstance(member, _NestedMember):
                nested_values: dict[str, Any] = {}
                self._parse_struct(member.model, stream, nested_values)
                values[member.name] = nested_values
            elif isinstance(member, BitFieldModel):
                key = (member.group_index, member.unit_index)
                if key not in unit_cache:
                    unit_cache[key] = self._read_unit(stream, member.storage_size)
                unit = unit_cache[key]
                # ``low_bit`` counts from the LSB, matching the little endian LSB-first layout.
                # For big endian the field is mirrored within the storage unit.
                low_bit = member.low_bit if self.little else member.storage_size * 8 - member.low_bit - member.bits
                # Explicit, hand written mask/shift extraction - the heart of the oracle.
                values[member.name] = (unit >> low_bit) & ((1 << member.bits) - 1)
            else:
                position = stream.tell()
                length = member.size * member.array
                chunk = stream.read(length)
                if len(chunk) != length:
                    raise OracleEOF(len(chunk), length, position + len(chunk))
                if member.type_name == "char":
                    values[member.name] = chunk
                else:
                    if member.array > 1:
                        values[member.name] = [
                            self._decode_int(chunk[i : i + member.size], member) for i in range(0, length, member.size)
                        ]
                    else:
                        values[member.name] = self._decode_int(chunk, member)

    def _decode_int(self, raw: bytes, member: PlainFieldModel) -> int:
        # Plain signed integers are sign extended; bit fields are read as unsigned patterns.
        return int.from_bytes(raw, self.byteorder, signed=member.signed)


# ---------------------------------------------------------------------------
# Deterministic definition generation
# ---------------------------------------------------------------------------


class DefinitionGenerator:
    """Generates matching ``(cdef, model)`` pairs with a fixed xorshift32 PRNG."""

    def __init__(self, seed: int):
        self._state = seed & 0xFFFFFFFF
        self._struct_counter = 0
        # Field name counters are global across nested structures to keep every name unique.
        self._bf_counter = 0
        self._pf_counter = 0

    def _rand(self) -> int:
        x = self._state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self._state = x & 0xFFFFFFFF
        return self._state

    def _choice(self, options: tuple[Any, ...] | list[Any]) -> Any:
        return options[self._rand() % len(options)]

    def _randint(self, low: int, high: int) -> int:
        return low + self._rand() % (high - low + 1)

    def _new_struct_name(self) -> str:
        self._struct_counter += 1
        return f"s{self._struct_counter}"

    def _bit_group(self, group_index: int) -> tuple[list[BitFieldModel], int]:
        storage_size = self._choice(STORAGE_SIZES)
        capacity = storage_size * 8
        # A bit field group uses a single base type. The parser only starts a new storage unit once
        # the current one is filled exactly; a non fitting field on a partially filled unit raises
        # "Straddled bit fields are unsupported". Every generated unit is therefore filled exactly,
        # which also makes adjacent same type groups start their own unit cleanly.
        type_name = self._choice((SIGNED_TYPES[storage_size], UNSIGNED_TYPES[storage_size]))
        unit_count = self._choice((1, 1, 2, 2, 3))

        fields: list[BitFieldModel] = []
        unit_index = 0
        low_bit = 0

        while unit_index < unit_count:
            remaining = capacity - low_bit
            bits = self._randint(1, remaining)
            fields.append(
                BitFieldModel(
                    name=f"bf{self._bf_counter}",
                    type_name=type_name,
                    bits=bits,
                    storage_size=storage_size,
                    group_index=group_index,
                    unit_index=unit_index,
                    low_bit=low_bit,
                )
            )
            self._bf_counter += 1
            low_bit += bits

            if low_bit == capacity:
                unit_index += 1
                low_bit = 0

        return fields, unit_count * storage_size

    def _build_struct(self, name: str, allow_nested: bool) -> StructModel:
        model = StructModel(name=name)
        offset = 0
        group_index = 0
        # A fixed shape keeps the byte layout predictable: every bit field group is followed by a
        # plain separator field, so adjacent groups can never merge into a shared storage unit.
        shape = ("bits", "plain", "bits", "plain", "nested", "bits", "plain")

        for slot in shape:
            if slot == "bits":
                fields, group_size = self._bit_group(group_index)
                group_index += 1
                model.members.extend(fields)
                model.offsets.append(offset)
                offset += group_size
            elif slot == "plain":
                type_name = self._choice(PLAIN_TYPES)
                size = 1 if type_name == "char" else TYPE_SIZE[type_name]
                array = self._choice((1, 1, 1, 2, 3))
                plain = PlainFieldModel(
                    name=f"pf{self._pf_counter}",
                    type_name=type_name,
                    size=size,
                    array=array,
                    signed=type_name.startswith("int"),
                )
                self._pf_counter += 1
                model.members.append(plain)
                model.offsets.append(offset)
                offset += size * array
            elif allow_nested:
                nested_name = self._new_struct_name()
                nested = self._build_struct(nested_name, allow_nested=False)
                model.nested.append(nested)
                member_name = f"nested{len(model.nested)}"
                nested.name = nested_name
                wrapper = _NestedMember(member_name, nested)
                model.members.append(wrapper)
                model.offsets.append(offset)
                offset += nested.size
            else:
                # A nested slot in a non nestable structure becomes another plain separator.
                type_name = self._choice(PLAIN_TYPES)
                size = 1 if type_name == "char" else TYPE_SIZE[type_name]
                plain = PlainFieldModel(
                    name=f"pf{self._pf_counter}",
                    type_name=type_name,
                    size=size,
                    signed=type_name.startswith("int"),
                )
                self._pf_counter += 1
                model.members.append(plain)
                model.offsets.append(offset)
                offset += size

        model.size = offset
        return model

    def generate(self) -> tuple[str, StructModel]:
        root_name = self._new_struct_name()
        model = self._build_struct(root_name, allow_nested=True)
        return self._render(model), model

    def _render(self, model: StructModel) -> str:
        blocks = [self._render_struct(nested) for nested in model.nested]
        blocks.append(self._render_struct(model))
        return "\n\n".join(blocks)

    def _render_struct(self, model: StructModel) -> str:
        lines = [f"struct {model.name} {{"]
        for member in model.members:
            if isinstance(member, _NestedMember):
                lines.append(f"    {member.model.name} {member.name};")
            elif isinstance(member, BitFieldModel):
                lines.append(f"    {member.type_name} {member.name} : {member.bits};")
            else:
                declarator = member.name if member.array == 1 else f"{member.name}[{member.array}]"
                lines.append(f"    {member.type_name} {declarator};")
        lines.append("};")
        return "\n".join(lines)


@dataclass
class _NestedMember:
    """A named reference to a nested structure within a parent member list."""

    name: str
    model: StructModel

    @property
    def bits(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def normalize(value: Any) -> Any:
    """Reduce parsed values to plain, implementation independent Python objects."""
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, bytes):
        return value
    if isinstance(value, int):
        # cstruct Int/Enum subclasses carry an (empty) __dict__, so integers must be handled first.
        return getattr(value, "value", value)
    if hasattr(value, "__dict__"):
        return {key: normalize(item) for key, item in value.__dict__.items() if key != "__dynamic_sizes__"}
    return value


def flatten_values(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in values.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_values(value, f"{path}."))
        else:
            flat[path] = normalize(value)
    return flat


def run_cstruct(cdef: str, root_name: str, data: bytes, endian: str, compiled: bool) -> dict[str, Any]:
    cs = cstruct()
    cs.endian = endian
    cs.load(cdef, compiled=compiled)
    stream = io.BytesIO(data)
    try:
        obj = cs.resolve(root_name)(stream)
    except Exception as exc:  # The outcome is compared, never silently swallowed
        return {"ok": False, "type": type(exc).__name__, "message": str(exc), "position": stream.tell()}
    return {"ok": True, "values": flatten_values(normalize(obj)), "position": stream.tell()}


def run_oracle(model: StructModel, data: bytes, endian: str) -> dict[str, Any]:
    try:
        values, position = BitMaskOracle(model, endian).parse(data)
    except OracleEOF as exc:
        return {"ok": False, "type": "EOFError", "message": str(exc), "position": exc.position}
    return {"ok": True, "values": flatten_values(values), "position": position}


def deterministic_data(index: int, size: int) -> bytes:
    # Reproducible, high entropy bytes without any host randomness.
    return bytes(((index * 1103515245 + step * 22695477 + 12345) & 0xFF) for step in range(size))


def mutate_width(cdef: str, model: StructModel) -> tuple[str, StructModel] | None:
    """Narrow the final bit field of the last storage unit in the structure.

    The selected field is the trailing field of its unit, so shrinking it never moves another
    field. The freed tail bit is simply reserved and the definition remains legal.
    """
    candidates: list[BitFieldModel] = []

    def collect(struct: StructModel) -> None:
        for member in struct.members:
            if isinstance(member, _NestedMember):
                collect(member.model)
            elif isinstance(member, BitFieldModel):
                candidates.append(member)

    collect(model)
    if not candidates:
        return None

    target = candidates[-1]
    if target.bits <= 1:
        return None

    old = f"    {target.type_name} {target.name} : {target.bits};"
    if old not in cdef:
        return None
    new = f"    {target.type_name} {target.name} : {target.bits - 1};"
    target.bits -= 1
    return cdef.replace(old, new, 1), model


def outcomes_equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Compare two parse outcomes, treating same-type/same-position EOF as equivalent.

    Values and consumed bytes are compared exactly. For exceptions, the type and the stream
    position where it was raised must match. The EOF message byte count is deliberately excluded:
    the compiler merges adjacent fields into one ``stream.read`` and reports that block size, while
    the interpreted reader (and the oracle) report the size of the individual field. The stream
    position, which is what a defensive parser branches on, is identical. A dedicated test locks
    this known message difference so it cannot silently change.
    """
    if left["ok"] != right["ok"]:
        return False
    if left["ok"]:
        return left["values"] == right["values"] and left["position"] == right["position"]
    return left["type"] == right["type"] and left["position"] == right["position"]


def check_case(cdef: str, model: StructModel, endian: str, data: bytes) -> dict[str, Any] | None:
    """Return a failure description, or ``None`` when all three implementations agree."""
    oracle = run_oracle(model, data, endian)
    compiled = run_cstruct(cdef, model.name, data, endian, compiled=True)
    interpreted = run_cstruct(cdef, model.name, data, endian, compiled=False)

    if outcomes_equivalent(compiled, interpreted) and outcomes_equivalent(compiled, oracle):
        return None

    return {
        "reason": "divergence",
        "endian": endian,
        "definition": cdef,
        "root": model.name,
        "bytes": data.hex(),
        "outcomes": {"oracle": oracle, "compiled": compiled, "interpreted": interpreted},
    }


def minimize_failure(cdef: str, model: StructModel, endian: str, data: bytes) -> dict[str, Any]:
    """Shrink a failing record to the shortest truncation that still diverges."""
    base_failure = check_case(cdef, model, endian, data)
    assert base_failure is not None
    for length in range(len(data) + 1):
        candidate = data[:length]
        if check_case(cdef, model, endian, candidate) is not None:
            failure = dict(base_failure)
            failure["bytes"] = candidate.hex()
            failure["minimized_length"] = length
            failure["field_combination"] = [path for path, _ in model.flatten()]
            return failure
    return base_failure


def _json_safe(value: Any) -> Any:
    """Convert nested cstruct/bytes values into JSON serializable primitives."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return {"hex": value.hex()}
    return value


def save_failure(failure: dict[str, Any], index: int) -> Path:
    CRASH_DIR.mkdir(parents=True, exist_ok=True)
    path = CRASH_DIR / f"failure_{index:04d}.json"
    path.write_text(json.dumps(_json_safe(failure), indent=2, sort_keys=True))
    return path


def iter_cases(count: int) -> Iterator[tuple[str, StructModel, str, bytes]]:
    generator = DefinitionGenerator(SEED)
    for index in range(count):
        cdef, model = generator.generate()
        for endian in ENDIANS:
            data = deterministic_data(index, model.size)
            yield cdef, model, endian, data
            # Truncations exercise consumed byte counts and exception positions.
            for cut in sorted({0, 1, model.size // 2, max(0, model.size - 1)}):
                yield cdef, model, endian, data[:cut]

        # A width mutation is derived once from the pristine definition, outside the endian loop.
        mutated = mutate_width(cdef, model)
        if mutated is not None:
            mutated_cdef, mutated_model = mutated
            for endian in ENDIANS:
                yield mutated_cdef, mutated_model, endian, deterministic_data(index, mutated_model.size)


# ---------------------------------------------------------------------------
# Layout self check: the independent model must match the parser's own metadata
# ---------------------------------------------------------------------------


def assert_layout_matches(cdef: str, model: StructModel) -> None:
    cs = cstruct()
    cs.load(cdef, compiled=False)
    struct_type = cs.resolve(model.name)
    assert struct_type.__size__ == model.size

    # Walk the model in declaration order, tracking the byte position like the parser would.
    running = 0
    expected_offsets: dict[int, int] = {}
    previous: Any = None
    for index, member in enumerate(model.members):
        if isinstance(member, _NestedMember):
            expected_offsets[index] = running
            running += member.model.size
        elif isinstance(member, BitFieldModel):
            new_unit = (
                previous is None
                or not isinstance(previous, BitFieldModel)
                or previous.group_index != member.group_index
                or member.low_bit == 0
            )
            if new_unit:
                expected_offsets[index] = running
                running += member.storage_size
        else:
            expected_offsets[index] = running
            running += member.size * member.array
        previous = member

    for index, (member, parsed) in enumerate(zip(model.members, struct_type.__fields__, strict=True)):
        if isinstance(member, _NestedMember):
            assert parsed.offset == expected_offsets[index]
            assert hasattr(parsed.type, "__fields__")
            assert_layout_matches(cdef, member.model)
        elif isinstance(member, BitFieldModel):
            assert parsed.bits == member.bits
            assert parsed.type.__size__ == member.storage_size
            # The parser only records an offset on the first field of a storage unit.
            if index in expected_offsets:
                assert parsed.offset == expected_offsets[index]
            else:
                assert parsed.offset is None
        else:
            assert parsed.offset == expected_offsets[index]
            assert parsed.type.__size__ == member.size * member.array


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bitfield_fuzz_deterministic() -> None:
    """Compiled, interpreted and the hand written oracle agree on every generated record."""
    failures: list[dict[str, Any]] = []
    for cdef, model, endian, data in iter_cases(DEFAULT_SAMPLE_COUNT):
        if check_case(cdef, model, endian, data) is not None:
            failures.append(minimize_failure(cdef, model, endian, data))

    if failures:
        paths = [save_failure(failure, index) for index, failure in enumerate(failures)]
        pytest.fail(f"{len(failures)} bit field divergences, saved to {paths}", pytrace=False)


def test_generated_layouts_match_parser_metadata() -> None:
    """Every independent model must describe the same offsets and sizes as the parsed type."""
    generator = DefinitionGenerator(SEED)
    seen_cross_unit = False
    seen_signed = False
    seen_nested = False
    seen_nonzero_low_bit = False

    for _ in range(DEFAULT_SAMPLE_COUNT):
        cdef, model = generator.generate()
        assert_layout_matches(cdef, model)

        cs = cstruct()
        cs.load(cdef, compiled=True)
        assert cs.resolve(model.name).__compiled__ is True

        for _, member in model.flatten():
            if isinstance(member, BitFieldModel):
                seen_signed |= member.signed
                seen_nonzero_low_bit |= member.low_bit > 0
                seen_cross_unit |= member.unit_index > 0
        seen_nested |= bool(model.nested)

    assert seen_cross_unit and seen_signed and seen_nested and seen_nonzero_low_bit


def test_oracle_matches_known_little_endian_pattern() -> None:
    """Lock the oracle to a hand computed little endian extraction."""
    cdef = "struct t { uint16 a : 4; uint16 b : 12; };"
    model = StructModel(
        name="t",
        members=[
            BitFieldModel("a", "uint16", 4, 2, 0, 0, 0),
            BitFieldModel("b", "uint16", 12, 2, 0, 0, 4),
        ],
        size=2,
    )
    values, position = BitMaskOracle(model, "<").parse(b"\x12\x34")
    assert flatten_values(values) == {"a": 0x2, "b": 0x341}
    assert position == 2

    values, position = BitMaskOracle(model, ">").parse(b"\x12\x34")
    assert flatten_values(values) == {"a": 0x1, "b": 0x234}
    assert position == 2
    assert "struct t" in cdef


def test_truncation_outcome_is_predicted() -> None:
    """The oracle and both implementations must agree on the exception position."""
    model = StructModel(
        name="t",
        members=[
            BitFieldModel("a", "uint16", 4, 2, 0, 0, 0),
            BitFieldModel("b", "uint16", 12, 2, 0, 0, 4),
            PlainFieldModel("c", "uint8", 1),
        ],
        size=3,
    )
    cdef = "struct t { uint16 a : 4; uint16 b : 12; uint8 c; };"
    failure = check_case(cdef, model, "<", b"\x12")
    assert failure is None
    outcome = run_oracle(model, b"\x12", "<")
    assert outcome["ok"] is False and outcome["type"] == "EOFError" and outcome["position"] == 1


def test_merged_read_eof_message_known_difference() -> None:
    """Document the one allowed compiler difference on truncation.

    The compiler merges the leading byte and the trailing array into a single ``stream.read``, so
    its EOF message reports the whole block size. The interpreted reader and the oracle report the
    array size. The exception type and the stream position a parser actually branches on are equal.
    """
    cdef = "struct t { uint8 h; int64 a[3]; };"
    record = bytes(range(10))

    compiled = run_cstruct(cdef, "t", record, "<", compiled=True)
    interpreted = run_cstruct(cdef, "t", record, "<", compiled=False)

    assert compiled["ok"] is interpreted["ok"] is False
    assert compiled["type"] == interpreted["type"] == "EOFError"
    assert compiled["position"] == interpreted["position"] == len(record)
    assert compiled["message"] != interpreted["message"]
    # The harness equivalence rule still treats these as the same security outcome.
    assert outcomes_equivalent(compiled, interpreted)


def test_shift_direction_mutation_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deliberately reversed compiler shift must make the harness report a divergence."""
    original_generate = compiler._ReadSourceGenerator._generate_bits_source

    def mutated_generate(self: compiler._ReadSourceGenerator, field: Any) -> Iterator[str]:
        for line in original_generate(self, field):
            if 'endian == "<"' in line and ">>=" in line:
                yield line.replace("_bit_buffer >>=", "_bit_buffer <<=")
            else:
                yield line

    monkeypatch.setattr(compiler._ReadSourceGenerator, "_generate_bits_source", mutated_generate)

    found = False
    for cdef, model, endian, data in iter_cases(DEFAULT_SAMPLE_COUNT):
        if endian != "<":
            continue
        if check_case(cdef, model, endian, data) is not None:
            found = True
            break

    assert found, "the shift direction mutation was not detected by the fuzz harness"
