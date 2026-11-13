"""Pytest coverage for the target-ABI layout experiment.

These tests pin the hand-derived offsets and total sizes for both target ABIs
and all three modelling policies, and prove compiled and interpreted reads
behave identically. They deliberately never use host ``ctypes``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from abi_layout import ABI_ILP32, ABI_LP64, layout  # noqa: E402
from byteprobe import (  # noqa: E402
    build_buffer,
    check_round_trip,
    expected_int,
    leaf_values,
    padding_offsets,
)
from cstruct_models import POLICIES, build_model, cstruct_field_offsets  # noqa: E402

# Hand-derived ground truth: policy -> (Rec size, Block size, field offsets).
TRUTH = {
    ABI_ILP32.pointer_size: {
        "packed":   (14, 28, {"magic": 0, "next": 4, "__anonymous_0__": 8, "count": 12}),
        "natural":  (16, 32, {"magic": 0, "next": 4, "__anonymous_0__": 8, "count": 12}),
        "explicit": (16, 32, {"magic": 0, "next": 4, "__anonymous_0__": 8, "count": 12}),
    },
    ABI_LP64.pointer_size: {
        "packed":   (18, 36, {"magic": 0, "next": 4, "__anonymous_0__": 12, "count": 16}),
        "natural":  (24, 48, {"magic": 0, "next": 8, "__anonymous_0__": 16, "count": 20}),
        "explicit": (24, 48, {"magic": 0, "next": 8, "__anonymous_0__": 16, "count": 20}),
    },
}

ABIS = (ABI_ILP32, ABI_LP64)
WATCHED = (
    "magic",
    "next",
    "__anonymous_0__.type",
    "__anonymous_0__.info.id",
    "__anonymous_0__.info.flags",
    "count",
)


def _engine_offsets(result):
    rec = result.node.children[0].children[0]
    return {
        (child.name or "__anonymous_0__"): child.offset
        for child in rec.children
        if child.kind != "pad"
    }


@pytest.mark.parametrize("abi", ABIS, ids=lambda a: str(a.pointer_size * 8))
@pytest.mark.parametrize("policy", POLICIES)
def test_engine_matches_hand_derived_offsets(abi, policy):
    result = layout(abi, policy)
    rec_size, block_size, offsets = TRUTH[abi.pointer_size][policy]
    rec = result.node.children[0].children[0]
    assert rec.size == rec_size
    assert result.size == block_size
    assert _engine_offsets(result) == offsets


@pytest.mark.parametrize("abi", ABIS, ids=lambda a: str(a.pointer_size * 8))
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("compiled", [False, True], ids=["interpreted", "compiled"])
def test_cstruct_matches_engine_and_execution_paths_agree(abi, policy, compiled):
    result = layout(abi, policy)
    model = build_model(abi, policy, compiled=compiled)
    actual = {
        name: offset
        for name, (offset, _size) in cstruct_field_offsets(model).items()
        if not name.startswith("pad")
    }
    assert actual == _engine_offsets(result)
    assert len(model.rec) == result.node.children[0].children[0].size
    assert len(model.block) == result.size


@pytest.mark.parametrize("abi", ABIS, ids=lambda a: str(a.pointer_size * 8))
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("compiled", [False, True], ids=["interpreted", "compiled"])
def test_second_array_element_offsets_and_values(abi, policy, compiled):
    result = layout(abi, policy)
    model = build_model(abi, policy, compiled=compiled)
    buf = build_buffer(result)
    check_round_trip(result, model, bytes(buf))

    obj = model.block(__import__("io").BytesIO(bytes(buf)))
    stride = result.node.children[0].children[0].size
    values = leaf_values(obj, 1)
    walked = {path: absolute for path, _, absolute in result.walk()}

    for leaf in WATCHED:
        path = f"Block.item[1].{leaf}"
        assert walked[path] == stride + walked[path.replace("[1]", "[0]")]
        flat = leaf.replace("__anonymous_0__.", "").replace("info.", "info.")
        assert values[flat] == expected_int(bytes(buf), result, path)


def test_second_element_absolute_offsets_are_explicit():
    # The headline check requested in the task: item[1] start per ABI/policy.
    expected_starts = {
        (4, "packed"): 14, (4, "natural"): 16, (4, "explicit"): 16,
        (8, "packed"): 18, (8, "natural"): 24, (8, "explicit"): 24,
    }
    for abi in ABIS:
        for policy in POLICIES:
            result = layout(abi, policy)
            stride = result.node.children[0].children[0].size
            assert stride == expected_starts[(abi.pointer_size, policy)]
            second_magic = next(
                absolute for path, _, absolute in result.walk()
                if path == "Block.item[1].magic"
            )
            assert second_magic == stride


def test_natural_implicit_padding_is_classified():
    result = layout(ABI_LP64, "natural")
    pads = padding_offsets(result)
    # 4 bytes before each pointer + 2 tail bytes per Rec, two elements.
    assert pads == {4, 5, 6, 7, 22, 23, 28, 29, 30, 31, 46, 47}
    assert layout(ABI_LP64, "packed") and padding_offsets(layout(ABI_LP64, "packed")) == set()


def test_explicit_padding_reproduces_natural_byte_for_byte():
    for abi in ABIS:
        natural = layout(abi, "natural")
        explicit = layout(abi, "explicit")
        assert explicit.size == natural.size
        nat_buf = build_buffer(natural)
        exp_buf = build_buffer(explicit)
        # Field bytes land at identical offsets; only pad ownership differs.
        nat_pads = padding_offsets(natural)
        for offset in range(natural.size):
            if offset not in nat_pads:
                assert nat_buf[offset] == exp_buf[offset]


def test_host_default_pointer_trap_is_demonstrable():
    from dissect.cstruct import cstruct

    definition = """
    struct Inner { uint16 id; uint16 flags; };
    struct Rec {
        uint32 magic;
        void*  next;
        union { uint32 type; struct Inner info; uint32 raw; };
        uint16 count;
    };
    """
    wrong = cstruct()  # host default pointer width
    right = cstruct(pointer="uint32")
    wrong.load(definition, align=True)
    right.load(definition, align=True)

    firmware = bytearray(24)
    firmware[0:4] = (0x11111111).to_bytes(4, "little")
    firmware[4:8] = (0x22222222).to_bytes(4, "little")
    firmware[8:12] = (0x33333333).to_bytes(4, "little")
    firmware[12:14] = (0x4444).to_bytes(2, "little")

    good = right.Rec(bytes(firmware))
    assert int(good.next) == 0x22222222
    assert int(good.type) == 0x33333333
    assert int(good.count) == 0x4444

    if sys.maxsize > 2**32:  # trap only bites on a 64-bit analysis host
        bad = wrong.Rec(bytes(firmware))
        assert int(bad.count) != 0x4444
        assert len(bad) == 24


def test_c_reference_agrees_with_engine():
    """Optional end-to-end check: real C compiler layout vs Python engine."""
    import shutil
    import compare_c_reference

    if shutil.which("cc") is None:
        pytest.skip("no C compiler available")
    assert compare_c_reference.main() == 0
