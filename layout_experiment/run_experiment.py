"""End-to-end reproduction and layout report.

Run with the repository's virtualenv::

    .venv/bin/python layout_experiment/run_experiment.py

The script never consults host ``ctypes``. All offsets come from either the
target-aware engine in :mod:`abi_layout` or a dissect.cstruct model pinned to
an explicit pointer type.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from abi_layout import ABI_ILP32, ABI_LP64, layout, render_diagram  # noqa: E402
from byteprobe import build_buffer, check_round_trip, expected_int, leaf_values, parse_block  # noqa: E402
from cstruct_models import POLICIES, build_model, cstruct_field_offsets  # noqa: E402

ABIS = (ABI_ILP32, ABI_LP64)

# Leaf paths that must be readable from both array elements.
WATCHED = (
    "magic",
    "next",
    "__anonymous_0__.type",
    "__anonymous_0__.info.id",
    "__anonymous_0__.info.flags",
    "count",
)


def _engine_offsets(result) -> dict[str, tuple[int, int]]:
    rec = result.node.children[0].children[0]
    out = {}
    for child in rec.children:
        if child.kind == "pad":
            continue
        out[child.name or "__anonymous_0__"] = (child.offset, child.size)
    return out


def report_layout_tables() -> None:
    print("=" * 78)
    print("1) FIELD OFFSET / SIZE TABLES (independent target-ABI engine)")
    print("=" * 78)
    for abi in ABIS:
        for policy in POLICIES:
            result = layout(abi, policy)
            rec = result.node.children[0].children[0]
            print(f"\n[{abi.pointer_size * 8}-bit {abi.name.split()[0]}] policy={policy:<8} "
                  f"sizeof(Rec)={rec.size:<3} sizeof(Block=Rec[2])={result.size}")
            for child in rec.children:
                tag = child.kind if child.kind != "scalar" else child.name
                print(f"    {tag:<18} offset={child.offset:#04x}  size={child.size}  "
                      f"align={child.alignment}")


def report_byte_maps() -> None:
    print("\n" + "=" * 78)
    print("2) HAND-CHECKABLE BYTE DIAGRAM (LP64 natural; CC = padding sentinel)")
    print("=" * 78)
    result = layout(ABI_LP64, "natural")
    print(render_diagram(result))


def cross_check_engine_vs_cstruct() -> None:
    print("\n" + "=" * 78)
    print("3) ENGINE vs dissect.cstruct OFFSET CROSS-CHECK (both execution paths)")
    print("=" * 78)
    failures = 0
    for abi in ABIS:
        for policy in POLICIES:
            engine = layout(abi, policy)
            expected = _engine_offsets(engine)
            for compiled in (False, True):
                model = build_model(abi, policy, compiled=compiled)
                actual = {k: v for k, v in cstruct_field_offsets(model).items()
                          if not k.startswith("pad")}
                ok = actual == expected and len(model.block) == engine.size
                failures += not ok
                path = "compiled" if compiled else "interpreted"
                print(f"  {abi.pointer_size * 8}-bit {policy:<9} {path:<12} "
                      f"Rec={len(model.rec):<3} Block={len(model.block):<3} "
                      f"match={ok}")
                assert ok, (abi, policy, compiled, actual, expected)
    print(f"  -> {failures} mismatches; definition drives both paths identically.")


def verify_nested_array_second_element() -> None:
    print("\n" + "=" * 78)
    print("4) NESTED ARRAY: offset of item[1] and its fields")
    print("=" * 78)
    for abi in ABIS:
        for policy in POLICIES:
            result = layout(abi, policy)
            model = build_model(abi, policy, compiled=True)
            buf = build_buffer(result)
            check_round_trip(result, model, bytes(buf))
            obj = parse_block(model, bytes(buf))
            rec_node = result.node.children[0].children[0]
            stride = rec_node.size
            base = stride
            print(f"\n  {abi.pointer_size * 8}-bit {policy:<9} sizeof(Rec)={stride} "
                  f"-> item[1] starts at {base:#04x}")
            values = leaf_values(obj, 1)
            offsets = {path: absolute for path, _, absolute in result.walk()}
            for leaf in WATCHED:
                key = f"Block.item[1].{leaf}"
                exp = expected_int(bytes(buf), result, key)
                flat = leaf.replace("__anonymous_0__.", "").replace("info.", "info.")
                got = values[flat]
                assert got == exp, (key, got, exp)
                print(f"      {leaf:<28} abs offset {offsets[key]:#04x}  value={got:#x}")


def isolate_pointer_width_alignment_union() -> None:
    print("\n" + "=" * 78)
    print("5) ISOLATING THE THREE CONTRIBUTING FACTORS")
    print("=" * 78)
    packed32 = layout(ABI_ILP32, "packed")
    packed64 = layout(ABI_LP64, "packed")
    natural64 = layout(ABI_LP64, "natural")
    rows = [
        ("pointer width alone (packed 32->64)", packed32, packed64),
        ("alignment alone (64 packed->natural)", packed64, natural64),
    ]
    for label, before, after in rows:
        b = _engine_offsets(before)
        a = _engine_offsets(after)
        print(f"\n  {label}")
        for name in b:
            move = a[name][0] - b[name][0]
            print(f"    {name:<18} {b[name][0]:#04x} -> {a[name][0]:#04x}  "
                  f"(delta {move:+d}, size {b[name][1]} -> {a[name][1]})")
    # Union size contribution: the union itself is max(member) = 4 in both
    # ABIs and never grows here; its 4-byte alignment decides where 'count' may
    # start once natural padding is enabled.
    def _union_node(result):
        rec = result.node.children[0].children[0]
        return next(child for child in rec.children if child.kind == "union")

    union32 = _union_node(packed32)
    union64 = _union_node(natural64)
    print("\n  union size (max of type/Inner/raw, rounded to union align 4):")
    print(f"    32-bit packed union size={union32.size}  "
          f"64-bit natural union size={union64.size} (unchanged across ABIs)")


def demonstrate_host_default_trap() -> None:
    print("\n" + "=" * 78)
    print("6) THE BUG: inheriting the HOST default pointer width")
    print("=" * 78)
    import sys as _sys

    host_ptr = 8 if _sys.maxsize > 2**32 else 4
    print(f"  analysis host pointer width: {host_ptr * 8}-bit "
          f"(cstruct() default pointer follows sys.maxsize)")
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
    # 32-bit firmware parsed without pinning the pointer type, on a 64-bit host.
    wrong = cstruct()  # no pointer= argument -> host default uint64
    wrong.load(definition, align=True)
    right = cstruct(pointer="uint32")
    right.load(definition, align=True)
    print(f"  wrong model (host default): next@{wrong.Rec.__fields__[1].offset} "
          f"size={len(wrong.Rec)}")
    print(f"  right model (pinned u32) : next@{right.Rec.__fields__[1].offset} "
          f"size={len(right.Rec)}")
    # Feed a genuinely 32-bit natural-layout Rec (16 bytes), padded to the
    # 24 bytes the wrong 64-bit model insists on consuming.
    firmware = bytearray(24)
    firmware[0:4] = (0x11111111).to_bytes(4, "little")    # magic
    firmware[4:8] = (0x22222222).to_bytes(4, "little")    # 32-bit pointer
    firmware[8:12] = (0x33333333).to_bytes(4, "little")   # anonymous union
    firmware[12:14] = (0x4444).to_bytes(2, "little")      # count
    bad = wrong.Rec(bytes(firmware))
    good = right.Rec(bytes(firmware))
    print(f"  parsed 'count': wrong={int(bad.count):#x} (reads union/ptr bytes) "
          f"correct={int(good.count):#x}")
    print(f"  parsed 'next' : wrong={int(bad.next):#x} (8 bytes at off 8) "
          f"correct={int(good.next):#x} (4 bytes at off 4)")
    print(f"  parsed union : wrong.type={int(bad.type):#x} "
          f"correct.type={int(good.type):#x}")


def main() -> int:
    report_layout_tables()
    report_byte_maps()
    cross_check_engine_vs_cstruct()
    verify_nested_array_second_element()
    isolate_pointer_width_alignment_union()
    demonstrate_host_default_trap()
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
