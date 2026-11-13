"""Compile the C reference and cross-check it against the Python engine.

The C program is built four ways (ILP32/LP64 x packed/natural) using only the
native compiler; ILP32 is simulated with a 4-byte integer pointer so no 32-bit
sysroot or multilib is required. This is an independent ABI check - it relies
on the real compiler's layout rules, never on host Python ``ctypes``.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from abi_layout import ABI_ILP32, ABI_LP64, layout  # noqa: E402

SOURCE = HERE / "reference_layout.c"

VARIANTS = (
    ("ILP32", "packed", ("-DSIMULATE_ILP32", "-DPACKED"), ABI_ILP32, "packed"),
    ("ILP32", "natural", ("-DSIMULATE_ILP32",), ABI_ILP32, "natural"),
    ("LP64", "packed", ("-DPACKED",), ABI_LP64, "packed"),
    ("LP64", "natural", (), ABI_LP64, "natural"),
)

FIELD_MAP = {
    "magic": "magic",
    "next": "next",
    "u": "__anonymous_0__",
    "count": "count",
}


def parse_c_output(text: str) -> dict[str, int]:
    values = {}
    for line in text.splitlines():
        match = re.match(r"sizeof\(struct Rec\) = (\d+)", line)
        if match:
            values["rec_size"] = int(match.group(1))
        match = re.match(r"sizeof\(struct Block=Rec\[2\]\) = (\d+)", line)
        if match:
            values["block_size"] = int(match.group(1))
        match = re.match(r"offsetof (magic|next|u|count)\s*=\s*(\d+)", line)
        if match:
            values[FIELD_MAP[match.group(1)]] = int(match.group(2))
        match = re.match(r"offsetof Block\.item\[1\]\s*=\s*(\d+)", line)
        if match:
            values["item1"] = int(match.group(1))
    return values


def engine_values(abi, policy) -> dict[str, int]:
    result = layout(abi, policy)
    rec = result.node.children[0].children[0]
    offsets = {
        (child.name or "__anonymous_0__"): child.offset
        for child in rec.children
        if child.kind != "pad"
    }
    return {
        "rec_size": rec.size,
        "block_size": result.size,
        "item1": rec.size,
        **offsets,
    }


def main() -> int:
    cc = os.environ.get("CC", "cc")
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for label, packing, flags, abi, policy in VARIANTS:
            binary = Path(tmp) / f"ref_{label}_{packing}"
            compiled = subprocess.run(
                [cc, "-Wall", "-Werror", "-std=c11", *flags, str(SOURCE), "-o", str(binary)],
                capture_output=True, text=True,
            )
            if compiled.returncode != 0:
                print(f"[SKIP] {label} {packing}: compiler unavailable\n{compiled.stderr}")
                continue
            run = subprocess.run([str(binary)], capture_output=True, text=True)
            c_values = parse_c_output(run.stdout)
            py_values = engine_values(abi, policy)
            ok = c_values == py_values
            failures += not ok
            print(f"[{'PASS' if ok else 'FAIL'}] C {label:<5} {packing:<7} vs Python engine")
            if not ok:
                print("  C     :", c_values)
                print("  Python:", py_values)
    print("\nAll C reference variants agree with the Python engine." if not failures
          else f"\n{failures} variant(s) diverged.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
