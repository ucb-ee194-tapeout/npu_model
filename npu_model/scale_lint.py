"""Opt-in lint: NVFP4 MXU reads of stale or never-loaded scales.

A tensor register's data and its scale sidecar are independent, and both are undefined after reset
(docs/nvfp4-mxu.md in Atlas). `vmatpush.weight.nv` and `vmatmul(.acc).nv` read m[vs1]'s data
together with its scales. The lint warns when such a read can see

- scales that no `vload.scale` loaded since reset, or
- data written after the register's last `vload.scale`: the scales can belong to older data.

It follows every path through branches and jumps and stops after `jalr`, `ecall` and `ebreak`. A
warning names the instruction by its byte address (pc). Nothing runs it by default:

    python -m npu_model.scale_lint PROGRAM.S ...
"""
import argparse
import sys
from pathlib import Path

from .isa import Instruction, SBType, UJType
from .isa_types import EXU, MatrixReg
from .util.converter import load_asm

SCALE_READS = ("vmatpush.weight.nv", "vmatmul.nv", "vmatmul.acc.nv")

# Number of m-registers, from vd up, whose data an instruction writes. The other VPU instructions
# write a BF16 pair; all other instructions write no m-register data.
WRITE_WIDTHS = {
    "vload": 1,
    "vli.col": 1,
    "vli.one": 1,
    "vpack.bf16.fp8": 1,
    "vtrpose.xlu": 1,
    "vmatpop.fp8.acc.mxu1": 1,
    "vmatpop.bf16.acc.mxu1": 2,
    "vmatpop.bf16.scaled.acc.mxu1": 2,
    "vmatpop.bf16.acc.nv": 2,
    "vmatpop.bf16.scaled.acc.nv": 2,
    "vmatpop.fp32.acc.nv": 4,
}

# The facts at an instruction (by its index in the program) are the registers whose scales may be
# unloaded, and for each register whose data may be written after its last vload.scale, the
# indices of those writes.
type Facts = tuple[frozenset[int], dict[int, frozenset[int]]]


def data_writes(insn: Instruction) -> range:
    """The m-registers whose data insn writes."""
    width = WRITE_WIDTHS.get(insn.mnemonic, 2 if insn.exu == EXU.VECTOR else 0)
    return range(insn.vd, insn.vd + width) if width else range(0)


def _signed(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def _successors(program: list[Instruction], i: int) -> list[int]:
    insn = program[i]
    if insn.mnemonic in ("ecall", "ebreak", "jalr"):  # the jalr target is not known
        nexts = []
    elif isinstance(insn, SBType):
        nexts = [i + 1, i + (_signed(insn.imm, 13) >> 2)]
    elif isinstance(insn, UJType):
        nexts = [i + (_signed(insn.imm, 21) >> 2)]
    else:
        nexts = [i + 1]
    return [n for n in nexts if 0 <= n < len(program)]


def _join(a: Facts, b: Facts) -> Facts:
    return a[0] | b[0], {r: a[1].get(r, frozenset()) | b[1].get(r, frozenset()) for r in a[1].keys() | b[1].keys()}


def _step(insn: Instruction, i: int, facts: Facts) -> Facts:
    unloaded, written = facts
    if insn.mnemonic == "vload.scale":
        return unloaded - {insn.vd}, {r: writes for r, writes in written.items() if r != insn.vd}
    return unloaded, written | {r: frozenset({i}) for r in data_writes(insn)}


def scale_warnings(program: list[Instruction]) -> list[tuple[int, str]]:
    """(pc, message) for each NV read of scales that may be stale or never loaded."""
    if not program:
        return []
    facts: dict[int, Facts] = {0: (frozenset(range(MatrixReg.upper_bound)), {})}
    work = [0]
    while work:
        i = work.pop()
        out = _step(program[i], i, facts[i])
        for nxt in _successors(program, i):
            new = _join(facts[nxt], out) if nxt in facts else out
            if new != facts.get(nxt):
                facts[nxt] = new
                work.append(nxt)
    warnings = []
    for i, (unloaded, written) in sorted(facts.items()):
        reg = program[i].vs1 if program[i].mnemonic in SCALE_READS else None
        if reg in unloaded:
            warnings.append((4 * i, f"m{reg}'s scales may not be loaded since reset"))
        elif reg in written:
            pcs = ", ".join(str(4 * w) for w in sorted(written[reg]))
            warnings.append((4 * i, f"m{reg}'s data may be written after its last vload.scale (pc {pcs})"))
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Warn on NVFP4 MXU reads of stale or never-loaded scales.")
    parser.add_argument("programs", nargs="+", type=Path, help="assembly (.S) files")
    found = 0
    for path in parser.parse_args().programs:
        program = load_asm(path)
        for pc, message in scale_warnings(program):
            print(f"{path}: pc {pc}: {program[pc // 4].mnemonic}: {message}")
            found += 1
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
