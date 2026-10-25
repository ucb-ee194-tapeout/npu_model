"""Opt-in lint: NVFP4 MXU reads of stale or never-loaded scales.

A tensor register's data and its scale sidecar are independent, and both are undefined after reset
(docs/nvfp4-mxu.md in Atlas). `vmatpush.weight.nv` and `vmatmul(.acc).nv` read m[vs1]'s data
together with its scales. The lint warns when such a read can see

- scales that no `vload.scale` loaded since reset, or
- data written after the register's last `vload.scale`: the scales can belong to older data.

It follows every path through branches (a taken branch or jump runs its delay slot first) and
stops after `jalr`, `ecall` and `ebreak`. Nothing runs it by default:

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
    "vmatpop.bf16.acc.nv": 2,
    "vmatpop.fp32.acc.nv": 4,
}

# A program point is (pc, landing): landing is where a taken branch or jump goes after the delay
# slot at pc, or None. The facts there are the registers whose scales may be unloaded, and for
# each register whose data may be written after its last vload.scale, the PCs of those writes.
type Node = tuple[int, int | None]
type Facts = tuple[frozenset[int], dict[int, frozenset[int]]]


def data_writes(insn: Instruction) -> range:
    """The m-registers whose data insn writes."""
    width = WRITE_WIDTHS.get(insn.mnemonic, 2 if insn.exu == EXU.VECTOR else 0)
    return range(insn.vd, insn.vd + width) if width else range(0)


def _signed(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def _successors(program: list[Instruction], node: Node) -> list[Node]:
    pc, landing = node
    insn = program[pc]
    if insn.mnemonic in ("ecall", "ebreak"):
        nexts = []
    elif landing is not None:
        nexts = [(landing, None)]
    elif isinstance(insn, SBType):
        nexts = [(pc + 1, None), (pc + 1, pc + (_signed(insn.imm, 13) >> 1))]
    elif isinstance(insn, UJType):
        nexts = [(pc + 1, pc + (_signed(insn.imm, 21) >> 1))]
    elif insn.mnemonic == "jalr":
        nexts = [(pc + 1, len(program))]  # the target is not known
    else:
        nexts = [(pc + 1, None)]
    return [n for n in nexts if 0 <= n[0] < len(program)]


def _join(a: Facts, b: Facts) -> Facts:
    return a[0] | b[0], {r: a[1].get(r, frozenset()) | b[1].get(r, frozenset()) for r in a[1].keys() | b[1].keys()}


def _step(insn: Instruction, pc: int, facts: Facts) -> Facts:
    unloaded, written = facts
    if insn.mnemonic == "vload.scale":
        return unloaded - {insn.vd}, {r: pcs for r, pcs in written.items() if r != insn.vd}
    return unloaded, written | {r: frozenset({pc}) for r in data_writes(insn)}


def scale_warnings(program: list[Instruction]) -> list[tuple[int, str]]:
    """(pc, message) for each NV read of scales that may be stale or never loaded."""
    if not program:
        return []
    facts: dict[Node, Facts] = {(0, None): (frozenset(range(MatrixReg.upper_bound)), {})}
    work = [(0, None)]
    while work:
        node = work.pop()
        out = _step(program[node[0]], node[0], facts[node])
        for nxt in _successors(program, node):
            new = _join(facts[nxt], out) if nxt in facts else out
            if new != facts.get(nxt):
                facts[nxt] = new
                work.append(nxt)
    at_pc: dict[int, Facts] = {}
    for (pc, _), f in facts.items():
        at_pc[pc] = _join(at_pc[pc], f) if pc in at_pc else f
    warnings = []
    for pc, (unloaded, written) in sorted(at_pc.items()):
        reg = program[pc].vs1 if program[pc].mnemonic in SCALE_READS else None
        if reg in unloaded:
            warnings.append((pc, f"m{reg}'s scales may not be loaded since reset"))
        elif reg in written:
            pcs = ", ".join(map(str, sorted(written[reg])))
            warnings.append((pc, f"m{reg}'s data may be written after its last vload.scale (pc {pcs})"))
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Warn on NVFP4 MXU reads of stale or never-loaded scales.")
    parser.add_argument("programs", nargs="+", type=Path, help="assembly (.S) files")
    found = 0
    for path in parser.parse_args().programs:
        program = load_asm(path)
        for pc, message in scale_warnings(program):
            print(f"{path}: pc {pc}: {program[pc].mnemonic}: {message}")
            found += 1
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
