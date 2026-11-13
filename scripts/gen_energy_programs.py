#!/usr/bin/env python3
"""Generate energy-characterization programs: one per ISA instruction.

Each program repeats a single instruction ``@ENERGY_REPS`` times on random
operands inside a measurement window, so RTL power simulation (Cadence Joules)
of that window, together with the EnergyOps the model attributes to the same
window, gives energy per EnergyOp. Rerunning the programs on modified RTL gives
the points for fitting each op's analytical model to uarch parameters.

Every size comes from a hardware config (``--hardware``, ``--param``): tile
bytes, VMEM layout, how long each unit holds a command and the setup drain all
follow ``npu_model/geometry.py``, the rules the timed model runs. The checked-in
set (``npu_model/configs/programs/energy/asm``) is for ``DefaultHardwareConfig``;
other sets go to ``--out`` with a ``manifest.json``.

Program layout (``energy_<mnemonic>.S``):

* Setup. An on-chip xorshift32 loop fills the VMEM regions the instruction
  reads, so a program is self-contained and needs no DRAM preload from the RTL
  harness. The loop masks each word to keep values finite and representative:
  BF16 values are in +-[2^-7, 2) (exponent 120..127), E4M3 values in
  +-[2^-3, 2) (never NaN), and scalar memory gets uniform 32-bit words. The
  setup then loads matrix registers, accumulators, weight buffers, scale and
  scalar registers as the instruction needs, and drains every unit.
* Window. ``CSR_DBG1 <- mcycle`` (first DBG1 write) starts it. The instruction
  runs ``@ENERGY_REPS`` times. Operands rotate through distinct random
  registers, tiles and immediates between repetitions. The core has no
  interlocks, so tensor commands are spaced by software (``--spacing``):
  ``isolated`` (default) issues each after the previous one's last write plus
  ``SLACK``, so every repetition runs alone; ``dense`` issues each at the
  earliest cycle the issue rules (``npu_model.scheduling``) allow, so they
  overlap as in a steady-state kernel. DMA transfers are each waited.
  ``CSR_DBG1 <- window cycles`` (second DBG1 write) ends the window once
  everything has drained.
* End. ``CSR_DBG0 <- 1``, ``ecall``, as the DMA calibration programs do.

ecall and ebreak halt the core, so they cannot repeat inside a window; every
program already ends with one. Programs for delay (and the idle cycles of every
window) give the background power to subtract.

    python scripts/gen_energy_programs.py           # rewrites the checked-in programs
    python scripts/gen_energy_programs.py --check   # exit 1 if they are stale
    python scripts/gen_energy_programs.py --hardware Tile64HardwareConfig --out build/tile64 --run
    python scripts/gen_energy_programs.py --param vmem_size=0x200000 --spacing dense --out build/dense

``--run`` simulates every written program on its config and prints the window
length and the EnergyOps each repetition records.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

import npu_model.configs.hardware as hardware_configs  # noqa: E402
import npu_model.configs.isa_definition  # noqa: E402,F401  (registers the ISA)
from npu_model.geometry import Geometry, validate  # noqa: E402
from npu_model.hardware.config import HardwareConfig, apply_overrides  # noqa: E402
from npu_model.isa import IsaSpec  # noqa: E402
from npu_model.scheduling import Scheduler, op_from_instruction  # noqa: E402
from npu_model.util.converter import stream_to_instrs  # noqa: E402

OUT = MODEL / "npu_model/configs/programs/energy/asm"

SCALAR_REPS = 64
"""Repetitions of a single-cycle scalar, branch, CSR, LSU-scalar or DMA-control instruction."""
TENSOR_REPS = 16
"""Repetitions of a multi-cycle VPU, XLU, MXU, VLOAD/VSTORE or DMA-transfer instruction."""
SLACK = 4
"""Idle cycles added to a unit's occupancy between isolated commands: margin for
RTL whose units hold a command a little longer than the model's rules."""
BASE_TIMEOUT = 200_000
"""Cycle limit at 32 x 32; the fill loops, which dominate, scale with the tile."""
WORD_BYTES = 1024
"""Uniform words for scalar loads and seld, and the scalar store window."""
SPACINGS = ("isolated", "dense")
DEFAULT_HARDWARE = "DefaultHardwareConfig"

EXCLUDED = {
    "ecall": "halts the core, so it executes once per program (every program ends with it)",
    "ebreak": "halts the core, so it executes once per program",
}

# VMEM layout, in tensor registers (tiles) of the config's geometry:
#   BF16 pool   tiles 0..7    -> m0..m7, four BF16 pairs
#   FP8 pool    tiles 8..11   -> m8..m11, four E4M3 tiles
#   word pool   tile 12       WORD_BYTES of uniform words for scalar loads and seld
#   scratch     tiles 32..39  destination of stores, vstore and dma.load

BF16_PAIRS = (0, 2, 4, 6)      # random BF16 pairs (m, m+1)
FP8_TILES = (8, 9, 10, 11)     # random E4M3 tiles
DST = (32, 34, 36, 38)         # destinations; pairs or singles
SCALE = 1                      # e1 = 127, unit scale for vpack, vunpack and FP8 pops
SRC_X = tuple(range(10, 26))   # x10..x25 hold random words
DST_X = (26, 27)

# xorshift32 output mask and fill bits per data kind.
FILL = {
    "bf16": (0x83FF83FF, 0x3C003C00),   # sign, exponent 120..127 and mantissa
    "bf16+": (0x03FF03FF, 0x3C003C00),  # the same magnitudes, positive (vsqrt, vlog2)
    "fp8": (0x9F9F9F9F, 0x20202020),    # E4M3 sign, exponent 4..7 and mantissa
    "word": (0xFFFFFFFF, 0x00000000),   # uniform words
}

HEADER = """# energy_{stem}.S: energy characterization, {reps} x {mnemonic} on random operands.
# Window: the first CSR_DBG1 write (DBG1 <- start mcycle) to the second (DBG1 <- window cycles).
# Generated by npu-model/scripts/gen_energy_programs.py; do not edit.
# @ENERGY_OP {mnemonic}
# @ENERGY_REPS {reps}
# @TIMEOUT {timeout}
# @HARDWARE {hardware}
# @TILE {rows} {row_bytes}
# @SPACING {spacing}
# @PERF_REPORT
"""


@dataclass(frozen=True)
class Layout:
    """Everything size-dependent about a program set, from one hardware config."""

    hardware: str
    g: Geometry
    dma_channels: int
    dma_max_transfer: int
    spacing: str = "isolated"

    @classmethod
    def of(cls, config: HardwareConfig, spacing: str = "isolated") -> Layout:
        assert spacing in SPACINGS, spacing
        layout = cls(type(config).__name__, validate(config), config.dma_num_channels,
                     config.dma_max_transfer_bytes, spacing)
        if layout.scratch + 8 * layout.tile > layout.g.vmem_size:
            raise ValueError(f"the program layout needs {40 * layout.tile} bytes of VMEM; "
                             f"{layout.hardware} has {layout.g.vmem_size}")
        return layout

    @property
    def tile(self) -> int:
        return self.g.tile_bytes

    @property
    def bf16_pool(self) -> int:
        return 0

    @property
    def fp8_pool(self) -> int:
        return 8 * self.tile

    @property
    def word_pool(self) -> int:
        return 12 * self.tile

    @property
    def scratch(self) -> int:
        return 32 * self.tile

    @property
    def dma_bytes(self) -> int:
        """Bytes per DMA transfer under test: one register, or the largest transfer."""
        return min(self.tile, self.dma_max_transfer)

    @property
    def timeout(self) -> int:
        return BASE_TIMEOUT * max(1, self.tile // 1024)

    @property
    def setup_drain(self) -> int:
        """Delay ending the setup, longer than any unit's command, so the window starts idle."""
        g = self.g
        return max(max(g.vpu_latencies().values()), max(g.mxu_latencies().values()),
                   g.transpose_cycles, g.vls_latency) + 10

    @property
    def vls_delay(self) -> str:
        """After a VLOAD/VSTORE: its path is free again (VMEM T+1..T+R, MREG through T+R+2)."""
        return f"    delay {self.g.vls_latency - 1}"

    @property
    def scratch_base(self) -> str:
        """x0 while VSTORE immediates (128-byte units, 12-bit signed) reach the scratch
        tiles; past that (from 128 x 128 tiles) x9 holds the scratch word address."""
        return "x0" if (self.scratch + 8 * self.tile) // 128 < 2048 else "x9"

    @property
    def scratch_setup(self) -> tuple[str, ...]:
        return () if self.scratch_base == "x0" else (li("x9", self.scratch // 4),)

    def vls(self, vmem: int) -> str:
        """``imm(base)`` operand of a VLOAD/VSTORE of the register at VMEM byte ``vmem``."""
        assert vmem % self.tile == 0, f"VMEM 0x{vmem:x} is not register aligned"
        base, start = ("x9", self.scratch) if vmem >= self.scratch and self.scratch_base == "x9" else ("x0", 0)
        imm = (vmem - start) // 128
        assert imm < 2048, f"VMEM 0x{vmem:x} is out of VLS immediate range"
        return f"{imm}({base})"

    def vload(self, mreg: int, vmem: int) -> list[str]:
        return [f"    vload m{mreg}, {self.vls(vmem)}", self.vls_delay]

    def tensor_delay(self, latency: int) -> str:
        return f"    delay {latency + SLACK}"


def stem(mnemonic: str) -> str:
    return mnemonic.replace(".", "_")


def li(reg: str, value: int) -> str:
    return f"    li    {reg}, 0x{value & 0xFFFFFFFF:08x}"


def nop(comment: str = "") -> str:
    return "    addi  x0, x0, 0" + (f"    # {comment}" if comment else "")


def fill(label: str, vmem: int, nbytes: int, kind: str, seed: int) -> list[str]:
    """xorshift32 words, masked to ``kind``, into VMEM[vmem, vmem + nbytes)."""
    mask, bits = FILL[kind]
    return [
        f"    # VMEM[0x{vmem:04x}, 0x{vmem + nbytes:04x}): {kind} data, xorshift32 seed 0x{seed:08x}",
        li("x2", vmem), li("x3", vmem + nbytes), li("x4", seed), li("x5", mask), li("x6", bits),
        f"{label}:",
        "    slli  x7, x4, 13", "    xor   x4, x4, x7",
        "    srli  x7, x4, 17", "    xor   x4, x4, x7",
        "    slli  x7, x4, 5", "    xor   x4, x4, x7",
        "    and   x7, x4, x5", "    or    x7, x7, x6",
        "    sw    x7, 0(x2)",
        "    addi  x2, x2, 4",
        f"    blt   x2, x3, {label}",
        nop("branch delay slot"),
    ]


@dataclass
class Setup:
    """What a program loads before its window."""
    pools: tuple[str, ...] = ()        # "bf16" | "bf16+" | "fp8" | "word"
    mregs: bool = True                 # vload the BF16/FP8 pools into m0..m11
    scalars: bool = False              # x10..x25 <- random words
    scale: bool = False                # e1 <- 127
    weights: str | None = None         # "mxu0" | "mxu1": w0, w1 <- m8, m9
    accumulators: str | None = None    # "mxu0" | "mxu1": acc0, acc1 <- m0, m2
    lines: tuple[str, ...] = ()        # extra setup, last


@dataclass
class Spec:
    mnemonic: str
    reps: int
    setup: Setup
    body: Callable[[int, random.Random, dict[int, int]], list[str]]
    """Lines of repetition ``i``, given the random x10..x25 values."""
    spaced: bool = False
    """The body is one tensor command and the delay after it; ``--spacing`` sets the delay."""


def tensor_spec(mnemonic: str, setup: Setup, body: Callable) -> Spec:
    return Spec(mnemonic, TENSOR_REPS, setup, body, spaced=True)


def rx(rng: random.Random) -> int:
    return rng.choice(SRC_X)


def bf16_bits(rng: random.Random) -> int:
    mask, bits = FILL["bf16"]
    return (rng.getrandbits(16) & mask & 0xFFFF) | (bits & 0xFFFF)


# --------------------------------------------------------------------------
# Instruction families
# --------------------------------------------------------------------------

R_TYPE = ("add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and")
I_TYPE = ("addi", "slti", "sltiu", "xori", "ori", "andi")
SHIFT_IMM = ("slli", "srli", "srai")
BRANCHES = {
    "beq": lambda a, b: a == b, "bne": lambda a, b: a != b,
    "blt": lambda a, b: _signed(a) < _signed(b), "bge": lambda a, b: _signed(a) >= _signed(b),
    "bltu": lambda a, b: a < b, "bgeu": lambda a, b: a >= b,
}
CSR_REG = ("csrrw", "csrrs", "csrrc")
CSR_IMM = ("csrrwi", "csrrsi", "csrrci")
CSR_TARGET = 0xC01  # instret: writable, and neither the window clock (0xC00) nor DBG0/DBG1
SCALAR_LOADS = {"lb": 1, "lbu": 1, "lh": 2, "lhu": 2, "lw": 4}
SCALAR_STORES = {"sb": 1, "sh": 2, "sw": 4}
VPU_BINARY = ("vadd.bf16", "vsub.bf16", "vmul.bf16", "vminimum.bf16", "vmaximum.bf16")
VPU_UNARY = ("vredsum.bf16", "vredmin.bf16", "vredmax.bf16", "vredsum.row.bf16",
             "vredmin.row.bf16", "vredmax.row.bf16", "vmov", "vrecip.bf16", "vexp.bf16",
             "vexp2.bf16", "vrelu.bf16", "vsin.bf16", "vcos.bf16", "vtanh.bf16",
             "vlog2.bf16", "vsqrt.bf16", "vsquare.bf16", "vcube.bf16")
POSITIVE_DOMAIN = {"vlog2.bf16", "vsqrt.bf16"}
VLI = ("vli.all", "vli.row", "vli.col", "vli.one")


def _signed(v: int) -> int:
    return v - (1 << 32) if v & 0x80000000 else v


def _dst(i: int) -> str:
    return f"x{DST_X[i % len(DST_X)]}"


def specs(L: Layout) -> list[Spec]:
    out: list[Spec] = []
    scalar = Setup(mregs=False, scalars=True)

    for m in R_TYPE:
        out.append(Spec(m, SCALAR_REPS, scalar,
                        lambda i, rng, x, m=m: [f"    {m:5s} {_dst(i)}, x{rx(rng)}, x{rx(rng)}"]))
    for m in I_TYPE:
        out.append(Spec(m, SCALAR_REPS, scalar,
                        lambda i, rng, x, m=m: [f"    {m:5s} {_dst(i)}, x{rx(rng)}, {rng.randint(-2048, 2047)}"]))
    for m in SHIFT_IMM:
        out.append(Spec(m, SCALAR_REPS, scalar,
                        lambda i, rng, x, m=m: [f"    {m:5s} {_dst(i)}, x{rx(rng)}, {rng.randint(0, 31)}"]))
    for m in ("lui", "auipc"):
        out.append(Spec(m, SCALAR_REPS, scalar,
                        lambda i, rng, x, m=m: [f"    {m:5s} {_dst(i)}, 0x{rng.getrandbits(20):05x}"]))
    out.append(Spec("seli", SCALAR_REPS, scalar,
                    lambda i, rng, x: [f"    seli  e{2 + i % 8}, {rng.randint(-2048, 2047)}"]))
    out.append(Spec("fence", SCALAR_REPS, scalar, lambda i, rng, x: ["    fence"]))
    out.append(Spec("delay", SCALAR_REPS, scalar, lambda i, rng, x: [f"    delay {rng.randint(0, 63)}"]))

    # Branches: each repetition is taken or not with probability 1/2, and its
    # target is the instruction after its delay slot, so the path is the same.
    for m, cond in BRANCHES.items():
        def branch(i, rng, x, m=m, cond=cond):
            taken = rng.random() < 0.5
            while True:
                a, b = rx(rng), rx(rng)
                if m in ("beq", "bne") and taken == (m == "beq"):
                    b = a          # equal operands: the same register
                elif a == b:
                    continue
                if cond(x[a], x[b]) != taken:
                    a, b = b, a
                if cond(x[a], x[b]) == taken:
                    break
            return [f"    {m:5s} x{a}, x{b}, br_{i}", nop("delay slot"), f"br_{i}:"]
        out.append(Spec(m, SCALAR_REPS, scalar, branch))
    out.append(Spec("jal", SCALAR_REPS, scalar, lambda i, rng, x: [
        f"    jal   x{(1, 26, 27)[i % 3]}, jt_{i}", nop("delay slot"), f"jt_{i}:"]))
    # jalr targets are absolute: x28 holds the PC of the auipc ending the setup.
    out.append(Spec("jalr", SCALAR_REPS, Setup(mregs=False, scalars=True, lines=("    auipc x28, 0",)),
                    lambda i, rng, x: [f"    jalr  x{(1, 26, 27)[i % 3]}, x28, @jt_{i}",
                                       nop("delay slot"), f"jt_{i}:"]))

    for m in CSR_REG:
        out.append(Spec(m, SCALAR_REPS, scalar,
                        lambda i, rng, x, m=m: [f"    {m:5s} {_dst(i)}, 0x{CSR_TARGET:03x}, x{rx(rng)}"]))
    for m in CSR_IMM:
        out.append(Spec(m, SCALAR_REPS, scalar,
                        lambda i, rng, x, m=m: [f"    {m:6s} {_dst(i)}, 0x{CSR_TARGET:03x}, {rng.randint(0, 31)}"]))

    # Scalar LSU. A load's response must not meet another load or a scalar
    # writeback, so loads are spaced by a short delay; stores issue back to back.
    loads = Setup(pools=("word",), mregs=False, lines=(li("x8", L.word_pool),))
    for m, size in SCALAR_LOADS.items():
        out.append(Spec(m, SCALAR_REPS, loads, lambda i, rng, x, m=m, size=size: [
            f"    {m:5s} {_dst(i)}, {rng.randrange(0, WORD_BYTES, size)}(x8)", "    delay 2"]))
    out.append(Spec("seld", SCALAR_REPS, loads, lambda i, rng, x: [
        f"    seld  e{2 + i % 8}, {rng.randrange(0, WORD_BYTES, 4)}(x8)", "    delay 2"]))
    stores = Setup(mregs=False, scalars=True, lines=(li("x8", L.scratch),))
    for m, size in SCALAR_STORES.items():
        out.append(Spec(m, SCALAR_REPS, stores, lambda i, rng, x, m=m, size=size: [
            f"    {m:5s} x{rx(rng)}, {rng.randrange(0, WORD_BYTES, size)}(x8)"]))

    # Vector LSU: eight random source tiles each way.
    out.append(tensor_spec("vload", Setup(pools=("bf16",), mregs=False), lambda i, rng, x: [
        f"    vload m{32 + i % 8}, {L.vls(L.bf16_pool + L.tile * (i % 8))}", L.vls_delay]))
    out.append(tensor_spec("vstore", Setup(pools=("bf16",), lines=L.scratch_setup), lambda i, rng, x: [
        f"    vstore m{i % 8}, {L.vls(L.scratch + L.tile * (i % 8))}", L.vls_delay]))

    # VPU. Sources rotate through four random pairs; binary ops pair each with the next.
    for m in VPU_BINARY:
        out.append(tensor_spec(m, Setup(pools=("bf16",)), lambda i, rng, x, m=m: [
            f"    {m} m{DST[i % 4]}, m{BF16_PAIRS[i % 4]}, m{BF16_PAIRS[(i + 1) % 4]}",
            L.tensor_delay(L.g.vpu_latency(m))]))
    for m in VPU_UNARY:
        kind = "bf16+" if m in POSITIVE_DOMAIN else "bf16"
        out.append(tensor_spec(m, Setup(pools=(kind,)), lambda i, rng, x, m=m: [
            f"    {m} m{DST[i % 4]}, m{BF16_PAIRS[i % 4]}", L.tensor_delay(L.g.vpu_latency(m))]))
    out.append(tensor_spec("vpack.bf16.fp8", Setup(pools=("bf16",), scale=True), lambda i, rng, x: [
        f"    vpack.bf16.fp8 m{DST[i % 4]}, m{BF16_PAIRS[i % 4]}, e{SCALE}",
        L.tensor_delay(L.g.vpu_latency("vpack.bf16.fp8"))]))
    out.append(tensor_spec("vunpack.fp8.bf16", Setup(pools=("fp8",), scale=True), lambda i, rng, x: [
        f"    vunpack.fp8.bf16 m{DST[i % 4]}, m{FP8_TILES[i % 4]}, e{SCALE}",
        L.tensor_delay(L.g.vpu_latency("vunpack.fp8.bf16"))]))
    for m in VLI:
        out.append(tensor_spec(m, Setup(mregs=False), lambda i, rng, x, m=m: [
            f"    {m} m{DST[i % 4]}, 0x{bf16_bits(rng):04x}", L.tensor_delay(L.g.vpu_latency(m))]))

    out.append(tensor_spec("vtrpose.xlu", Setup(pools=("bf16",)), lambda i, rng, x: [
        f"    vtrpose.xlu m{32 + i % 4}, m{i % 8}", L.tensor_delay(L.g.transpose_cycles)]))

    # MXUs. Two weight slots and two accumulators hold random data; inputs
    # rotate through four random E4M3 tiles.
    for unit in ("mxu0", "mxu1"):
        def mxu(op, body, setup, unit=unit):
            m = f"{op}.{unit}"
            out.append(tensor_spec(m, setup, lambda i, rng, x, m=m: [
                f"    {m} {body(i)}", L.tensor_delay(L.g.mxu_latency(m))]))
        mxu("vmatpush.weight", lambda i: f"w{i % 2}, m{FP8_TILES[i % 4]}", Setup(pools=("fp8",)))
        mxu("vmatpush.acc.fp8", lambda i: f"acc{i % 2}, m{FP8_TILES[i % 4]}", Setup(pools=("fp8",)))
        mxu("vmatpush.acc.bf16", lambda i: f"acc{i % 2}, m{BF16_PAIRS[i % 4]}", Setup(pools=("bf16",)))
        mxu("vmatpop.fp8.acc", lambda i: f"m{DST[i % 4]}, acc{i % 2}, e{SCALE}",
            Setup(pools=("bf16",), scale=True, accumulators=unit))
        mxu("vmatpop.bf16.acc", lambda i: f"m{DST[i % 4]}, acc{i % 2}",
            Setup(pools=("bf16",), accumulators=unit))
        mxu("vmatmul", lambda i: f"acc{i % 2}, m{FP8_TILES[i % 4]}, w{(i // 2) % 2}",
            Setup(pools=("fp8",), weights=unit))
        mxu("vmatmul.acc", lambda i: f"acc{i % 2}, m{FP8_TILES[i % 4]}, w{(i // 2) % 2}",
            Setup(pools=("bf16", "fp8"), weights=unit, accumulators=unit))

    # DMA. Transfers move one register (``Layout.dma_bytes``) between four random VMEM tiles and four DRAM
    # lines (dma.base = 1, so DRAM offsets are from 4 GiB), each waited.
    regs = [li("x31", 1), "    dma.config x31, ch0", li("x18", L.dma_bytes)]
    for k in range(4):
        regs += [li(f"x{10 + k}", (L.bf16_pool + L.tile * k) // 4),   # VMEM source words
                 li(f"x{14 + k}", 0x1000 * k),                        # DRAM offsets
                 li(f"x{22 + k}", (L.scratch + L.tile * k) // 4)]     # VMEM destination words
    seed_dram = [line for k in range(4) for line in
                 (f"    dma.store x{14 + k}, x{10 + k}, x18, ch0", "    dma.wait ch0")]
    out.append(Spec("dma.store", TENSOR_REPS, Setup(pools=("bf16",), mregs=False, lines=tuple(regs)),
                    lambda i, rng, x: [f"    dma.store x{14 + i % 4}, x{10 + i % 4}, x18, ch{i % 4}",
                                       f"    dma.wait ch{i % 4}"]))
    out.append(Spec("dma.load", TENSOR_REPS,
                    Setup(pools=("bf16",), mregs=False, lines=tuple(regs + seed_dram)),
                    lambda i, rng, x: [f"    dma.load x{22 + i % 4}, x{14 + i % 4}, x18, ch{i % 4}",
                                       f"    dma.wait ch{i % 4}"]))
    # No transfer follows, so a random dma.base is harmless.
    out.append(Spec("dma.config", SCALAR_REPS, scalar,
                    lambda i, rng, x: [f"    dma.config x{rx(rng)}, ch{rng.randrange(L.dma_channels)}"]))
    out.append(Spec("dma.wait", SCALAR_REPS, scalar,
                    lambda i, rng, x: [f"    dma.wait ch{rng.randrange(L.dma_channels)}"]))
    return out


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def setup_lines(setup: Setup, seed: int, rng: random.Random, L: Layout) -> tuple[list[str], dict[int, int]]:
    base = {"bf16": L.bf16_pool, "bf16+": L.bf16_pool, "fp8": L.fp8_pool, "word": L.word_pool}
    size = {"bf16": 8 * L.tile, "bf16+": 8 * L.tile, "fp8": 4 * L.tile, "word": WORD_BYTES}
    lines: list[str] = []
    for n, kind in enumerate(setup.pools):
        lines += fill(f"fill_{n}", base[kind], size[kind], kind, (seed + n * 0x9E3779B9) & 0xFFFFFFFF or 1)
    if setup.mregs:
        for kind in setup.pools:
            if kind in ("bf16", "bf16+"):
                lines += [line for k in range(8) for line in L.vload(k, L.bf16_pool + L.tile * k)]
            elif kind == "fp8":
                lines += [line for k, m in enumerate(FP8_TILES) for line in L.vload(m, L.fp8_pool + L.tile * k)]
    xs: dict[int, int] = {}
    if setup.scalars:
        for r in SRC_X:
            xs[r] = rng.getrandbits(32)
            lines.append(li(f"x{r}", xs[r]))
    if setup.scale:
        lines.append(f"    seli  e{SCALE}, 127")
    if setup.weights:
        m = f"vmatpush.weight.{setup.weights}"
        for w, src in enumerate(FP8_TILES[:2]):
            lines += [f"    {m} w{w}, m{src}", L.tensor_delay(L.g.mxu_latency(m))]
    if setup.accumulators:
        m = f"vmatpush.acc.bf16.{setup.accumulators}"
        for a, src in enumerate(BF16_PAIRS[:2]):
            lines += [f"    {m} acc{a}, m{src}", L.tensor_delay(L.g.mxu_latency(m))]
    lines += list(setup.lines)
    return lines, xs


def _words_before(lines: list[str]) -> dict[str, int]:
    """Instruction-word index of each label (all window lines are one word)."""
    index, labels = 0, {}
    for line in lines:
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        if text.endswith(":"):
            labels[text[:-1]] = index
        else:
            assert not text.startswith("li "), "window lines must be single words"
            index += 1
    return labels


def dense(reps: list[list[str]], L: Layout, xrf: list[int]) -> list[list[str]]:
    """``reps`` (each a tensor command and its delay) respaced so every command issues
    at the earliest cycle the issue rules allow after the ones before it. The last
    keeps its isolated delay, so the window still ends with every unit idle."""
    scheduler = Scheduler(L.g)
    cycles = [scheduler.place(op_from_instruction(stream_to_instrs(StringIO(rep[0]))[0], xrf,
                                                  geometry=L.g))[0] for rep in reps]
    out = []
    for i, rep in enumerate(reps):
        gap = cycles[i + 1] - cycles[i] if i + 1 < len(reps) else None
        delay = [rep[1]] if gap is None else [f"    delay {gap - 2}"] if gap > 1 else []
        out.append([rep[0]] + delay)
    return out


def program(spec: Spec, L: Layout) -> str:
    seed = zlib.crc32(spec.mnemonic.encode()) or 1
    rng = random.Random(seed)
    setup, xs = setup_lines(spec.setup, seed, rng, L)
    window = ["    # ---- measurement window ----",
              "    csrrs x29, 0xC00, x0",
              "    csrrw x0, 0xC11, x29         # CSR_DBG1 <- start cycle: window start"]
    reps = [spec.body(i, rng, xs) for i in range(spec.reps)]
    if spec.spaced and L.spacing == "dense":
        xrf = [0] * 32
        if L.scratch_base == "x9":
            xrf[9] = L.scratch // 4
        reps = dense(reps, L, xrf)
    for rep in reps:
        window += rep
    window += ["    csrrs x30, 0xC00, x0",
               "    sub   x30, x30, x29",
               "    csrrw x0, 0xC11, x30         # CSR_DBG1 <- window cycles: window end",
               "    # ---- end ----"]
    if any("@" in line for line in window):
        # jalr immediates are word offsets from the auipc that ends the setup.
        labels = _words_before(["    auipc x28, 0"] + window)
        window = [re.sub(r"@(\w+)", lambda match: str(labels[match[1]]), line) for line in window]
    tail = ["    addi  x31, x0, 1", "    csrrw x0, 0xC10, x31", "    ecall", ""]
    drain = [f"    delay {L.setup_drain}                    # setup drains before the window"]
    if setup and setup[-1].strip().startswith("auipc"):
        setup, auipc = setup[:-1], setup[-1:]
        drain += auipc
    head = HEADER.format(stem=stem(spec.mnemonic), mnemonic=spec.mnemonic, reps=spec.reps, timeout=L.timeout,
                         hardware=L.hardware, rows=L.g.rows, row_bytes=L.g.row_bytes, spacing=L.spacing)
    return head + "\n".join(["    # ---- setup ----"] + setup + drain + window + tail)


def generate(config: HardwareConfig | None = None, spacing: str = "isolated") -> dict[str, str]:
    """File name -> text of every program for ``config`` (default: ``DefaultHardwareConfig``)."""
    layout = Layout.of(config or hardware_configs.DefaultHardwareConfig(), spacing)
    all_specs = specs(layout)
    names = [s.mnemonic for s in all_specs]
    assert len(names) == len(set(names)), "duplicate program"
    missing = set(IsaSpec.operations) - set(names) - set(EXCLUDED)
    assert not missing, f"no energy program for {sorted(missing)}"
    return {f"energy_{stem(s.mnemonic)}.S": program(s, layout) for s in all_specs}


def parse_value(text: str) -> object:
    for convert in (lambda v: int(v, 0), float):
        try:
            return convert(text)
        except ValueError:
            pass
    return text


def build_config(hardware: str, params: dict[str, object]) -> HardwareConfig:
    """``hardware`` (a class in ``npu_model.configs.hardware``) with ``params`` applied."""
    try:
        config = getattr(hardware_configs, hardware)()
    except AttributeError:
        sys.exit(f"unknown --hardware {hardware}; choose from {', '.join(hardware_configs.__all__)}")
    # Copy the shared containers before overriding anything on this instance.
    for name in ("execution_units", "dma_memory_params", "energy_op_models"):
        setattr(config, name, dict(getattr(config, name)))
    return apply_overrides(config, params)


def report(directory: Path, config: HardwareConfig) -> None:
    """Simulate each program in ``directory`` on ``config``; print its window and per-repetition ops."""
    from npu_model.configs.programs.energy import load_programs
    from npu_model.energy import measure_window

    for mnemonic, cls in sorted(load_programs(directory).items()):
        window = measure_window(cls(), config, cls.kernel_max_cycles)
        if window.halt_reason != "ecall" or window.idle != [True, True]:
            print(f"{mnemonic:24s} FAILED: halt {window.halt_reason}, idle markers {window.idle}")
            continue
        reps = [uop for uop in window.uops if uop.insn.mnemonic == mnemonic]
        ops = Counter(op.kind for uop in reps for op in uop.energy_ops if op.kind != "instruction_fetch")
        per_rep = ", ".join(f"{kind} {count / len(reps):g}" for kind, count in sorted(ops.items()))
        print(f"{mnemonic:24s} {len(reps):3d} reps, {len(window.uops):4d} window instructions: {per_rep}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hardware", default=DEFAULT_HARDWARE,
                        help="hardware config class in npu_model.configs.hardware")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                        help="override a HardwareConfig or ArchStateConfig field (repeatable)")
    parser.add_argument("--spacing", choices=SPACINGS, default="isolated")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true", help="exit 1 if --out differs from the generator")
    parser.add_argument("--run", action="store_true", help="simulate the programs and print their EnergyOps")
    args = parser.parse_args()
    params = dict((key, parse_value(value)) for key, _, value in (p.partition("=") for p in args.param))
    config = build_config(args.hardware, params)
    custom = args.hardware != DEFAULT_HARDWARE or params or args.spacing != "isolated"
    if custom and args.out.resolve() == OUT.resolve():
        sys.exit("the checked-in set is DefaultHardwareConfig with isolated spacing; write others with --out")
    try:
        progs = generate(config, args.spacing)
    except ValueError as error:
        sys.exit(str(error))
    if args.check:
        on_disk = {p.name: p.read_text() for p in args.out.glob("energy_*.S")}
        stale = sorted(set(progs) ^ set(on_disk) | {n for n in progs if on_disk.get(n) != progs[n]})
        if stale:
            sys.exit(f"stale energy programs (rerun scripts/gen_energy_programs.py): {', '.join(stale)}")
        print(f"{len(progs)} energy programs are up to date")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("energy_*.S"):
        if old.name not in progs:
            old.unlink()
    for name, text in progs.items():
        (args.out / name).write_text(text)
    if custom:
        g = Geometry.from_config(config)
        manifest = {
            "hardware": args.hardware,
            "params": {key: value for key, value in params.items()},
            "tile": [g.rows, g.row_bytes],
            "spacing": args.spacing,
            "programs": {name: {key: value.strip() for key, _, value in
                                (line[3:].partition(" ") for line in text.splitlines() if line.startswith("# @"))
                                if key in ("ENERGY_OP", "ENERGY_REPS", "TIMEOUT")}
                         for name, text in sorted(progs.items())},
        }
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(progs)} programs in {args.out}")
    if args.run:
        report(args.out, config)


if __name__ == "__main__":
    main()
