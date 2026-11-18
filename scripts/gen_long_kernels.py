#!/usr/bin/env python3
"""Generate long, compute-heavy Atlas kernels for any tile size.

Each workload is a slice of a transformer layer whose problem size is given in
elements, so one problem compiles to N x N tiles for whatever geometry the
hardware config has (``--hardware DefaultHardwareConfig | Tile16HardwareConfig |
Tile64HardwareConfig``). Delays come from the issue rules (``npu_model.scheduling``)
at that geometry, so every program is dense (each instruction issues at the
earliest legal cycle after the ones before it, in program order) and legal.

Workloads:

* ``attention``: flash-attention prefill for one head, ``--seq`` queries against
  ``--seq`` keys of width ``--head-dim``. K and V^T are DMA'd into VMEM once and
  stay resident; Q tiles are prefetched (double-buffered) one query tile ahead and
  each output tile is DMA'd out. Per (query tile, key tile): QK^T on MXU0, online
  softmax on the VPU (scale, row max, running max, two exps, row sum, rescale),
  P quantized to E4M3 through an MXU0 accumulator readout, PV on MXU1. VPU-bound.
* ``mlp``: Gemma-style GeGLU MLP over ``--tokens`` tokens, ``--hidden`` ->
  ``--ffn`` -> ``--hidden``. X is DMA'd in once; gate/up/down weights for each
  N-column block of the FFN are prefetched (double-buffered) while the previous
  block runs over every token tile. Per (FFN block, token tile): gate and up
  projections (MXU0, MXU1), GELU-tanh on the VPU, quantize, and the down
  projection accumulated into BF16 Y tiles held in VMEM (read-modify-write).
  Y is DMA'd out at the end. More DMA- and VLOAD-heavy than attention.

Loops run on the core (one copy of each loop body in IMEM). The innermost loop
body gets a steady-state schedule: copies are scheduled back to back until the
delays repeat, then checked against the rules at the fixed delays. Loop entries
and exits, and every DMA wait, sit behind a drain (a delay until every unit is
idle), so a body's first iteration and the code after a loop always start from a
state the schedule covers. Tensor data is whatever DRAM holds; ``--run``
preloads random finite FP8 data, and timing does not depend on the values.

    python scripts/gen_long_kernels.py --workload attention --hardware Tile16HardwareConfig --out build/long
    python scripts/gen_long_kernels.py --workload all --hardware all --out build/long --run

Each program brackets the kernel with the energy programs' window markers
(CSR_DBG1 <- start mcycle, then CSR_DBG1 <- window cycles) and ends with
CSR_DBG0 <- 1, ecall.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

import torch  # noqa: E402

import npu_model.configs.hardware as hardware_configs  # noqa: E402
import npu_model.configs.isa_definition  # noqa: E402,F401  (registers the ISA)
from npu_model.geometry import Geometry, validate  # noqa: E402
from npu_model.hardware.config import HardwareConfig  # noqa: E402
from npu_model.scheduling import Scheduler, op_from_instruction  # noqa: E402
from npu_model.scheduling.issue_rules import FOREVER, Profile  # noqa: E402
from npu_model.util.converter import stream_to_instrs  # noqa: E402

HARDWARE = ("DefaultHardwareConfig", "Tile16HardwareConfig", "Tile64HardwareConfig")
WORKLOADS = ("attention", "mlp")
BRANCHES = {"beq", "bne", "blt", "bge", "bltu", "bgeu"}
MAX_DELAY = 2000
"""Largest stall one ``delay`` emits; longer drains use several."""
DRAIN_MARGIN = 2
"""Idle cycles added after the last unit activity at a drain."""
PLACE_LIMIT = 4096
"""Cycles the scheduler searches for an instruction's earliest legal issue."""
STEADY_COPIES = 8
"""Loop-body copies scheduled back to back before falling back to a drained loop."""
VERIFY_COPIES = 4
DMA_BASE = 1
"""dma.config base: DRAM operands address 4 GiB + offset (``Program.dram_base``)."""
SCALE_E = 1
"""e1 <- 127: unit scale for FP8 accumulator readouts."""

# Scalar registers the window markers and halt sequence own.
X_T0, X_T1, X_HALT = 29, 30, 31


def bf16_bits(value: float) -> int:
    return int(torch.tensor(value, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF


# --------------------------------------------------------------------------
# Scheduling helpers
# --------------------------------------------------------------------------

def _forever(p: Profile) -> bool:
    return any(h.last >= FOREVER // 2 for h in p.holds)


def _end(p: Profile) -> int:
    """Issue-relative cycle of the last thing ``p`` does (ignoring an unwaited DMA channel)."""
    ends = [h.last for h in p.holds if h.last < FOREVER // 2]
    ends += [r.at for r in p.rows] + [g.last for g in p.gates] + [c.last for c in p.counted]
    ends += [p.reads_free, p.writes_free]
    return max(ends, default=0)


def _prune(sched: Scheduler, start: int) -> None:
    """Drop instructions that can no longer constrain anything issued at ``start`` or later
    (every rule compares a later instruction's non-negative offsets against these windows)."""
    sched.placed = [(o, p, t) for (o, p, t) in sched.placed if _forever(p) or t + _end(p) >= start]


def _idle_after(sched: Scheduler) -> int:
    """First cycle at which every placed instruction has finished."""
    return max((t + _end(p) for _, p, t in sched.placed), default=0) + 1


def _parse(line: str):
    """One assembly line -> one Instruction (branch labels are resolved later)."""
    text = line.split("#", 1)[0].strip()
    tokens = [t for t in re.split(r"[\s,]+", text) if t]
    if tokens[0] in BRANCHES and not re.fullmatch(r"-?\d+|-?0x[0-9a-fA-F]+", tokens[-1]):
        text = text[: text.rfind(tokens[-1])] + "2"
    insns = stream_to_instrs(StringIO(text))
    assert len(insns) == 1, f"not a single-word instruction: {line!r}"
    return insns[0]


def _exec_scalar(line: str, xrf: list[int]) -> None:
    """Track the scalar registers the address operands depend on."""
    text = line.split("#", 1)[0].strip()
    tokens = [t for t in re.split(r"[\s,()]+", text) if t]
    if not tokens:
        return
    name, ops = tokens[0], tokens[1:]

    def reg(t: str) -> int:
        return int(t[1:])

    def imm(t: str) -> int:
        return int(t, 0)

    value = None
    if name == "addi":
        value = xrf[reg(ops[1])] + imm(ops[2])
    elif name == "add":
        value = xrf[reg(ops[1])] + xrf[reg(ops[2])]
    elif name == "sub":
        value = xrf[reg(ops[1])] - xrf[reg(ops[2])]
    elif name == "lui":
        value = imm(ops[1]) << 12
    if value is not None and reg(ops[0]) != 0:
        xrf[reg(ops[0])] = value & 0xFFFFFFFF


class Emitter:
    """Assembly text plus the issue-rule schedule of the segment being written."""

    def __init__(self, g: Geometry) -> None:
        self.g = g
        self.lines: list[str] = []
        self.xrf = [0] * 32
        self.words = 0
        self._labels = 0
        self.loops: list[dict] = []
        self._fresh()

    # -- segment state ----------------------------------------------------
    def _fresh(self) -> None:
        self.sched = Scheduler(self.g)
        self.last: int | None = None

    def _place(self, sched: Scheduler, insn, xrf: list[int], start: int) -> int:
        _prune(sched, start)
        at, _ = sched.place(op_from_instruction(insn, xrf, geometry=self.g), not_before=start,
                            limit=PLACE_LIMIT)
        return at

    def _delay_lines(self, gap: int) -> list[str]:
        """Lines that make the next instruction issue ``gap`` cycles after the previous one."""
        out, remaining = [], gap - 1
        while remaining >= 1:                 # one `delay n` takes n + 1 cycles
            n = min(MAX_DELAY, remaining - 1)
            out.append(f"    delay {n}")
            remaining -= n + 1
        return out

    def _raw(self, line: str) -> None:
        self.lines.append(line)
        text = line.split("#", 1)[0].strip()
        if text and not text.endswith(":"):
            self.words += 1

    # -- public -------------------------------------------------------------
    def label(self, stem: str) -> str:
        self._labels += 1
        return f"{stem}_{self._labels}"

    def comment(self, text: str) -> None:
        self.lines.append(f"    # {text}")

    def emit(self, line: str) -> None:
        """Append one instruction at its earliest legal cycle in the current segment."""
        insn = _parse(line)
        start = 0 if self.last is None else self.last + 1
        at = self._place(self.sched, insn, self.xrf, start)
        if self.last is not None:
            for d in self._delay_lines(at - self.last):
                self._raw(d)
        self._raw(f"    {line}")
        self.last = at
        _exec_scalar(line, self.xrf)

    def li(self, reg: int, value: int) -> None:
        """``li`` as single words, so the scheduler sees each one."""
        v = value & 0xFFFFFFFF
        s = v - (1 << 32) if v & 0x80000000 else v
        if -2048 <= s <= 2047:
            self.emit(f"addi x{reg}, x0, {s}")
            return
        lo = v & 0xFFF
        lo = lo - 0x1000 if lo & 0x800 else lo
        hi = ((v - lo) >> 12) & 0xFFFFF
        self.emit(f"lui x{reg}, 0x{hi:x}")
        if lo:
            self.emit(f"addi x{reg}, x{reg}, {lo}")

    def drain(self) -> None:
        """Delay until every unit is idle, then start a new segment."""
        if self.last is not None:
            target = max(_idle_after(self.sched) + DRAIN_MARGIN, self.last + 1)
            for d in self._delay_lines(target - self.last):
                self._raw(d)
            self._raw("    addi x0, x0, 0    # drain")
        self._fresh()

    def loop(self, name: str, counter: int, trips: int, body: list[str]) -> None:
        """A counted loop of ``trips`` iterations with a steady-state schedule.

        The loop is entered and left through drains. ``body`` is straight-line
        single-word instructions; the counter decrement and the branch are appended."""
        assert trips >= 1
        self.li(counter, trips)
        self.drain()
        top = self.label(name)
        lines = body + [f"addi x{counter}, x{counter}, -1", f"bne x{counter}, x0, {top}",
                        "addi x0, x0, 0    # delay slot"]
        insns = [_parse(line) for line in lines]
        steady = self._steady(insns, lines)
        mode = "steady"
        if steady is None:
            mode = "drained"
            steady = self._drained(insns, lines)
        entry, gaps, tail = steady
        self.lines.append(f"{top}:")
        if entry > 1:
            for d in self._delay_lines(entry):   # previous iteration's delay slot -> first op
                self._raw(d)
        for i, line in enumerate(lines):
            if i:
                for d in self._delay_lines(gaps[i - 1]):
                    self._raw(d)
            self._raw(f"    {line}")
        cycles = entry - 1 + sum(gaps) + 1     # one iteration, issue to issue
        self.loops.append({"loop": top, "mode": mode, "trips": trips, "body_words": len(lines),
                           "cycles_per_iteration": cycles})
        for _ in range(trips):
            for line in lines:
                _exec_scalar(line, self.xrf)
        # Exit drain: the last iteration's work finishes ``tail`` cycles after its delay slot.
        for d in self._delay_lines(max(tail + DRAIN_MARGIN, 1)):
            self._raw(d)
        self._raw("    addi x0, x0, 0    # drain")
        self._fresh()

    def _steady(self, insns, lines):
        sched, xrf = Scheduler(self.g), list(self.xrf)
        last, history = None, []
        for _ in range(STEADY_COPIES):
            first_last = last
            cycles = []
            for insn, line in zip(insns, lines):
                at = self._place(sched, insn, xrf, 0 if last is None else last + 1)
                cycles.append(at)
                last = at
                _exec_scalar(line, xrf)
            entry = None if first_last is None else cycles[0] - first_last
            gaps = [b - a for a, b in zip(cycles, cycles[1:])]
            history.append((entry, gaps))
            if len(history) >= 3 and history[-1] == history[-2] == history[-3]:
                return self._verify(insns, lines, entry, gaps)
        return None

    def _drained(self, insns, lines):
        """Fallback: every iteration starts from an idle machine."""
        sched, xrf, last = Scheduler(self.g), list(self.xrf), None
        cycles = []
        for insn, line in zip(insns, lines):
            at = self._place(sched, insn, xrf, 0 if last is None else last + 1)
            cycles.append(at)
            last = at
            _exec_scalar(line, xrf)
        gaps = [b - a for a, b in zip(cycles, cycles[1:])]
        entry = _idle_after(sched) + DRAIN_MARGIN - last
        result = self._verify(insns, lines, entry, gaps)
        assert result is not None, "a drained loop body must be legal"
        return result

    def _verify(self, insns, lines, entry, gaps):
        """Check ``VERIFY_COPIES`` iterations at the fixed delays, from an idle machine.
        Returns (entry, gaps, tail) or None if any rule breaks."""
        sched, xrf, at = Scheduler(self.g), list(self.xrf), -entry
        for _ in range(VERIFY_COPIES):
            at += entry
            for i, (insn, line) in enumerate(zip(insns, lines)):
                if i:
                    at += gaps[i - 1]
                _prune(sched, at)
                if sched.check(op_from_instruction(insn, xrf, geometry=self.g), at):
                    return None
                _exec_scalar(line, xrf)
        return entry, gaps, _idle_after(sched) - at

    # -- DMA ------------------------------------------------------------------
    def dma(self, kind: str, vmem_reg: int, dram_reg: int, nbytes: int, channels: list[int],
            size_reg: int, words_reg: int, tmp_v: int, tmp_d: int, max_transfer: int) -> list[int]:
        """``nbytes`` between VMEM (word address in ``vmem_reg``) and DRAM (offset in
        ``dram_reg``), in transfers of at most ``max_transfer`` bytes spread over
        ``channels``; a channel is waited before it is reused. Returns the channels used.
        Callers drain first: a wait stalls issue."""
        chunk = min(max_transfer, nbytes)
        assert nbytes % chunk == 0 and chunk % 4 == 0
        count = nbytes // chunk
        self.li(size_reg, chunk)
        if count > 1:
            self.li(words_reg, chunk // 4)
        self.emit(f"addi x{tmp_v}, x{vmem_reg}, 0")
        self.emit(f"addi x{tmp_d}, x{dram_reg}, 0")
        used: list[int] = []
        for c in range(count):
            ch = channels[c % len(channels)]
            if ch in used:
                self.emit(f"dma.wait ch{ch}")
            else:
                used.append(ch)
            if kind == "load":
                self.emit(f"dma.load x{tmp_v}, x{tmp_d}, x{size_reg}, ch{ch}")
            else:
                self.emit(f"dma.store x{tmp_d}, x{tmp_v}, x{size_reg}, ch{ch}")
            if c + 1 < count:
                self.emit(f"add x{tmp_v}, x{tmp_v}, x{words_reg}")   # VMEM operands count words
                self.emit(f"add x{tmp_d}, x{tmp_d}, x{size_reg}")
        return used

    def wait(self, channels) -> None:
        for ch in channels:
            self.emit(f"dma.wait ch{ch}")


# --------------------------------------------------------------------------
# Layout helpers
# --------------------------------------------------------------------------

class Registers:
    """Sequential tensor-register allocator."""

    def __init__(self, count: int) -> None:
        self.count, self.next = count, 0

    def take(self, n: int = 1) -> list[int]:
        regs = list(range(self.next, self.next + n))
        self.next += n
        if self.next > self.count:
            raise ValueError(f"the kernel needs {self.next} tensor registers; the core has {self.count}")
        return regs

    def one(self) -> int:
        return self.take()[0]

    def pair(self) -> int:
        """A BF16 pair (m, m + 1); the VPU writes pairs only to even registers."""
        self.next += self.next % 2
        return self.take(2)[0]


class Vmem:
    """Regions placed at bank boundaries, so each region's accesses stay in its own banks."""

    def __init__(self, g: Geometry) -> None:
        self.g, self.next, self.regions = g, 0, {}

    def region(self, name: str, nbytes: int) -> int:
        bank = self.g.vmem_bank_bytes
        base = -(-self.next // bank) * bank
        if base + nbytes > self.g.vmem_size:
            raise ValueError(f"VMEM region {name} ({nbytes} B at 0x{base:x}) does not fit in "
                             f"{self.g.vmem_size} B of VMEM; shrink the problem")
        self.regions[name] = (base, nbytes)
        self.next = base + nbytes
        return base


class Dram:
    def __init__(self) -> None:
        self.next, self.regions = 0, {}

    def region(self, name: str, nbytes: int, data: str | None) -> int:
        base = -(-self.next // 4096) * 4096
        self.regions[name] = {"offset": base, "bytes": nbytes, "init": data}
        self.next = base + nbytes
        return base


@dataclass
class Kernel:
    name: str
    hardware: str
    g: Geometry
    params: dict
    text: str
    words: int
    dram: dict
    vmem: dict
    loops: list[dict]
    estimate: int
    notes: list[str] = field(default_factory=list)


def _vls(g: Geometry, base_reg: int, byte_offset: int) -> str:
    assert byte_offset % 128 == 0 and -2048 <= byte_offset // 128 < 2048
    return f"{byte_offset // 128}(x{base_reg})"


def _setup(e: Emitter, cfg: HardwareConfig) -> None:
    e.comment("---- setup ----")
    e.emit(f"seli e{SCALE_E}, 127")
    e.li(1, DMA_BASE)
    for ch in range(cfg.dma_num_channels):
        e.emit(f"dma.config x1, ch{ch}")


def _window_start(e: Emitter) -> None:
    e.drain()
    e.comment("---- measurement window ----")
    e.emit(f"csrrs x{X_T0}, 0xC00, x0")
    e.emit(f"csrrw x0, 0xC11, x{X_T0}         # CSR_DBG1 <- start cycle: window start")


def _window_end(e: Emitter) -> None:
    e.drain()
    e.emit(f"csrrs x{X_T1}, 0xC00, x0")
    e.emit(f"sub x{X_T1}, x{X_T1}, x{X_T0}")
    e.emit(f"csrrw x0, 0xC11, x{X_T1}         # CSR_DBG1 <- window cycles: window end")
    e.comment("---- end ----")
    e.emit(f"addi x{X_HALT}, x0, 1")
    e.emit(f"csrrw x0, 0xC10, x{X_HALT}")
    e.emit("ecall")


# --------------------------------------------------------------------------
# Workloads
# --------------------------------------------------------------------------

def attention(cfg: HardwareConfig, hardware: str, seq: int, head_dim: int) -> Kernel:
    g = validate(cfg)
    n, tb = g.rows, g.tile_bytes
    if seq % (2 * n) or head_dim % n:
        raise ValueError(f"attention needs seq % {2 * n} == 0 and head_dim % {n} == 0 at tile {n}")
    dt, qt, kt = head_dim // n, seq // n, seq // n
    maxt, chans = cfg.dma_max_transfer_bytes, list(range(cfg.dma_num_channels))
    q_chans, o_chans = chans[: len(chans) // 2], chans[len(chans) // 2:]

    # Tensor registers.
    r = Registers(g.num_m_registers)
    q = r.take(dt)
    k = r.take(dt)
    v = r.take(dt)
    o = [r.pair() for _ in range(dt)]
    scale, ma, mb, lsum, s, red, alpha, p = (r.pair() for _ in range(8))
    t = [r.pair(), r.pair()]
    pf = r.one()

    # Memory. Tiles are N x N E4M3 (K, Q, V^T) or BF16 pairs (O: two registers).
    vm, dr = Vmem(g), Dram()
    k_vm = vm.region("K", kt * dt * tb)
    v_vm = vm.region("Vt", kt * dt * tb)
    q_vm = [vm.region("Q0", dt * tb), vm.region("Q1", dt * tb)]
    o_vm = vm.region("O", dt * 2 * tb)
    q_dr = dr.region("Q", (qt + 1) * dt * tb, "fp8")     # +1: the last prefetch reads past the end
    k_dr = dr.region("K", kt * dt * tb, "fp8")
    v_dr = dr.region("Vt", kt * dt * tb, "fp8")
    o_dr = dr.region("O", qt * dt * 2 * tb, None)

    # Scalar registers.
    QCNT, KCNT, XK, XV, XQCUR, XQNEXT, XDQ, XDO, XOBUF = 2, 3, 4, 5, 6, 7, 8, 9, 10
    XKSTEP, XQSTEP, XOSTEP, XTMP = 11, 12, 13, 14
    XSIZE, XSTEP, XTV, XTD = 16, 17, 18, 19

    e = Emitter(g)
    e.lines.append(f"# Flash-attention prefill: seq {seq}, head_dim {head_dim}, {n}x{n} tiles "
                   f"({qt} query tiles x {kt} key tiles, {dt} head-dim tiles).")
    _setup(e, cfg)
    e.emit(f"vli.all m{scale}, 0x{bf16_bits(1.0 / math.sqrt(head_dim)):04x}    # softmax scale 1/sqrt(d)")
    _window_start(e)

    e.comment("K and V^T stay resident in VMEM")
    for vm_base, dr_base in ((k_vm, k_dr), (v_vm, v_dr)):
        e.li(XTMP, vm_base // 4)
        e.li(XDQ, dr_base)
        used = e.dma("load", XTMP, XDQ, kt * dt * tb, chans, XSIZE, XSTEP, XTV, XTD, maxt)
        e.wait(used)
    e.comment("prefetch query tile 0")
    e.li(XQCUR, q_vm[1] // 4)
    e.li(XQNEXT, q_vm[0] // 4)
    e.li(XDQ, q_dr)
    e.li(XDO, o_dr)
    e.li(XOBUF, o_vm // 4)
    e.li(XQSTEP, dt * tb)
    e.li(XOSTEP, dt * 2 * tb)
    e.li(XKSTEP, dt * tb // 4)
    used = e.dma("load", XQNEXT, XDQ, dt * tb, q_chans, XSIZE, XSTEP, XTV, XTD, maxt)
    e.emit(f"add x{XDQ}, x{XDQ}, x{XQSTEP}")
    e.li(QCNT, qt)
    e.drain()

    outer = e.label("query_tile")
    e.lines.append(f"{outer}:")
    e.comment("query tile i: Q[i] is in Qnext; swap buffers, prefetch Q[i+1]")
    e.wait(used)
    e.emit(f"addi x{XTMP}, x{XQCUR}, 0")
    e.emit(f"addi x{XQCUR}, x{XQNEXT}, 0")
    e.emit(f"addi x{XQNEXT}, x{XTMP}, 0")
    used = e.dma("load", XQNEXT, XDQ, dt * tb, q_chans, XSIZE, XSTEP, XTV, XTD, maxt)
    e.emit(f"add x{XDQ}, x{XDQ}, x{XQSTEP}")
    for d in range(dt):
        e.emit(f"vload m{q[d]}, {_vls(g, XQCUR, d * tb)}")
    e.emit(f"vli.all m{ma}, 0xc2c8    # running max <- -100")
    e.emit(f"vli.all m{lsum}, 0x0000  # running sum <- 0")
    for d in range(dt):
        e.emit(f"vli.all m{o[d]}, 0x0000")
    e.li(XK, k_vm // 4)
    e.li(XV, v_vm // 4)

    def key_tile(m_old: int, m_new: int) -> list[str]:
        b = [f"vload m{k[d]}, {_vls(g, XK, d * tb)}" for d in range(dt)]
        b += [f"vload m{v[d]}, {_vls(g, XV, d * tb)}" for d in range(dt)]
        for d in range(dt):   # S = Q K^T over the head dim
            b += [f"vmatpush.weight.mxu0 w{d % 2}, m{k[d]}",
                  f"vmatmul{'.acc' if d else ''}.mxu0 acc0, m{q[d]}, w{d % 2}"]
        b += [f"vmatpop.bf16.acc.mxu0 m{s}, acc0",
              f"vmul.bf16 m{s}, m{s}, m{scale}",
              f"vredmax.row.bf16 m{red}, m{s}",
              f"vmaximum.bf16 m{m_new}, m{m_old}, m{red}",
              f"vsub.bf16 m{alpha}, m{m_old}, m{m_new}",
              f"vsub.bf16 m{p}, m{s}, m{m_new}",
              f"vexp.bf16 m{alpha}, m{alpha}",
              f"vexp.bf16 m{p}, m{p}",
              f"vredsum.row.bf16 m{red}, m{p}",
              f"vmatpush.acc.bf16.mxu0 acc1, m{p}",              # P -> E4M3 through a readout
              f"vmatpop.fp8.acc.mxu0 m{pf}, acc1, e{SCALE_E}"]
        b += [f"vmul.bf16 m{o[d]}, m{o[d]}, m{alpha}" for d in range(dt)]
        b += [f"vmul.bf16 m{lsum}, m{lsum}, m{alpha}",
              f"vadd.bf16 m{lsum}, m{lsum}, m{red}"]
        for d in range(dt):   # O += P V
            b += [f"vmatpush.weight.mxu1 w{d % 2}, m{v[d]}",
                  f"vmatmul.mxu1 acc{d % 2}, m{pf}, w{d % 2}",
                  f"vmatpop.bf16.acc.mxu1 m{t[d % 2]}, acc{d % 2}",
                  f"vadd.bf16 m{o[d]}, m{o[d]}, m{t[d % 2]}"]
        b += [f"add x{XK}, x{XK}, x{XKSTEP}", f"add x{XV}, x{XV}, x{XKSTEP}"]
        return b

    e.loop("key_tiles", KCNT, kt // 2, key_tile(ma, mb) + key_tile(mb, ma))

    e.comment("O[i] /= l; store it")
    e.wait(o_chans)
    e.emit(f"vrecip.bf16 m{red}, m{lsum}")
    for d in range(dt):
        e.emit(f"vmul.bf16 m{o[d]}, m{o[d]}, m{red}")
        e.emit(f"vstore m{o[d]}, {_vls(g, XOBUF, d * 2 * tb)}")
        e.emit(f"vstore m{o[d] + 1}, {_vls(g, XOBUF, d * 2 * tb + tb)}")
    e.drain()
    e.dma("store", XOBUF, XDO, dt * 2 * tb, o_chans, XSIZE, XSTEP, XTV, XTD, maxt)
    e.emit(f"add x{XDO}, x{XDO}, x{XOSTEP}")
    e.emit(f"addi x{QCNT}, x{QCNT}, -1")
    e.emit(f"bne x{QCNT}, x0, {outer}")
    e.emit("addi x0, x0, 0    # delay slot")
    e.drain()
    e.wait(used + o_chans)
    _window_end(e)

    inner = e.loops[-1]["cycles_per_iteration"]
    estimate = qt * (kt // 2) * inner
    return Kernel("attention", hardware, g, {"seq": seq, "head_dim": head_dim}, "\n".join(e.lines) + "\n",
                  e.words, dr.regions, vm.regions, e.loops, estimate)


def mlp(cfg: HardwareConfig, hardware: str, tokens: int, hidden: int, ffn: int) -> Kernel:
    g = validate(cfg)
    n, tb = g.rows, g.tile_bytes
    if tokens % n or hidden % n or ffn % n:
        raise ValueError(f"mlp needs tokens, hidden and ffn to be multiples of {n}")
    tt, ht, ft = tokens // n, hidden // n, ffn // n
    maxt, chans = cfg.dma_max_transfer_bytes, list(range(cfg.dma_num_channels))
    block = 3 * ht * tb          # gate, up and down weight tiles of one FFN block

    r = Registers(g.num_m_registers)
    x = r.take(ht)
    pool = min(ht, 4)
    wg, wu, wd = r.take(pool), r.take(pool), r.take(pool)
    gate, up, c, c1, c2, one, half = (r.pair() for _ in range(7))
    y = [r.pair(), r.pair()]
    hf = r.one()

    vm, dr = Vmem(g), Dram()
    x_vm = vm.region("X", tt * ht * tb)
    y_vm = vm.region("Y", tt * ht * 2 * tb)
    w_vm = [vm.region("W0", block), vm.region("W1", block)]
    x_dr = dr.region("X", tt * ht * tb, "fp8")
    w_dr = dr.region("W", (ft + 1) * block, "fp8")        # per block: Wg | Wu | Wd; +1 for the last prefetch
    y_dr = dr.region("Y", tt * ht * 2 * tb, None)

    FCNT, TCNT, XX, XY, XWCUR, XWNEXT, XDW, XWSTEP, XXSTEP, XYSTEP, XTMP = 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    XSIZE, XSTEP, XTV, XTD = 16, 17, 18, 19

    e = Emitter(g)
    e.lines.append(f"# GeGLU MLP: {tokens} tokens, hidden {hidden}, ffn {ffn}, {n}x{n} tiles "
                   f"({tt} token tiles, {ht} hidden tiles, {ft} FFN blocks).")
    _setup(e, cfg)
    for reg, value in ((c1, 0.044715), (c2, math.sqrt(2.0 / math.pi)), (one, 1.0), (half, 0.5)):
        e.emit(f"vli.all m{reg}, 0x{bf16_bits(value):04x}")
    _window_start(e)

    e.comment("X stays resident in VMEM")
    e.li(XTMP, x_vm // 4)
    e.li(XDW, x_dr)
    e.wait(e.dma("load", XTMP, XDW, tt * ht * tb, chans, XSIZE, XSTEP, XTV, XTD, maxt))
    e.comment("prefetch FFN block 0's weights")
    e.li(XWCUR, w_vm[1] // 4)
    e.li(XWNEXT, w_vm[0] // 4)
    e.li(XDW, w_dr)
    e.li(XWSTEP, block)
    e.li(XXSTEP, ht * tb // 4)
    e.li(XYSTEP, ht * 2 * tb // 4)
    used = e.dma("load", XWNEXT, XDW, block, chans, XSIZE, XSTEP, XTV, XTD, maxt)
    e.emit(f"add x{XDW}, x{XDW}, x{XWSTEP}")
    e.li(FCNT, ft)
    e.drain()

    outer = e.label("ffn_block")
    e.lines.append(f"{outer}:")
    e.comment("FFN block f: its weights are in Wnext; swap buffers, prefetch block f+1")
    e.wait(used)
    e.emit(f"addi x{XTMP}, x{XWCUR}, 0")
    e.emit(f"addi x{XWCUR}, x{XWNEXT}, 0")
    e.emit(f"addi x{XWNEXT}, x{XTMP}, 0")
    used = e.dma("load", XWNEXT, XDW, block, chans, XSIZE, XSTEP, XTV, XTD, maxt)
    e.emit(f"add x{XDW}, x{XDW}, x{XWSTEP}")
    e.li(XX, x_vm // 4)
    e.li(XY, y_vm // 4)

    b = [f"vload m{x[i]}, {_vls(g, XX, i * tb)}" for i in range(ht)]
    for i in range(ht):   # gate = X Wg (MXU0), up = X Wu (MXU1)
        b += [f"vload m{wg[i % pool]}, {_vls(g, XWCUR, i * tb)}",
              f"vmatpush.weight.mxu0 w{i % 2}, m{wg[i % pool]}",
              f"vmatmul{'.acc' if i else ''}.mxu0 acc0, m{x[i]}, w{i % 2}",
              f"vload m{wu[i % pool]}, {_vls(g, XWCUR, (ht + i) * tb)}",
              f"vmatpush.weight.mxu1 w{i % 2}, m{wu[i % pool]}",
              f"vmatmul{'.acc' if i else ''}.mxu1 acc0, m{x[i]}, w{i % 2}"]
    b += [f"vmatpop.bf16.acc.mxu0 m{gate}, acc0",
          f"vmatpop.bf16.acc.mxu1 m{up}, acc0",
          # GELU-tanh(gate) * up
          f"vcube.bf16 m{c}, m{gate}",
          f"vmul.bf16 m{c}, m{c}, m{c1}",
          f"vadd.bf16 m{c}, m{c}, m{gate}",
          f"vmul.bf16 m{c}, m{c}, m{c2}",
          f"vtanh.bf16 m{c}, m{c}",
          f"vadd.bf16 m{c}, m{c}, m{one}",
          f"vmul.bf16 m{c}, m{c}, m{gate}",
          f"vmul.bf16 m{c}, m{c}, m{half}",
          f"vmul.bf16 m{c}, m{c}, m{up}",
          f"vmatpush.acc.bf16.mxu0 acc1, m{c}",
          f"vmatpop.fp8.acc.mxu0 m{hf}, acc1, e{SCALE_E}"]
    for j in range(ht):   # Y[t, j] += h Wd[f, j]
        a = j % 2
        b += [f"vload m{y[a]}, {_vls(g, XY, j * 2 * tb)}",
              f"vload m{y[a] + 1}, {_vls(g, XY, j * 2 * tb + tb)}",
              f"vload m{wd[j % pool]}, {_vls(g, XWCUR, (2 * ht + j) * tb)}",
              f"vmatpush.acc.bf16.mxu1 acc{a}, m{y[a]}",
              f"vmatpush.weight.mxu1 w{a}, m{wd[j % pool]}",
              f"vmatmul.acc.mxu1 acc{a}, m{hf}, w{a}",
              f"vmatpop.bf16.acc.mxu1 m{y[a]}, acc{a}",
              f"vstore m{y[a]}, {_vls(g, XY, j * 2 * tb)}",
              f"vstore m{y[a] + 1}, {_vls(g, XY, j * 2 * tb + tb)}"]
    b += [f"add x{XX}, x{XX}, x{XXSTEP}", f"add x{XY}, x{XY}, x{XYSTEP}"]
    e.loop("token_tiles", TCNT, tt, b)

    e.emit(f"addi x{FCNT}, x{FCNT}, -1")
    e.emit(f"bne x{FCNT}, x0, {outer}")
    e.emit("addi x0, x0, 0    # delay slot")
    e.drain()
    e.wait(used)
    e.comment("store Y")
    e.li(XTMP, y_vm // 4)
    e.li(XDW, y_dr)
    e.wait(e.dma("store", XTMP, XDW, tt * ht * 2 * tb, chans, XSIZE, XSTEP, XTV, XTD, maxt))
    _window_end(e)

    inner = e.loops[-1]["cycles_per_iteration"]
    estimate = ft * tt * inner
    return Kernel("mlp", hardware, g, {"tokens": tokens, "hidden": hidden, "ffn": ffn},
                  "\n".join(e.lines) + "\n", e.words, dr.regions, vm.regions, e.loops, estimate)


# --------------------------------------------------------------------------
# Output and simulation
# --------------------------------------------------------------------------

def header(k: Kernel) -> str:
    timeout = max(1_000_000, 3 * k.estimate)
    lines = [f"# {k.name}_{k.g.rows}.S: generated by npu-model/scripts/gen_long_kernels.py; do not edit.",
             f"# @WORKLOAD {k.name} " + " ".join(f"{a}={b}" for a, b in k.params.items()),
             f"# @HARDWARE {k.hardware}",
             f"# @TILE {k.g.rows} {k.g.row_bytes}",
             f"# @TIMEOUT {timeout}",
             f"# @ESTIMATED_CYCLES {k.estimate}",
             "# @PERF_REPORT",
             "# DRAM (offsets from dma.base = 4 GiB; 'fp8' regions are inputs, preloaded by --run):"]
    lines += [f"#   {name:3s} 0x{r['offset']:08x} {r['bytes']:>9d} B  {r['init'] or 'output'}"
              for name, r in k.dram.items()]
    lines += ["# VMEM:"] + [f"#   {name:3s} 0x{base:06x} {size:>9d} B" for name, (base, size) in k.vmem.items()]
    lines += ["# Loops:"] + [f"#   {l['loop']}: {l['mode']}, {l['trips']} trips x {l['cycles_per_iteration']} cycles"
                             for l in k.loops]
    return "\n".join(lines) + "\n"


def random_fp8(nbytes: int, seed: int) -> torch.Tensor:
    """Finite E4M3 bytes: sign, exponent 4..7 (magnitudes 2^-3..2) and mantissa."""
    gen = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=gen)
    return (raw & 0x9F) | 0x20


def simulate(path: Path, k: Kernel, cfg: HardwareConfig, max_cycles: int | None) -> dict:
    from npu_model.logging import LoggerConfig
    from npu_model.simulation import Simulation
    from npu_model.software.program import InstantiableProgram
    from npu_model.util.converter import load_asm

    program = InstantiableProgram(load_asm(path))
    program.tile = (k.g.rows, k.g.row_bytes)
    program.memory_regions = [(r["offset"], random_fp8(r["bytes"], seed=i + 1))
                              for i, r in enumerate(k.dram.values()) if r["init"]]
    limit = max_cycles or max(1_000_000, 3 * k.estimate)
    sim = Simulation(cfg, LoggerConfig(filename=os.devnull), program, verbose=False)
    t0 = time.time()
    sim.run(max_cycles=limit)
    wall = time.time() - t0
    stats = sim.get_stats()
    state = sim.core.arch_state
    result = {
        "program": str(path), "hardware": k.hardware, "tile": k.g.rows, "cycles": stats.cycles,
        "estimated_cycles": k.estimate, "halt": state.halt_reason, "runtime_errors": stats.runtime_errors,
        "instructions": stats.total_instructions, "wall_seconds": round(wall, 1),
        "utilization": {name: round(s.utilization, 4) for name, s in stats.exu_stats.items()},
    }
    sim.close()
    return result


def build(workload: str, hardware: str, args) -> tuple[Kernel, HardwareConfig]:
    cfg = getattr(hardware_configs, hardware)()
    if workload == "attention":
        return attention(cfg, hardware, args.seq, args.head_dim), cfg
    return mlp(cfg, hardware, args.tokens, args.hidden, args.ffn), cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workload", choices=WORKLOADS + ("all",), default="all")
    parser.add_argument("--hardware", default="all",
                        help=f"a class in npu_model.configs.hardware, or 'all' ({', '.join(HARDWARE)})")
    parser.add_argument("--out", type=Path, default=MODEL / "build/long_kernels")
    parser.add_argument("--seq", type=int, default=2048, help="attention: queries = keys")
    parser.add_argument("--head-dim", type=int, default=64, help="attention: head dimension")
    parser.add_argument("--tokens", type=int, default=1024, help="mlp: tokens")
    parser.add_argument("--hidden", type=int, default=256, help="mlp: model width")
    parser.add_argument("--ffn", type=int, default=1024, help="mlp: FFN width")
    parser.add_argument("--run", action="store_true", help="simulate each program and report cycles")
    parser.add_argument("--max-cycles", type=int, help="simulation limit (default: 3x the estimate)")
    args = parser.parse_args()

    workloads = WORKLOADS if args.workload == "all" else (args.workload,)
    hardware = HARDWARE if args.hardware == "all" else (args.hardware,)
    args.out.mkdir(parents=True, exist_ok=True)
    for hw in hardware:
        for wl in workloads:
            try:
                kernel, cfg = build(wl, hw, args)
            except ValueError as error:
                print(f"{wl} on {hw}: {error}")
                continue
            path = args.out / f"{wl}_tile{kernel.g.rows}.S"
            path.write_text(header(kernel) + kernel.text)
            print(f"{path}: {kernel.words} words, ~{kernel.estimate:,} cycles estimated "
                  f"({', '.join(f'{l['loop']} {l['mode']} {l['cycles_per_iteration']}c x{l['trips']}' for l in kernel.loops)})")
            if args.run:
                result = simulate(path, kernel, cfg, args.max_cycles)
                path.with_suffix(".result.json").write_text(json.dumps(result, indent=2) + "\n")
                busy = ", ".join(f"{u} {f:.0%}" for u, f in result["utilization"].items() if f > 0.005)
                print(f"  ran {result['cycles']:,} cycles in {result['wall_seconds']}s, halt "
                      f"{result['halt']}, {result['runtime_errors']} runtime errors; busy: {busy}",
                      flush=True)


if __name__ == "__main__":
    main()
