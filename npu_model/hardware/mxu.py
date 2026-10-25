"""Port sequencers and row pipelines corresponding to the two RTL MXUs.

The timing follows the default 32x32 geometry and two-stage IPT. Values come
from Instruction.exec; this unit decides only when rows are read and written.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .exu import ExecutionUnit, StagedExecution
from ..logging.logger import Logger
from .arch_state import ArchState
from ..software.instruction import Uop
from ..isa import EXU
from .config import HardwareConfig

PORT_CYCLES = 32
"""Cycles a command holds each sequencer port it takes, one row per cycle."""
PORT_CHOICES = {
    "compute": ((0,),),
    "push_bf16": ((0, 1),),
    "weight": ((1,), (0,)),
    "push_fp8": ((1,), (0,)),
    "pop_fp8": ((2,),),
    "pop_bf16": ((2, 3),),
}
"""Sequencer ports each command kind can take, in order of preference: read
ports 0 and 1, write ports 2 and 3."""
_PORTS_BUSY = {
    "compute": "compute issued while read port or in-flight tracker is busy",
    "push_bf16": "BF16 push issued while read ports are busy",
    "weight": "push issued while read ports are busy",
    "push_fp8": "push issued while read ports are busy",
    "pop_fp8": "pop issued while write ports are busy",
    "pop_bf16": "pop issued while write ports are busy",
}
_ROWS = 32
_GRID = torch.arange(_ROWS)


@dataclass(frozen=True)
class MatrixTiming:
    """Source, accumulator and resource ages after command acceptance."""

    accumulator_write_first_age: int
    accumulator_write_last_age: int
    weight_slot_reuse_age: int
    inflight_depth: int
    source_read_last_age: int = 31
    mreg_release_request_age: int = 32
    weight_read_first_age: int = 1
    buffer_write_first_age: int = 1  # Push and pop row writes.
    buffer_write_last_age: int = 32
    push_path_release_age: int = 32
    accumulator_push_reader_gap: int = 2

    @property
    def accumulator_row0_usable_age(self) -> int:
        """First edge a same-accumulator consumer can pass the row-0 gate."""
        return self.accumulator_write_first_age + 1


MXU0_TIMING = MatrixTiming(accumulator_write_first_age=63, accumulator_write_last_age=94, weight_slot_reuse_age=63,
                           inflight_depth=3)
MXU1_TIMING = MatrixTiming(accumulator_write_first_age=3, accumulator_write_last_age=34, weight_slot_reuse_age=32,
                           inflight_depth=2)
MXU_OP_LATENCIES = {
    f"{op}.mxu{unit}": (timing.accumulator_write_last_age if op.startswith("vmatmul")
                        else timing.buffer_write_last_age) + 1
    for unit, timing in enumerate((MXU0_TIMING, MXU1_TIMING))
    for op in ("vmatpush.weight", "vmatpush.acc.fp8", "vmatpush.acc.bf16",
               "vmatmul.acc", "vmatmul", "vmatpop.fp8.acc", "vmatpop.bf16.acc")
}


@dataclass
class _Operation:
    uop: Uop
    issued: int
    kind: str
    ports: tuple[int, ...]
    mreg: int
    acc: int
    weight: int
    owner: str
    staged: StagedExecution
    mreg_released: bool = False

    @property
    def banks(self) -> tuple[int, ...]:
        return (self.mreg, self.mreg + 1) if self.kind in {"push_bf16", "pop_bf16"} else (self.mreg,)


class _MatrixExecutionUnit(ExecutionUnit):
    mxu: str
    exu_type: EXU
    timing: MatrixTiming
    compute_first: int
    inflight_depth: int

    def __init__(self, name: str, logger: Logger, arch_state: ArchState,
                 lane_id: int = 0, config: HardwareConfig | None = None) -> None:
        super().__init__(name, logger, arch_state, lane_id, config)
        self.reset()

    def can_handle(self, uop: Uop) -> bool:
        return uop.insn.exu == self.exu_type

    def reset(self) -> None:
        self.cycle = 0
        self._ops: list[_Operation] = []
        self._pending_completions: list[Uop] = []
        self._complete_count = 0
        self._total_instructions = 0
        self._busy_cycles = 0
        self._last_compute: dict[int, int] = {}

    @property
    def in_flight(self) -> list[Uop]:
        return [op.uop for op in self._ops]

    def abort(self) -> None:
        for op in self._ops:
            self.arch_state.conflict_checker.release_mreg(op.owner)
        self._ops = []
        self._pending_completions = []

    def _execution_latency(self, uop: Uop) -> int:
        return MXU_OP_LATENCIES[uop.insn.mnemonic]

    @staticmethod
    def _kind(mnemonic: str) -> str:
        if mnemonic.startswith("vmatmul"):
            return "compute"
        if mnemonic.startswith("vmatpush.weight"):
            return "weight"
        if mnemonic.startswith("vmatpush"):
            return "push_bf16" if "bf16" in mnemonic else "push_fp8"
        return "pop_bf16" if "bf16" in mnemonic else "pop_fp8"

    @staticmethod
    def _push_path(kind: str) -> str | None:
        """Weight and accumulator pushes each write through one output mux,
        which writes a single push per cycle and drops the others' rows."""
        return "weight" if kind == "weight" else "accumulator" if kind.startswith("push") else None

    def _accept(self, uop: Uop) -> None:
        assert uop.insn.exu == self.exu_type, "Instruction sent to wrong MXU"
        insn, kind = uop.insn, self._kind(uop.insn.mnemonic)
        pop = kind.startswith("pop")
        acc = int(insn.vs2 if pop else insn.vd) & 1
        weight = int(insn.vs2 if kind == "compute" else insn.vd) & 1
        mreg = int(insn.vd if pop else insn.vs1)
        active = [op for op in self._ops if self.cycle - op.issued < PORT_CYCLES]
        compute = [op for op in self._ops if op.kind == "compute"]
        free = lambda port: not any(port in op.ports for op in active)
        row0 = self.timing.accumulator_row0_usable_age
        if kind != "weight" and any(op.acc == acc and self.cycle - op.issued < row0 for op in compute):
            raise RuntimeError(f"{self.mxu}: command issued before previous compute row-0 writeback")
        # Take the first choice whose ports are all free.
        ports = None
        for choice in PORT_CHOICES[kind]:
            if all(free(port) for port in choice):
                ports = choice
                break
        if ports is None or (kind == "compute" and len(compute) >= self.timing.inflight_depth):
            raise RuntimeError(f"{self.mxu}: {_PORTS_BUSY[kind]}")
        if self.mxu == "mxu0" and kind == "weight":
            reuse = self.timing.weight_slot_reuse_age
            if weight in self._last_compute and self.cycle - self._last_compute[weight] < reuse:
                raise RuntimeError("mxu0: weight push issued before previous matmul on same slot drained")
        if self.mxu == "mxu1":
            weight_pushes = [op for op in self._ops if op.kind == "weight" and op.weight == weight
                             and self.cycle - op.issued < self.timing.push_path_release_age]
            acc_pushes = [op for op in self._ops if op.kind.startswith("push") and op.acc == acc
                          and self.cycle - op.issued <= self.timing.buffer_write_last_age]
            if kind == "weight" and (weight_pushes or any(
                    op.kind == "compute" and op.weight == weight
                    and self.cycle - op.issued < self.timing.weight_slot_reuse_age for op in self._ops)):
                raise RuntimeError("mxu1: weight slot busy")
            if kind == "compute" and (weight_pushes or acc_pushes):
                raise RuntimeError("mxu1: compute reads buffer with an active push")
            if kind == "push_fp8" and acc_pushes:
                raise RuntimeError("mxu1: accumulation push target busy")
        path = self._push_path(kind)
        if path and any(self._push_path(op.kind) == path
                        and self.cycle - op.issued < self.timing.push_path_release_age for op in self._ops):
            raise RuntimeError(f"{self.mxu}: {path} push issued while another {path} push is still writing")
        # A reader one cycle behind a push would read each row just before it lands.
        if (pop or (kind == "compute" and ".acc." in insn.mnemonic)) and any(
                self._push_path(op.kind) == "accumulator" and op.acc == acc
                and self.cycle - op.issued < self.timing.accumulator_push_reader_gap
                for op in self._ops):
            raise RuntimeError(f"{self.mxu}: accumulator read issued the cycle after a push to it")
        owner = f"{self.name}:{uop.id}"
        banks = frozenset({mreg, mreg + 1}) if kind in {"push_bf16", "pop_bf16"} else frozenset({mreg})
        self.arch_state.conflict_checker.reserve_mreg(owner, reads=frozenset() if pop else banks,
                                                     writes=banks if pop else frozenset())
        mregs = [("mrf", bank) for bank in sorted(banks)]
        accumulator, weights = ("acc", self.mxu, acc), ("wb", self.mxu, weight)
        if pop:
            reads, writes = [accumulator], mregs
        elif kind == "weight":
            reads, writes = mregs, [weights]
        elif kind == "compute":
            reads = mregs + [weights] + ([accumulator] if ".acc." in insn.mnemonic else [])
            writes = [accumulator]
        else:
            reads, writes = mregs, [accumulator]
        staged = StagedExecution(uop, self.arch_state, reads, writes, self.mxu)
        self._ops.append(_Operation(uop, self.cycle, kind, ports, mreg, acc, weight, owner, staged))
        if kind == "compute":
            self._last_compute[weight] = self.cycle
        uop.execute_delay = self._execution_latency(uop)
        self._total_instructions += 1
        self.logger.log_stage_start(uop.id, "E", lane=self.lane_id, cycle=self.cycle)

    def _sample(self, op: _Operation, row: int) -> None:
        state = self.arch_state
        accumulator = ("acc", self.mxu, op.acc)
        if accumulator in op.staged.operands:
            op.staged.sample(accumulator, row, state.acc[self.mxu][op.acc][row])
        if op.kind.startswith("pop"):
            return
        for bank in op.banks:
            state.conflict_checker.access_mreg(self.cycle, bank, row, False, op.owner)
            op.staged.sample(("mrf", bank), row, state.read_mrf_u8(bank)[row])

    def _weight_uses(self, age: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(output column, inner) weights read at this age, and which are row 0's."""
        if self.mxu == "mxu1":
            # Each IPT row reads the whole tile, row r at age r + 1.
            first = self.timing.weight_read_first_age
            used = torch.full((_ROWS, _ROWS), first <= age < first + _ROWS)
            return used, used & (age == self.timing.weight_read_first_age)
        # SA PE(i, j) evaluates row r at T+1+r+i+j, reading the weight
        # present at that PE's use cycle.
        row = age - self.timing.weight_read_first_age - _GRID[:, None] - _GRID[None, :]
        return (row >= 0) & (row < _ROWS), row == 0

    def _sample_weights(self, op: _Operation, age: int) -> None:
        """Every use of a weight follows row 0's, so a change seen there is
        exact to re-run; any later change would give rows different weights,
        which no legal push schedule produces."""
        location = ("wb", self.mxu, op.weight)
        live = self.arch_state.read_wb_u8(self.mxu, op.weight)
        used, first = self._weight_uses(age)
        changed = used & (live != op.staged.operands[location])
        if (changed & ~first).any():
            raise RuntimeError(f"{self.mxu}: weights changed while a matmul was using them")
        if changed.any():
            op.staged.sample(location, changed, live[changed])

    def _pop_row(self, op: _Operation, row: int) -> None:
        for bank in op.banks:
            self.arch_state.conflict_checker.access_mreg(self.cycle, bank, row, True, op.owner)
        for bank in op.banks:
            self.arch_state.read_mrf_u8(bank)[row] = op.staged.result(("mrf", bank))[row]

    def tick(self, uop: Uop | None) -> None:
        self.cycle += 1
        self.arch_state.conflict_checker.begin_cycle(self.cycle)
        self.flush_completions()
        self._complete_count = 0
        if uop is not None:
            self._accept(uop)
        if self._ops:
            self._busy_cycles += 1
        # Read all SRAM operands before applying this edge's writes. Each
        # accumulator has one read port, shared by pops and accumulate-mode
        # computes; an overwrite compute never reads it.
        readers: dict[int, str] = {}
        for op in self._ops:
            age = self.cycle - op.issued
            if 0 <= age <= self.timing.source_read_last_age:
                if ("acc", self.mxu, op.acc) in op.staged.operands:
                    if op.acc in readers:
                        raise RuntimeError(f"{self.mxu}: compute and pop read the same accumulator")
                    readers[op.acc] = op.owner
                self._sample(op, age)
        for op in self._ops:
            if op.kind == "compute":
                self._sample_weights(op, self.cycle - op.issued)
        # A shared sequencer output mux prefers ReadP1 over ReadP0 for pushes.
        weight_push = None
        acc_push = None
        compute_writes: list[tuple[_Operation, int]] = []
        for op in self._ops:
            age = self.cycle - op.issued
            if (op.kind == "compute"
                    and self.timing.accumulator_write_first_age <= age <= self.timing.accumulator_write_last_age):
                compute_writes.append((op, age - self.timing.accumulator_write_first_age))
            in_buffer_write = self.timing.buffer_write_first_age <= age <= self.timing.buffer_write_last_age
            if in_buffer_write:
                if op.kind == "weight" and (weight_push is None or op.ports[0] > weight_push.ports[0]):
                    weight_push = op
                elif op.kind.startswith("push") and (acc_push is None or op.ports[0] > acc_push.ports[0]):
                    acc_push = op
            if op.kind.startswith("pop") and in_buffer_write:
                self._pop_row(op, age - self.timing.buffer_write_first_age)
        if acc_push is not None and any(op.acc == acc_push.acc for op, _ in compute_writes):
            raise RuntimeError(f"{self.mxu}: compute and push write the same accumulator")
        for op, row in compute_writes:
            self.arch_state.acc[self.mxu][op.acc][row] = op.staged.result(("acc", self.mxu, op.acc))[row]
        if weight_push is not None:
            row = self.cycle - weight_push.issued - self.timing.buffer_write_first_age
            location = ("wb", self.mxu, weight_push.weight)
            self.arch_state.read_wb_u8(self.mxu, weight_push.weight)[row] = weight_push.staged.result(location)[row]
        if acc_push is not None:
            row = self.cycle - acc_push.issued - self.timing.buffer_write_first_age
            location = ("acc", self.mxu, acc_push.acc)
            self.arch_state.acc[self.mxu][acc_push.acc][row] = acc_push.staged.result(location)[row]
        remaining = []
        for op in self._ops:
            age = self.cycle - op.issued
            if age == self.timing.mreg_release_request_age and not op.mreg_released:
                self.arch_state.conflict_checker.release_mreg(op.owner)
                op.mreg_released = True
            last = self.timing.accumulator_write_last_age if op.kind == "compute" else self.timing.buffer_write_last_age
            if age == last:
                self._complete_count += 1
                self._pending_completions.append(op.uop)
            else:
                remaining.append(op)
        self._ops = remaining

    def flush_completions(self) -> None:
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id, cycle=self.cycle)
            self.logger.log_retire(uop.id)
        self._pending_completions = []

    def is_busy(self) -> bool:
        return self.has_in_flight

    @property
    def has_in_flight(self) -> bool:
        return bool(self._ops)

    @property
    def complete_count(self) -> int:
        return self._complete_count

    @property
    def total_instructions(self) -> int:
        return self._total_instructions

    @property
    def busy_cycles(self) -> int:
        return self._busy_cycles


class MatrixExecutionUnitSystolic(_MatrixExecutionUnit):
    mxu = "mxu0"
    exu_type = EXU.MATRIX_SYSTOLIC
    timing = MXU0_TIMING
    compute_first = timing.accumulator_write_first_age
    inflight_depth = timing.inflight_depth


class MatrixExecutionUnitInner(_MatrixExecutionUnit):
    mxu = "mxu1"
    exu_type = EXU.MATRIX_INNER
    timing = MXU1_TIMING
    compute_first = timing.accumulator_write_first_age
    inflight_depth = timing.inflight_depth
