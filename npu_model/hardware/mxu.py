"""Port sequencer and row pipeline of the RTL's inner-product-tree MXU (mxu1).

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

MXU_OP_LATENCIES = {
    f"{op}.mxu1": 35 if op.startswith("vmatmul") else 65 if "fp32" in op else 33
    for op in ("vmatpush.weight", "vmatpush.acc.fp8", "vmatpush.acc.bf16", "vmatpush.acc.fp32",
               "vmatmul.acc", "vmatmul", "vmatpop.fp8.acc", "vmatpop.bf16.acc", "vmatpop.bf16.scaled.acc",
               "vmatpop.fp32.acc")
}


def _banks(kind: str, mreg: int, step: int) -> tuple[int, ...]:
    """The m-registers that step `step` of a push or pop reads or writes. An FP32 transfer moves
    columns 0-15 through m[r], m[r+1] in steps 0-31, then columns 16-31 through m[r+2], m[r+3]
    in steps 32-63."""
    if kind.endswith("fp32"):
        return (mreg + 2 * (step // 32), mreg + 2 * (step // 32) + 1)
    return (mreg, mreg + 1) if kind in {"push_bf16", "pop_bf16"} else (mreg,)


@dataclass
class _Operation:
    uop: Uop
    issued: int
    kind: str
    steps: int
    ports: tuple[int, ...]
    mreg: int
    acc: int
    weight: int
    owner: str
    staged: StagedExecution
    released: int = 0  # m-register reservations released: one per 32 steps

    def columns(self, step: int) -> slice:
        """The accumulator columns that step `step` moves."""
        return slice(16 * (step // 32), 16 * (step // 32) + 16) if self.kind.endswith("fp32") else slice(None)


class _MatrixExecutionUnit(ExecutionUnit):
    mxu: str
    exu_type: EXU
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

    @property
    def in_flight(self) -> list[Uop]:
        return [op.uop for op in self._ops]

    def abort(self) -> None:
        for op in self._ops:
            for phase in range(op.released, op.steps // 32):
                self.arch_state.conflict_checker.release_mreg(f"{op.owner}/{phase}")
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
            return f"push_{mnemonic.split('.')[2]}"
        return f"pop_{mnemonic.split('.')[1]}"

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
        steps = 64 if kind.endswith("fp32") else 32
        active = [op for op in self._ops if self.cycle - op.issued < op.steps]
        compute = [op for op in self._ops if op.kind == "compute"]
        free = lambda port: not any(port in op.ports for op in active)
        if kind != "weight" and any(op.acc == acc and self.cycle - op.issued <= self.compute_first for op in compute):
            raise RuntimeError(f"{self.mxu}: command issued before previous compute row-0 writeback")
        if kind == "compute":
            if not free(0) or len(compute) >= self.inflight_depth:
                raise RuntimeError(f"{self.mxu}: compute issued while read port or in-flight tracker is busy")
            ports = (0,)
        elif kind in ("push_bf16", "push_fp32"):
            if not (free(0) and free(1)):
                raise RuntimeError(f"{self.mxu}: BF16 or FP32 push issued while read ports are busy")
            ports = (0, 1)
        elif pop:
            ports = (2,) if kind == "pop_fp8" else (2, 3)
            if not all(free(port) for port in ports):
                raise RuntimeError(f"{self.mxu}: pop issued while write ports are busy")
        else:
            if not (free(0) or free(1)):
                raise RuntimeError(f"{self.mxu}: push issued while read ports are busy")
            ports = (1,) if free(1) else (0,)
        weight_pushes = [op for op in active if op.kind == "weight" and op.weight == weight]
        acc_pushes = [op for op in self._ops if op.kind.startswith("push") and op.acc == acc
                      and self.cycle - op.issued <= op.steps]
        if kind == "weight" and (weight_pushes or any(op.kind == "compute" and op.weight == weight for op in active)):
            raise RuntimeError("mxu1: weight slot busy")
        if kind == "compute" and (weight_pushes or acc_pushes):
            raise RuntimeError("mxu1: compute reads buffer with an active push")
        if kind == "push_fp8" and acc_pushes:
            raise RuntimeError("mxu1: accumulation push target busy")
        path = self._push_path(kind)
        if path and any(self._push_path(op.kind) == path for op in active):
            raise RuntimeError(f"{self.mxu}: {path} push issued while another {path} push is still writing")
        # A reader one cycle behind a push would read each row just before it lands: it waits
        # until the push has written row 0 for the last time (an FP32 push writes it in both phases).
        if (pop or (kind == "compute" and ".acc." in insn.mnemonic)) and any(
                self._push_path(op.kind) == "accumulator" and op.acc == acc
                and self.cycle - op.issued < op.steps - 30 for op in self._ops):
            raise RuntimeError(f"{self.mxu}: accumulator read issued the cycle after a push to it")
        # An FP32 pop reads every row again for columns 16-31: no write to its accumulator before then.
        if (kind == "compute" or kind.startswith("push")) and any(
                op.kind == "pop_fp32" and op.acc == acc and self.cycle - op.issued <= 32 for op in self._ops):
            raise RuntimeError(f"{self.mxu}: accumulator write issued during the first half of an FP32 pop")
        owner = f"{self.name}:{uop.id}"
        phases = [frozenset(_banks(kind, mreg, step)) for step in range(0, steps, 32)]
        for phase, banks in enumerate(phases):
            self.arch_state.conflict_checker.reserve_mreg(f"{owner}/{phase}", reads=frozenset() if pop else banks,
                                                          writes=banks if pop else frozenset())
        mregs = [("mrf", bank) for bank in sorted(frozenset().union(*phases))]
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
        self._ops.append(_Operation(uop, self.cycle, kind, steps, ports, mreg, acc, weight, owner, staged))
        uop.execute_delay = self._execution_latency(uop)
        self._total_instructions += 1
        self.logger.log_stage_start(uop.id, "E", lane=self.lane_id, cycle=self.cycle)

    def _sample(self, op: _Operation, step: int) -> None:
        state = self.arch_state
        row = step % 32
        accumulator = ("acc", self.mxu, op.acc)
        if accumulator in op.staged.operands:
            index = (row, op.columns(step))
            op.staged.sample(accumulator, index, state.acc[self.mxu][op.acc][index])
        if op.kind.startswith("pop"):
            return
        for bank in _banks(op.kind, op.mreg, step):
            state.conflict_checker.access_mreg(self.cycle, bank, row, False, op.owner)
            op.staged.sample(("mrf", bank), row, state.read_mrf_u8(bank)[row])

    def _weight_uses(self, age: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(output column, inner) weights read at this age, and which are row 0's."""
        # Each IPT row reads the whole tile, row r at age r + 1.
        used = torch.full((32, 32), 1 <= age <= 32)
        return used, used & (age == 1)

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

    def _pop_row(self, op: _Operation, step: int) -> None:
        row, banks = step % 32, _banks(op.kind, op.mreg, step)
        for bank in banks:
            self.arch_state.conflict_checker.access_mreg(self.cycle, bank, row, True, op.owner)
        for bank in banks:
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
            if 0 <= age < op.steps:
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
            if op.kind == "compute" and self.compute_first <= age <= self.compute_first + 31:
                compute_writes.append((op, age - self.compute_first))
            if 1 <= age <= op.steps:
                if op.kind == "weight" and (weight_push is None or op.ports[0] > weight_push.ports[0]):
                    weight_push = op
                elif op.kind.startswith("push") and (acc_push is None or op.ports[0] > acc_push.ports[0]):
                    acc_push = op
                elif op.kind.startswith("pop"):
                    self._pop_row(op, age - 1)
        if acc_push is not None and any(op.acc == acc_push.acc for op, _ in compute_writes):
            raise RuntimeError(f"{self.mxu}: compute and push write the same accumulator")
        for op, row in compute_writes:
            self.arch_state.acc[self.mxu][op.acc][row] = op.staged.result(("acc", self.mxu, op.acc))[row]
        if weight_push is not None:
            row = self.cycle - weight_push.issued - 1
            location = ("wb", self.mxu, weight_push.weight)
            self.arch_state.read_wb_u8(self.mxu, weight_push.weight)[row] = weight_push.staged.result(location)[row]
        if acc_push is not None:
            step = self.cycle - acc_push.issued - 1
            index = (step % 32, acc_push.columns(step))
            location = ("acc", self.mxu, acc_push.acc)
            self.arch_state.acc[self.mxu][acc_push.acc][index] = acc_push.staged.result(location)[index]
        remaining = []
        for op in self._ops:
            age = self.cycle - op.issued
            if age == 32 * (op.released + 1):
                self.arch_state.conflict_checker.release_mreg(f"{op.owner}/{op.released}")
                op.released += 1
            last = self.compute_first + 31 if op.kind == "compute" else op.steps
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


class MatrixExecutionUnitInner(_MatrixExecutionUnit):
    mxu = "mxu1"
    exu_type = EXU.MATRIX_INNER
    compute_first = 3
    inflight_depth = 2
