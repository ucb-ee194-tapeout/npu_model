from dataclasses import dataclass

from .exu import ExecutionUnit
from ..logging.logger import Logger, LaneType
from ..hardware.arch_state import ArchState
from ..software.instruction import Uop
from ..isa import EXU
from .stage_data import StageData
from .config import HardwareConfig
from ..isa_patterns import (
    MXUAccumulatorPop,
    MXUAccumulatorPopE1,
    MXUAccumulatorPush,
    MXUMatMul,
    MXUWeightPush,
)


MXU_OP_LATENCIES = {
    "vmatpush.weight.mxu0": 32,
    "vmatpush.acc.fp8.mxu0": 32,
    "vmatpush.acc.bf16.mxu0": 32,
    "vmatmul.acc.mxu0": 96,
    "vmatmul.mxu0": 96,
    "vmatpop.fp8.acc.mxu0": 32,
    "vmatpop.bf16.acc.mxu0": 32,
    "vmatpush.weight.mxu1": 32,
    "vmatpush.acc.fp8.mxu1": 32,
    "vmatpush.acc.bf16.mxu1": 32,
    "vmatmul.acc.mxu1": 35,
    "vmatmul.mxu1": 35,
    "vmatpop.fp8.acc.mxu1": 32,
    "vmatpop.bf16.acc.mxu1": 32,
}

MXU_MAX_RECENT_SPACING = 64
MXU_MRF_BUSY_CYCLES = 32
MXU_MAX_OUTSTANDING_MATMULS = 3
MXU_PORT_BUSY_CYCLES = 32
MXU_PORT_STRIDE_CYCLES = 33


@dataclass
class _InFlightMXUOp:
    uop: Uop
    issue_cycle: int
    label: str
    mrf_reads: frozenset[int]
    mrf_writes: frozenset[int]
    mrf_release_cycle: int
    mrf_released: bool = False
    port_release_cycle: int = 0


@dataclass(frozen=True)
class _IssuedMXUOp:
    uop: Uop
    issue_cycle: int
    port_release_cycle: int


class _MatrixExecutionUnit(ExecutionUnit):
    target_exu: EXU
    display_name: str

    def __init__(
        self,
        name: str,
        logger: Logger,
        arch_state: ArchState,
        lane_id: int = 0,
        config: HardwareConfig | None = None,
    ) -> None:
        super().__init__(
            name,
            logger,
            arch_state,
            lane_id,
            config,
        )
        self.reset()

    def can_handle(self, uop: Uop) -> bool:
        return True

    def reset(self) -> None:
        self.in_flight: list[_InFlightMXUOp] = []
        self._recent_issues: list[_IssuedMXUOp] = []
        self._complete_count = 0
        self._pending_completions: list[Uop] = []
        self._total_instructions = 0
        self._busy_cycles = 0
        self._stalled = False

    def _execution_latency(self, uop: Uop) -> int:
        return MXU_OP_LATENCIES[uop.insn.mnemonic]

    def can_accept(self, uop: Uop) -> bool:
        if uop.insn.mnemonic == "delay":
            return True

        next_issue_cycle = self.cycle + 1
        self._prune_recent_issues(next_issue_cycle)

        if (
            self._is_matmul(uop)
            and self._outstanding_matmuls() >= MXU_MAX_OUTSTANDING_MATMULS
        ):
            return False

        for issued in self._recent_issues:
            spacing = self._minimum_spacing(issued.uop, uop)
            if next_issue_cycle < issued.issue_cycle + spacing:
                return False

        return True

    def tick(self, idu_output: StageData[Uop | None]) -> None:
        self.cycle += 1

        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id, cycle=self.cycle)
            self.logger.log_retire(uop.id)
        self._pending_completions = []
        self._complete_count = 0

        self._release_mrf_resources()
        self._prune_recent_issues(self.cycle + 1)

        if self.can_accept_pending(idu_output):
            uop = idu_output.peek()
            assert uop is not None
            assert (
                uop.insn.exu == self.target_exu
            ), f"Non-Matrix instruction passed to {self.display_name}"
            self._issue(uop)
            idu_output.claim()
            self._stalled = False
        elif idu_output.peek() is not None:
            self._stalled = True
        else:
            self._stalled = False

        if self.is_busy():
            self._busy_cycles += 1

        remaining: list[_InFlightMXUOp] = []
        for entry in self.in_flight:
            entry.uop.execute_delay -= 1
            if entry.uop.execute_delay <= 0:
                entry.uop.insn.exec(self.arch_state)
                self._complete_count += 1
                if not entry.mrf_released:
                    self.arch_state.conflict_checker.release_mrf_access(
                        entry.mrf_reads, entry.mrf_writes, entry.label
                    )
                self._pending_completions.append(entry.uop)
            else:
                remaining.append(entry)
        self.in_flight = remaining

    def can_accept_pending(self, idu_output: StageData[Uop | None]) -> bool:
        uop = idu_output.peek()
        return uop is not None and self.can_accept(uop)

    def _issue(self, uop: Uop) -> None:
        label = f"{self.name}:{uop.insn.mnemonic}"
        mrf_reads = self._mrf_reads(uop)
        mrf_writes = self._mrf_writes(uop)
        self.arch_state.conflict_checker.acquire_mrf_access(
            mrf_reads, mrf_writes, label
        )

        uop.execute_delay = self._execution_latency(uop)
        port_release_cycle = self.cycle + MXU_PORT_BUSY_CYCLES
        if self._is_matmul(uop) or self._is_weight_push(uop) or self._is_acc_push(uop):
            port_release_cycle = self.cycle + MXU_PORT_BUSY_CYCLES
        entry = _InFlightMXUOp(
            uop=uop,
            issue_cycle=self.cycle,
            label=label,
            mrf_reads=mrf_reads,
            mrf_writes=mrf_writes,
            mrf_release_cycle=self.cycle + MXU_MRF_BUSY_CYCLES,
            port_release_cycle=port_release_cycle,
        )
        self.in_flight.append(entry)
        self._recent_issues.append(
            _IssuedMXUOp(
                uop=uop, issue_cycle=self.cycle, port_release_cycle=port_release_cycle
            )
        )
        self._total_instructions += 1

        self.logger.log_stage_end(
            uop.id,
            "D",
            lane=LaneType.DIU.value,
            cycle=self.cycle,
        )
        self.logger.log_stage_start(
            uop.id,
            "E",
            lane=self.lane_id,
            cycle=self.cycle,
        )

    def _release_mrf_resources(self) -> None:
        for entry in self.in_flight:
            if not entry.mrf_released and self.cycle >= entry.mrf_release_cycle:
                self.arch_state.conflict_checker.release_mrf_access(
                    entry.mrf_reads, entry.mrf_writes, entry.label
                )
                entry.mrf_released = True

    def _prune_recent_issues(self, reference_cycle: int) -> None:
        self._recent_issues = [
            issued
            for issued in self._recent_issues
            if reference_cycle < issued.issue_cycle + MXU_MAX_RECENT_SPACING
        ]

    def _outstanding_matmuls(self) -> int:
        return sum(1 for entry in self.in_flight if self._is_matmul(entry.uop))

    def _minimum_spacing(self, prev: Uop, next_uop: Uop) -> int:
        spacing = 1
        if self._is_matmul(prev) and self._is_matmul(next_uop):
            spacing = max(spacing, MXU_PORT_STRIDE_CYCLES)
            if self._same_acc(prev, next_uop):
                spacing = max(spacing, 64)

        if self._is_matmul(prev) and self._is_weight_push(next_uop):
            if self._same_wslot(prev, next_uop):
                spacing = max(spacing, 63)
            else:
                spacing = max(spacing, 1)

        if self._is_weight_push(prev) and self._is_matmul(next_uop):
            if self._same_wslot(prev, next_uop):
                spacing = max(spacing, 1)

        if self._is_matmul(prev) and self._is_acc_push(next_uop):
            if self._same_acc(prev, next_uop):
                spacing = max(spacing, 64)

        if self._is_matmul(prev) and self._is_acc_pop(next_uop):
            if self._same_acc(prev, next_uop):
                spacing = max(spacing, 64)

        if self._is_acc_push(prev) and self._is_matmul(next_uop):
            if self._same_acc(prev, next_uop):
                spacing = max(spacing, 1)

        if self._is_acc_push(prev) and self._is_acc_pop(next_uop):
            if self._same_acc(prev, next_uop):
                spacing = max(spacing, 33)

        if self._is_fp8_acc_pop(prev) and self._is_matmul(next_uop):
            if self._same_acc(prev, next_uop):
                spacing = max(spacing, 1)

        if self._is_fp8_acc_pop(prev) and self._is_acc_pop(next_uop):
            spacing = max(spacing, 33)

        if self._is_bf16_acc_pop(prev) and self._is_acc_pop(next_uop):
            spacing = max(spacing, 33)

        prev_reads = self._mrf_reads(prev)
        prev_writes = self._mrf_writes(prev)
        next_reads = self._mrf_reads(next_uop)
        next_writes = self._mrf_writes(next_uop)
        if (prev_writes & next_reads) or (prev_reads & next_writes):
            spacing = max(spacing, 33)

        return spacing

    def _uses_p_read_port(self, uop: Uop) -> bool:
        return (
            self._is_matmul(uop)
            or self._is_weight_push(uop)
            or self._is_acc_push(uop)
        )

    def _mrf_reads(self, uop: Uop) -> frozenset[int]:
        insn = uop.insn
        if isinstance(insn, MXUWeightPush):
            return frozenset({int(insn.vs1)})
        if isinstance(insn, MXUAccumulatorPush):
            if self._is_bf16_acc_push(uop):
                return frozenset({int(insn.vs1), int(insn.vs1) + 1})
            return frozenset({int(insn.vs1)})
        if isinstance(insn, MXUMatMul):
            return frozenset({int(insn.vs1)})
        return frozenset()

    def _mrf_writes(self, uop: Uop) -> frozenset[int]:
        insn = uop.insn
        if isinstance(insn, MXUAccumulatorPopE1):
            return frozenset({int(insn.vd)})
        if isinstance(insn, MXUAccumulatorPop):
            return frozenset({int(insn.vd), int(insn.vd) + 1})
        return frozenset()

    def _same_acc(self, prev: Uop, next_uop: Uop) -> bool:
        prev_acc = self._acc_slot(prev.insn)
        next_acc = self._acc_slot(next_uop.insn)
        return prev_acc is not None and prev_acc == next_acc

    def _same_wslot(self, prev: Uop, next_uop: Uop) -> bool:
        prev_wslot = self._wslot(prev.insn)
        next_wslot = self._wslot(next_uop.insn)
        return prev_wslot is not None and prev_wslot == next_wslot

    def _acc_slot(self, insn) -> int | None:
        if isinstance(insn, (MXUAccumulatorPush, MXUMatMul)):
            return int(insn.vd)
        if isinstance(insn, (MXUAccumulatorPop, MXUAccumulatorPopE1)):
            return int(insn.vs2)
        return None

    def _wslot(self, insn) -> int | None:
        if isinstance(insn, MXUWeightPush):
            return int(insn.vd)
        if isinstance(insn, MXUMatMul):
            return int(insn.vs2)
        return None

    def _is_matmul(self, uop: Uop) -> bool:
        return isinstance(uop.insn, MXUMatMul)

    def _is_acc_push(self, uop: Uop) -> bool:
        return isinstance(uop.insn, MXUAccumulatorPush)

    def _is_weight_push(self, uop: Uop) -> bool:
        return isinstance(uop.insn, MXUWeightPush)

    def _is_acc_pop(self, uop: Uop) -> bool:
        return isinstance(uop.insn, (MXUAccumulatorPop, MXUAccumulatorPopE1))

    def _is_fp8_acc_pop(self, uop: Uop) -> bool:
        return isinstance(uop.insn, MXUAccumulatorPopE1)

    def _is_bf16_acc_pop(self, uop: Uop) -> bool:
        return isinstance(uop.insn, MXUAccumulatorPop)

    def _is_bf16_acc_push(self, uop: Uop) -> bool:
        return uop.insn.mnemonic.startswith("vmatpush.acc.bf16")

    def flush_completions(self) -> None:
        for entry in self.in_flight:
            if not entry.mrf_released:
                self.arch_state.conflict_checker.release_mrf_access(
                    entry.mrf_reads, entry.mrf_writes, entry.label
                )
                entry.mrf_released = True
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id)
            self.logger.log_retire(uop.id)
        self._pending_completions = []

    def is_busy(self) -> bool:
        return (
            any(entry.uop.insn.mnemonic != "delay" for entry in self.in_flight)
            or self._stalled
        )

    @property
    def has_in_flight(self) -> bool:
        return len(self.in_flight) > 0

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
    """MXU0: Execution unit for matrix operations."""

    target_exu = EXU.MATRIX_SYSTOLIC
    display_name = "MXU0"


class MatrixExecutionUnitInner(_MatrixExecutionUnit):
    """MXU1: Execution unit for matrix operations."""

    target_exu = EXU.MATRIX_INNER
    display_name = "MXU1"
