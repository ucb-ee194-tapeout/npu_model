import math

from .exu import ExecutionUnit
from ..logging.logger import Logger, LaneType
from ..hardware.arch_state import ArchState
from ..software.instruction import Uop
from ..isa import EXU
from .stage_data import StageData
from .config import HardwareConfig
from .bank_conflict import mrf_accesses, weight_buffer_accesses, acc_buffer_accesses


LOCAL_TRANSFER_TILE_BYTES = {
    "vmatpush.weight.mxu0": 1024,
    "vmatpush.weight.mxu1": 1024,
    "vmatpush.acc.fp8.mxu0": 1024,
    "vmatpush.acc.fp8.mxu1": 1024,
    "vmatpush.acc.bf16.mxu0": 2048,
    "vmatpush.acc.bf16.mxu1": 2048,
    "vmatpop.fp8.acc.mxu0": 1024,
    "vmatpop.fp8.acc.mxu1": 1024,
    "vmatpop.bf16.acc.mxu0": 2048,
    "vmatpop.bf16.acc.mxu1": 2048,
    "vmatpop.mxu0": 2048,
    "vmatpop.mxu1": 2048,
}


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

class MatrixExecutionUnitSystolic(ExecutionUnit):
    """MXU0: Execution unit for matrix operations."""

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
        self.in_flight: Uop | None = None
        self._in_flight_mrf_banks: frozenset[int] = frozenset()
        self._in_flight_weight_banks: frozenset[int] = frozenset()
        self._in_flight_acc_banks: frozenset[int] = frozenset()
        self._complete_count = 0
        self._pending_completions: list[Uop] = []
        self._total_instructions = 0
        self._busy_cycles = 0

    def _execution_latency(self, uop: Uop) -> int:
        return MXU_OP_LATENCIES.get(uop.insn.mnemonic, 32)

    def tick(self, idu_output: StageData[Uop | None]) -> None:
        self.cycle += 1
        # Log deferred completions from last cycle
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id, cycle=self.cycle)
            self.logger.log_retire(uop.id)

        self._pending_completions = []

        self._complete_count = 0

        if self.in_flight is None:
            # Peek instruction from DIU
            uop = None
            if self.in_flight is None:
                uop = idu_output.peek()

            # Accept new instruction
            if uop is not None:
                assert uop.insn.exu == EXU.MATRIX_SYSTOLIC, "Non-Matrix instruction passed to MXU0"

                label = f"{self.name}:{uop.insn.mnemonic}"
                mrf_banks = mrf_accesses(uop.insn)
                weight_banks = weight_buffer_accesses(uop.insn)
                acc_banks = acc_buffer_accesses(uop.insn)

                self.arch_state.conflict_checker.acquire_mrf(mrf_banks, label)
                self.arch_state.conflict_checker.acquire_weight_buf(weight_banks, label)
                self.arch_state.conflict_checker.acquire_acc_buf(acc_banks, label)

                self._in_flight_mrf_banks = mrf_banks
                self._in_flight_weight_banks = weight_banks
                self._in_flight_acc_banks = acc_banks

                uop.execute_delay = self._execution_latency(uop)
                self.in_flight = uop
                self._total_instructions += 1
                # Log: end dispatch, start execute
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

        # Track if EXU was busy
        if self.is_busy():
            self._busy_cycles += 1

        # Process in-flight instructions
        if self.in_flight:
            self.in_flight.execute_delay -= 1
            if self.in_flight.execute_delay <= 0:
                # execute the instruction
                self.in_flight.insn.exec(self.arch_state)
                self._complete_count = 1

                # Release acquired MRF/acc/weight banks before retiring the instruction.
                self.arch_state.conflict_checker.release_mrf(self._in_flight_mrf_banks)
                self.arch_state.conflict_checker.release_weight_buf(
                    self._in_flight_weight_banks
                )
                self.arch_state.conflict_checker.release_acc_buf(
                    self._in_flight_acc_banks
                )

                self._in_flight_mrf_banks = frozenset[int]()
                self._in_flight_weight_banks = frozenset[int]()
                self._in_flight_acc_banks = frozenset[int]()
                # Defer completion logging to next tick
                self._pending_completions.append(self.in_flight)
                # claim the uop from the DIU
                idu_output.claim()
                self.in_flight = None

    def flush_completions(self) -> None:
        """Flush any pending completions (call at end of simulation)."""
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id)
            self.logger.log_retire(uop.id)
        self._pending_completions = []

    def is_busy(self) -> bool:
        """Check if the EXU is busy."""
        return self.in_flight is not None and self.in_flight.insn.mnemonic != "delay"

    @property
    def has_in_flight(self) -> bool:
        """Check if there are any in-flight instructions."""
        return self.in_flight is not None

    @property
    def complete_count(self) -> int:
        """Instructions completed this cycle."""
        return self._complete_count

    @property
    def total_instructions(self) -> int:
        """Total instructions executed."""
        return self._total_instructions

    @property
    def busy_cycles(self) -> int:
        """Number of cycles the EXU was busy."""
        return self._busy_cycles


class MatrixExecutionUnitInner(ExecutionUnit):
    """MXU1: Execution unit for matrix operations."""

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
        self.in_flight: Uop | None = None
        self._in_flight_mrf_banks: frozenset[int] = frozenset()
        self._in_flight_weight_banks: frozenset[int] = frozenset()
        self._in_flight_acc_banks: frozenset[int] = frozenset()
        self._complete_count = 0
        self._pending_completions: list[Uop] = []
        self._total_instructions = 0
        self._busy_cycles = 0

    def _execution_latency(self, uop: Uop) -> int:
        mnemonic = uop.insn.mnemonic
        if mnemonic in LOCAL_TRANSFER_TILE_BYTES:
            return max(
                1,
                math.ceil(
                    LOCAL_TRANSFER_TILE_BYTES[mnemonic]
                    / self.config.vmem_bytes_per_cycle
                ),
            )
        return self.config.mxu1_matmul_latency_cycles

    def tick(self, idu_output: StageData[Uop | None]) -> None:
        self.cycle += 1
        # Log deferred completions from last cycle
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id, cycle=self.cycle)
            self.logger.log_retire(uop.id)

        self._pending_completions = []

        self._complete_count = 0

        if self.in_flight is None:
            # Peek instruction from DIU
            uop = None
            if self.in_flight is None:
                uop = idu_output.peek()

            # Accept new instruction
            if uop is not None:
                assert uop.insn.exu == EXU.MATRIX_INNER, "Non-Matrix instruction passed to MXU1"

                label = f"{self.name}:{uop.insn.mnemonic}"
                mrf_banks = mrf_accesses(uop.insn)
                weight_banks = weight_buffer_accesses(uop.insn)
                acc_banks = acc_buffer_accesses(uop.insn)

                self.arch_state.conflict_checker.acquire_mrf(mrf_banks, label)
                self.arch_state.conflict_checker.acquire_weight_buf(weight_banks, label)
                self.arch_state.conflict_checker.acquire_acc_buf(acc_banks, label)

                self._in_flight_mrf_banks = mrf_banks
                self._in_flight_weight_banks = weight_banks
                self._in_flight_acc_banks = acc_banks

                uop.execute_delay = self._execution_latency(uop)
                self.in_flight = uop
                self._total_instructions += 1
                # Log: end dispatch, start execute
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

        # Track if EXU was busy
        if self.is_busy():
            self._busy_cycles += 1

        # Process in-flight instructions
        if self.in_flight:
            self.in_flight.execute_delay -= 1
            if self.in_flight.execute_delay <= 0:
                # execute the instruction
                self.in_flight.insn.exec(self.arch_state)
                self._complete_count = 1
                # Release acquired MRF/acc/weight banks before retiring the instruction.
                self.arch_state.conflict_checker.release_mrf(self._in_flight_mrf_banks)
                self.arch_state.conflict_checker.release_weight_buf(
                    self._in_flight_weight_banks
                )
                self.arch_state.conflict_checker.release_acc_buf(
                    self._in_flight_acc_banks
                )

                self._in_flight_mrf_banks = frozenset[int]()
                self._in_flight_weight_banks = frozenset[int]()
                self._in_flight_acc_banks = frozenset[int]()
                # Defer completion logging to next tick
                self._pending_completions.append(self.in_flight)
                # claim the uop from the DIU
                idu_output.claim()
                self.in_flight = None

    def flush_completions(self) -> None:
        """Flush any pending completions (call at end of simulation)."""
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id)
            self.logger.log_retire(uop.id)
        self._pending_completions = []

    def is_busy(self) -> bool:
        """Check if the EXU is busy."""
        return self.in_flight is not None and self.in_flight.insn.mnemonic != "delay"

    @property
    def has_in_flight(self) -> bool:
        """Check if there are any in-flight instructions."""
        return self.in_flight is not None

    @property
    def complete_count(self) -> int:
        """Instructions completed this cycle."""
        return self._complete_count

    @property
    def total_instructions(self) -> int:
        """Total instructions executed."""
        return self._total_instructions

    @property
    def busy_cycles(self) -> int:
        """Number of cycles the EXU was busy."""
        return self._busy_cycles
