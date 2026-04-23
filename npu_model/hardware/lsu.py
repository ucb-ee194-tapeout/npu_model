from __future__ import annotations
import math

from .exu import ExecutionUnit
from ..logging.logger import Logger, LaneType
from ..hardware.arch_state import ArchState
from ..software.instruction import Uop
from ..isa import EXU
from .stage_data import StageData
from .config import HardwareConfig
from .bank_conflict import mrf_accesses, vmem_accesses

LSU_OP_LATENCIES = {
    "lb": 2,
    "lh": 2,
    "lw": 2,
    "lbu": 2,
    "lhu": 2,
    "sb": 1,
    "sh": 1,
    "sw": 1,
    "seld": 2,
    "vload": 34,
    "vstore": 34,
}


class LoadStoreUnit(ExecutionUnit):
    """
    Load/Store Unit for memory operations.

    Only allows 1 in-flight instruction at a time. When busy, it won't
    claim new instructions, causing backpressure on the pipeline.
    """

    def __init__(
        self,
        name: str,
        logger: Logger,
        arch_state: ArchState,
        lane_id: int = 0,
        config: HardwareConfig | None = None,
    ) -> None:
        super().__init__(name, logger, arch_state, lane_id, config)
        self.reset()

    def can_handle(self, uop: Uop) -> bool:
        return uop.insn.exu == EXU.LSU

    def reset(self) -> None:
        self.in_flight: Uop | None = None
        self._in_flight_mrf_banks: frozenset[int] = frozenset()
        self._in_flight_vmem_banks: frozenset[int] = frozenset()
        self._complete_count = 0
        self._pending_completions: list[Uop] = []
        self._total_instructions = 0
        self._busy_cycles = 0

    def _get_latency(self, uop: Uop) -> int:
        return LSU_OP_LATENCIES[uop.insn.mnemonic]

    def tick(self, idu_output: StageData[Uop | None]) -> None:
        self.cycle += 1
        # Log deferred completions from last cycle
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id, cycle=self.cycle)
            self.logger.log_retire(uop.id)

        self._pending_completions = []
        self._complete_count = 0

        if self.in_flight is None:
            uop = idu_output.peek()
            if uop is not None:
                assert (
                    uop.insn.exu == EXU.LSU
                ), "Non-LSU instruction passed to LoadStoreUnit."
                label = f"{self.name}:{uop.insn.mnemonic}"
                mrf_banks = mrf_accesses(uop.insn)
                vmem_banks = vmem_accesses(uop.insn, self.arch_state)

                checker = self.arch_state.conflict_checker
                checker.acquire_mrf(mrf_banks, label)
                checker.acquire_vmem(vmem_banks, label)
                self._in_flight_mrf_banks = mrf_banks
                self._in_flight_vmem_banks = vmem_banks

                uop.execute_delay = self._get_latency(uop)
                self.in_flight = uop
                self._total_instructions += 1
                self.logger.log_stage_end(
                    uop.id, "D", lane=LaneType.DIU.value, cycle=self.cycle
                )
                self.logger.log_stage_start(
                    uop.id, "E", lane=self.lane_id, cycle=self.cycle
                )

        # Track if EXU was busy
        if self.is_busy():
            self._busy_cycles += 1

        if self.in_flight is not None:
            self.in_flight.execute_delay -= 1
            if self.in_flight.execute_delay <= 0:
                self.in_flight.insn.exec(self.arch_state)

                checker = self.arch_state.conflict_checker
                checker.release_mrf(self._in_flight_mrf_banks)
                checker.release_vmem(self._in_flight_vmem_banks)
                self._in_flight_mrf_banks = frozenset[int]()
                self._in_flight_vmem_banks = frozenset[int]()

                self._complete_count = 1
                self._pending_completions.append(self.in_flight)
                idu_output.claim()
                self.in_flight = None

    def flush_completions(self) -> None:
        """Flush any pending completions (call at end of simulation)."""
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id)
            self.logger.log_retire(uop.id)
        self._pending_completions = []

    def is_busy(self) -> bool:
        """Check if the LSU is busy."""
        return self.in_flight is not None and self.in_flight.insn.mnemonic != "delay"

    @property
    def has_in_flight(self) -> bool:
        """Check if there is an in-flight instruction."""
        return self.in_flight is not None

    @property
    def complete_count(self) -> int:
        """Instructions completed this cycle."""
        return self._complete_count

    @property
    def total_instructions(self) -> int:
        return self._total_instructions

    @property
    def busy_cycles(self) -> int:
        return self._busy_cycles
