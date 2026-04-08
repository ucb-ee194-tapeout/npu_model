from typing import List, Any
import math

from .exu import ExecutionUnit
from ..logging.logger import Logger, LaneType
from .arch_state import ArchState
from ..software.instruction import Uop, is_dma_uop
from ..isa import InstructionType, AsmInstructionType, DmaArgs
from .stage_data import StageData
from .config import HardwareConfig


class DmaExecutionUnit(ExecutionUnit):
    """Execution unit for matrix operations."""

    def _bytes_for_dma_uop(self, uop: Uop[DmaArgs]) -> int:
        """
        Determine transfer size in bytes for DMA ops.

        Current programs conventionally place the byte count in XRF[rs2] for
        dma.load/store (R-type).
        """
        args = uop.insn.args
        if getattr(args, "rs2", 0):
            # print("size:", int(self.arch_state.read_xrf(args.rs2)))
            return int(self.arch_state.read_xrf(args.rs2))
        return 0

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

    def reset(self) -> None:
        self.in_flight: List[Uop[DmaArgs]] = []
        self._complete_count = 0
        self._pending_completions: List[Uop[DmaArgs]] = []
        self._total_instructions = 0
        self._busy_cycles = 0

    def tick(self, idu_output: StageData[Uop[Any] | None]) -> None:
        self.cycle += 1
        # Log deferred completions from last cycle
        for uop in self._pending_completions:
            # FIXME: We should really rewrite this entire thing, this is a terrible way to do args.
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id, cycle=self.cycle)
            self.logger.log_retire(uop.id)
            # clear the flag
            self.arch_state.clear_flag(uop.insn.args.channel)
            print(f"DMA {self.name} cleared flag {uop.insn.args.channel}")

            if len(self.in_flight) != 0:
                # Log: start execute
                self.logger.log_stage_start(
                    self.in_flight[0].id,
                    "E",
                    lane=self.lane_id,
                    cycle=self.cycle,
                )

        self._pending_completions = []

        self._complete_count = 0

        # If there are less than 8 instructions queued, check if we can queue more.
        if len(self.in_flight) < 8:
            uop = None
            if len(self.in_flight) < 8:
                uop = idu_output.peek()

            # Accept new instruction
            if uop is not None:
                assert is_dma_uop(uop), "Invalid arguments passed to DMA Engine"
                # tag instruction with execution delay
                if uop.insn.mnemonic == "dma.config.ch<N>":
                    # Config is a control op; keep it fixed-latency.
                    uop.execute_delay = 1
                else:
                    nbytes = self._bytes_for_dma_uop(uop)
                    uop.execute_delay = max(
                        1,
                        math.ceil(nbytes / self.config.vmem_bytes_per_cycle),
                    )
                self.in_flight.append(uop)
                self._total_instructions += 1

                # claim the uop from the DIU
                # I think this needs to happen here since our entire goal
                # with doing this is to not block. Not 100% sure.
                idu_output.claim()

                # Log: End dispatch
                self.logger.log_stage_end(
                    uop.id,
                    "D",
                    lane=LaneType.DIU.value,
                    cycle=self.cycle,
                )

                if len(self.in_flight) == 1:
                    # Log: start execute
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
        if len(self.in_flight) != 0:
            self.in_flight[0].execute_delay -= 1
            if self.in_flight[0].execute_delay <= 0:
                # execute the instruction
                if self.in_flight[0].execute_fn != None:
                    self.in_flight[0].execute_fn(
                        self.arch_state, self.in_flight[0].insn.args
                    )
                else:
                    raise ValueError("No execute function specified for Uop.")
                self._complete_count = 1
                # Defer completion logging to next tick
                self._pending_completions.append(self.in_flight[0])
                # print(f"MXU {self.name} completed instruction {self.in_flight[0].id}")
                self.in_flight = self.in_flight[1:]

    def flush_completions(self) -> None:
        """Flush any pending completions (call at end of simulation)."""
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id)
            self.logger.log_retire(uop.id)
        self._pending_completions = []

    def is_busy(self) -> bool:
        """Check if the EXU is busy."""
        return len(self.in_flight) != 0 and self.in_flight[0].insn.mnemonic != "delay"

    @property
    def has_in_flight(self) -> bool:
        """Check if there are any in-flight instructions."""
        return len(self.in_flight) != 0

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

    @property
    def supported_instruction_types(self) -> List[AsmInstructionType]:
        return [InstructionType.DMA.R, InstructionType.DMA.I, InstructionType.BARRIER.I]
