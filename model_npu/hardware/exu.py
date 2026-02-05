from abc import abstractmethod
from typing import Optional, List

from .hardware import Module
from .stage_data import StageData
from ..logging.logger import Logger, LaneType
from ..hardware.arch_state import ArchState
from ..software.instruction import Uop
from ..isa import InstructionType
from ..hardware.config import HardwareConfig


class ExecutionUnit(Module):
    """
    Abstract base class for execution units.

    Defines the interface that all execution units must implement.
    Subclass SimpleExecutionUnit for a ready-to-use implementation
    with latency modeling and Kanata logging.
    """

    def __init__(
        self,
        # name for logging purposes
        name: str,
        # handle to the logger
        logger: Logger,
        # handle to the architectural state
        arch_state: ArchState,
        # lane id for logging purposes
        lane_id: int = 0,
        # hardware configuration
        config: HardwareConfig = None,
    ) -> None:
        self.name = name
        self.logger = logger
        self.arch_state = arch_state
        self.lane_id = lane_id
        self.config = config
        self.cycle = 0

    @abstractmethod
    def reset(self) -> None:
        """Reset the execution unit state."""
        pass

    @abstractmethod
    def tick(self, diu_output: StageData[Optional[Uop]]) -> None:
        """Execute one cycle, claiming instruction from DIU output."""
        pass

    @abstractmethod
    def flush_completions(self) -> None:
        """Flush any pending completions (call at end of simulation)."""
        pass

    @abstractmethod
    def has_in_flight(self) -> bool:
        """Check if there are any in-flight instructions."""
        pass

    @property
    @abstractmethod
    def complete_count(self) -> int:
        """Number of instructions completed this cycle."""
        pass

    @property
    @abstractmethod
    def total_instructions(self) -> int:
        """Total instructions executed."""
        pass

    @property
    @abstractmethod
    def busy_cycles(self) -> int:
        """Number of cycles the EXU was busy."""
        pass

    @property
    @abstractmethod
    def supported_instruction_types(self) -> List[InstructionType]:
        """List of instruction types supported by the execution unit."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class ScalarExecutionUnit(ExecutionUnit):
    """
    Execution unit for scalar operations.
    Always executes 1 scalar instruction per cycle.
    Execute delay is always 1 cycle.
    """

    def __init__(
        self,
        name: str,
        logger: Logger,
        arch_state: ArchState,
        lane_id: int = 0,
        config: HardwareConfig = None,
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
        # variables
        self._pending_completion_uop: Optional["Uop"] = None
        # logging variables
        self._total_instructions = 0
        self._busy_cycles = 0

    def tick(self, idu_output: StageData[Optional["Uop"]]) -> None:
        self.cycle += 1
        # Log deferred completions from last cycle
        if self._pending_completion_uop is not None:
            self.logger.log_stage_end(
                self._pending_completion_uop.id,
                "E",
                lane=self.lane_id,
                cycle=self.cycle,
            )
            self.logger.log_retire(self._pending_completion_uop.id)
            self._pending_completion_uop = None

        # reset cycle states
        self._complete_count = 0

        # Claim instruction from DIU
        uop = idu_output.claim()

        # Accept new instruction
        if uop is not None:
            # tag instruction with execution delay
            uop.execute_delay = 1
            self._pending_completion_uop = uop
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

            self._busy_cycles += uop.insn.mnemonic != "nop"
            self._complete_count = 1
            # execute the instruction and modify the arch state

            uop.execute_fn(self.arch_state, uop.insn.args)

    def flush_completions(self) -> None:
        """Flush any pending completions (call at end of simulation)."""
        if self._pending_completion_uop is not None:
            self.logger.log_stage_end(
                self._pending_completion_uop.id,
                "E",
                lane=self.lane_id,
                cycle=self.cycle,
            )
            self.logger.log_retire(self._pending_completion_uop.id)
            self._pending_completion_uop = None

    @property
    def total_instructions(self) -> int:
        """Total instructions executed."""
        return self._total_instructions

    @property
    def busy_cycles(self) -> int:
        """Number of cycles the EXU was busy."""
        return self._busy_cycles

    @property
    def complete_count(self) -> int:
        """Number of instructions completed this cycle."""
        return self._complete_count

    @property
    def has_in_flight(self) -> bool:
        return False

    @property
    def supported_instruction_types(self) -> List[InstructionType]:
        return [InstructionType.SCALAR]
