from .hardware import Module
from .stage_data import StageData
from ..software.program import Program
from ..software.instruction import Uop
from ..logging.logger import Logger, LaneType
from ..hardware.arch_state import ArchState
from typing import Optional


class InstructionFetch(Module):
    """
    Instruction Fetch Unit.
    Fetches up to `width` instructions per cycle from the program.
    Logs fetch events to Kanata trace.

    Uses StageData for output - downstream must claim before new fetch.
    When stalled, ends F stage early - gap in trace shows stall period.
    """

    def __init__(
        self,
        width: int,
        logger: Logger,
        arch_state: ArchState,
    ) -> None:
        self.width = width
        self.logger = logger
        self.arch_state = arch_state
        self.program = None
        self.cycle = 0
        self.reset()

    def load_program(self, program: Program):
        self.program = program

    def reset(self) -> None:
        self.output: StageData[Optional[Uop]] = StageData(None)
        self.arch_state.set_pc(0)
        self._stalled = False

    def is_finished(self) -> bool:
        """Check if all instructions have been fetched."""
        return (
            self.program.is_finished(self.arch_state.pc) and not self.output.is_valid()
        )

    def tick(self) -> None:
        """
        Fetch instructions from the program.
        """
        self.cycle += 1
        # Stall if downstream hasn't claimed our output
        if self.output.should_stall():
            if not self._stalled:
                # Just started stalling - end F stage for waiting insns
                uop = self.output.peek()
                if uop is not None:
                    self.logger.log_stage_end(
                        uop.id, "F", lane=LaneType.IFU.value, cycle=self.cycle
                    )
            self._stalled = True
            return

        self._stalled = False

        # Nothing more to fetch
        if self.program.is_finished(self.arch_state.pc):
            self.output.prepare(None)
            return

        fetched_instruction = self.program.get_instruction(self.arch_state.pc)

        uop = Uop(fetched_instruction)

        # Log instruction and start fetch stage
        self.logger.log_insn(uop.id, str(uop.insn))
        self.logger.log_stage_start(
            uop.id, "F", lane=LaneType.IFU.value, cycle=self.cycle
        )

        self.output.prepare(uop)

        # set the program counter to the next instruction, only if we did not stall
        self.arch_state.set_pc(self.arch_state.npc)

    @property
    def is_stalled(self) -> bool:
        """Check if IFU is currently stalled."""
        return self._stalled
