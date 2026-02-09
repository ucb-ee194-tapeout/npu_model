from .hardware import Module
from .stage_data import StageData
from typing import List, Dict, Type, Optional, TYPE_CHECKING
from .exu import ExecutionUnit
from ..logging.logger import Logger, LaneType
from ..isa import IsaSpec
from ..hardware.arch_state import ArchState
from ..isa import InstructionType
from .exu import *  # noqa: F401, F403

if TYPE_CHECKING:
    from ..software.insn import Uop


class InstructionDecode(Module):
    """
    Dispatch Unit.
    Routes fetched instructions to their target execution units.
    Logs dispatch events (end F, start D) to Kanata trace.

    Uses StageData for both input (claimed from IFU) and output (per-EXU).
    When stalled, ends D stage early - gap in trace shows stall period.
    """

    def __init__(
        self,
        exus: List[ExecutionUnit],
        logger: Logger,
        isa: IsaSpec,
        arch_state: ArchState,
        dispatch_strategy: str
    ) -> None:
        self.exus = exus
        self.logger = logger
        self.isa = isa
        self.arch_state = arch_state
        self.uop = None  # the current uop in flight
        self.lane_id = 1
        self.cycle = 0
        self.dispatch_strategy = dispatch_strategy
        # Build a mapping from EXU type to EXU instances
        self.exu_map: Dict[InstructionType, List[ExecutionUnit]] = {}
        for exu in exus:
            exu_types = exu.supported_instruction_types
            for t in exu_types:
                if t not in self.exu_map:
                    self.exu_map[t] = []
                self.exu_map[t].append(exu)
        self.reset()

    def reset(self) -> None:
        # Per-EXU output stage data
        self.outputs: Dict[ExecutionUnit, StageData[Optional["Uop"]]] = {
            exu: StageData(None) for exu in self.exus
        }
        self._stalled = False

    def is_finished(self) -> bool:
        """Check if DIU is finished."""
        return self.uop is None and all(
            not output.is_valid() for output in self.outputs.values()
        )

    def tick(self, ifu_output: StageData[List["Uop"]]) -> None:
        """
        Dispatch fetched instructions to their target execution units.
        Logs: E (end F stage), S (start D stage)
        When stalled, ends D stage early to show gap.
        """
        self.cycle += 1
        # check if we currently have a uop in flight
        if self.uop is not None:

            # check for dispatch delay
            if self.uop.dispatch_delay > 0:
                self.uop.dispatch_delay -= 1
                return

            if self.check_backpressure(self.uop):
                return

            # dispatch uop
            self.dispatch()
            self._stalled = False

        else:
            uop = ifu_output.claim()
            if uop is None:
                return

            instr_definition = self.isa.operations[uop.insn.mnemonic]
            uop.execute_fn = instr_definition.effect

            self.logger.log_stage_end(
                uop.id, "F", lane=LaneType.IFU.value, cycle=self.cycle
            )
            self.logger.log_stage_start(
                uop.id, "D", lane=LaneType.DIU.value, cycle=self.cycle
            )

            # tag instruction with dispatch delay
            uop.dispatch_delay = uop.insn.delay
            self.uop = uop

            if uop.dispatch_delay > 0:
                self.uop.dispatch_delay -= 1
                self._stalled = True
                return
            else:
                if self.check_backpressure(uop):
                    return
                self.dispatch()

    @property
    def is_stalled(self) -> bool:
        """Check if DIU is currently stalled."""
        return self._stalled

    def dispatch(self) -> None:
        assert self.uop is not None
        assert self.uop.dispatch_delay == 0
        # Prepare empty outputs for this cycle
        exu_type = self.isa.operations[self.uop.insn.mnemonic].instruction_type

        if exu_type == InstructionType.BARRIER:
            self.logger.log_stage_end(
                self.uop.id, "D", lane=LaneType.DIU.value, cycle=self.cycle + 1
            )
            self.uop = None
            return

        target_exu = self.choose_target_exu(exu_type)
        self.outputs[target_exu].prepare(self.uop)

        # if we dispatched a DMA instruction, set flag as busy here
        if exu_type == InstructionType.DMA:
            assert (
                self.arch_state.check_flag(self.uop.insn.args["flag"]) == False
            ), f"Flag {self.uop.insn.args['flag']} is already set, erroneous program"
            self.arch_state.set_flag(self.uop.insn.args["flag"])
        self.uop = None

    def claim_uop(self, ifu_output: StageData[Optional["Uop"]]) -> None:
        """Claim a new uop from IFU"""
        assert self.uop is None

    def check_backpressure(self, uop: "Uop") -> bool:
        instr_definition = self.isa.operations[uop.insn.mnemonic]
        exu_type = instr_definition.instruction_type

        if exu_type == InstructionType.BARRIER:
            if self.arch_state.check_flag(uop.insn.args["flag"]):
                self._stalled = True
                return True
            else:
                self._stalled = False
                return False

        exu_list = self.exu_map[exu_type]
        target_exu = exu_list[0] # always choose the first EXU
        if self.outputs[target_exu].should_stall():
            # Don't end D stage - keep it active to show instruction is waiting
            # The D stage will end when we actually dispatch
            self._stalled = True
            return True

        # Backpressure cleared - if we were stalled, the D stage continues:
        self._stalled = False
        return False
    
    def choose_target_exu(self, exu_type: InstructionType) -> ExecutionUnit:
        exu_list = self.exu_map[exu_type]
        if self.dispatch_strategy == "round_robin":
            return exu_list[self.cycle % len(exu_list)]
        elif self.dispatch_strategy == "greedy":
            # always choose the first EXU that is not busy
            target_exu = exu_list[0]
            for exu in exu_list:
                if not exu.is_busy():
                    return exu
            return target_exu
        elif self.dispatch_strategy == "dummy":
            return exu_list[0]
        else:
            raise ValueError(f"Invalid dispatch strategy: {self.dispatch_strategy}")
