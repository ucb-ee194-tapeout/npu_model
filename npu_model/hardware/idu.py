from __future__ import annotations
from .hardware import Module
from .stage_data import StageData
from typing import TYPE_CHECKING
from .exu import ExecutionUnit
from ..logging.logger import Logger, LaneType
from ..isa import IsaSpec, is_scalar_itype, RType, SBType, UJType
from ..configs.isa_definition import (
    DELAY,
    DMA_WAIT_CH0,
    DMA_WAIT_CH1,
    DMA_WAIT_CH2,
    DMA_WAIT_CH3,
    DMA_WAIT_CH4,
    DMA_WAIT_CH5,
    DMA_WAIT_CH6,
    DMA_WAIT_CH7,
)
from ..isa_types import EXU
from ..hardware.arch_state import ArchState
from .exu import *  # noqa: F401, F403


_DMA_WAIT_TYPES = (
    DMA_WAIT_CH0,
    DMA_WAIT_CH1,
    DMA_WAIT_CH2,
    DMA_WAIT_CH3,
    DMA_WAIT_CH4,
    DMA_WAIT_CH5,
    DMA_WAIT_CH6,
    DMA_WAIT_CH7,
)

if TYPE_CHECKING:
    from ..software.instruction import Uop


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
        exus: list[ExecutionUnit],
        logger: Logger,
        isa: type[IsaSpec],
        arch_state: ArchState,
    ) -> None:
        self.exus = exus
        self.logger = logger
        self.isa = isa
        self.arch_state = arch_state
        self.uop = None  # the current uop in flight
        self.lane_id = 1
        self.cycle = 0
        # Build a mapping from EXU type to EXU instances
        self.exu_map: dict[EXU, ExecutionUnit] = {}
        for exu in exus:
            exu_name = exu.__class__.__name__
            self.exu_map[EXU(exu_name)] = exu
        # The scalar core lives inside decode: we keep a reference to the
        # scalar EXU only to borrow its trace lane and stat counters.
        self.scalar_exu: ExecutionUnit | None = self.exu_map.get(EXU.SCALAR)
        self.reset()

    def reset(self) -> None:
        # Per-EXU output stage data
        self.outputs: dict[ExecutionUnit, StageData[Uop | None]] = {
            exu: StageData(None) for exu in self.exus
        }
        self._stalled = False
        self._control_flow_delay_slots_remaining = 0
        self._pending_scalar_completion: Uop | None = None
        self._complete_count = 0

    def is_finished(self) -> bool:
        """Check if DIU is finished."""
        return (
            self.uop is None
            and self._pending_scalar_completion is None
            and all(not output.is_valid() for output in self.outputs.values())
        )

    def tick(self, ifu_output: StageData[Uop | None]) -> None:
        """
        Dispatch fetched instructions to their target execution units.
        Logs: E (end F stage), S (start D stage)
        When stalled, ends D stage early to show gap.

        Scalar instructions are executed inline in this stage (the scalar
        core sits inside decode in the real hardware), so they retire one
        cycle earlier than instructions routed to a separate FU.
        """
        self.cycle += 1
        self._complete_count = 0

        # Retire any scalar instruction that executed inline last cycle.
        self._retire_pending_scalar()

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

            if self._is_control_flow_delay_slot_violation(uop):
                raise RuntimeError(
                    f"Illegal control-flow instruction '{uop.insn.mnemonic}' decoded "
                    f"in a delay-slot position on cycle {self.cycle}"
                )
            self._consume_delay_slot_if_needed()

            self.logger.log_stage_end(
                uop.id, "F", lane=LaneType.IFU.value, cycle=self.cycle
            )
            self.logger.log_stage_start(
                uop.id, "D", lane=LaneType.DIU.value, cycle=self.cycle
            )

            # Tag instruction with dispatch delay.
            if uop.dispatch_delay == 0 and isinstance(uop.insn, DELAY):
                uop.dispatch_delay = uop.insn.imm

            if self._is_control_flow_instruction(uop):
                self._control_flow_delay_slots_remaining = 1
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

    def force_unstall(self) -> None:
        self._stalled = False

    def dispatch(self) -> None:
        assert self.uop is not None
        assert self.uop.dispatch_delay == 0
        # Prepare empty outputs for this cycle

        if isinstance(self.uop.insn, _DMA_WAIT_TYPES):
            self.logger.log_stage_end(
                self.uop.id, "D", lane=LaneType.DIU.value, cycle=self.cycle + 1
            )
            self.uop = None
            return

        # Scalar instructions execute inline in the decode stage; everything
        # else is forwarded to its target FU through the sequencer queue.
        if self.uop.insn.exu == EXU.SCALAR:
            self._execute_scalar_inline()
            return

        target_exu = self.exu_map[self.uop.insn.exu]
        self.outputs[target_exu].prepare(self.uop)

        # if we dispatched a DMA instruction, set flag as busy here
        if self.uop.insn.exu == EXU.DMA and (
            is_scalar_itype(self.uop.insn) or isinstance(self.uop.insn, RType)
        ):
            assert not self.arch_state.check_flag(
                self.uop.insn.funct3
            ), f"Flag {self.uop.insn.funct3} is already set, erroneous program"
            self.arch_state.set_flag(self.uop.insn.funct3)
        self.uop = None

    def _execute_scalar_inline(self) -> None:
        """Run a scalar instruction inside the decode stage."""
        uop = self.uop
        assert uop is not None
        scalar_lane = (
            self.scalar_exu.lane_id if self.scalar_exu is not None else LaneType.DIU.value
        )

        # Open the execute lane in parallel with decode so the trace shows
        # that decode and scalar execute share the same cycle.
        self.logger.log_stage_start(
            uop.id, "E", lane=scalar_lane, cycle=self.cycle
        )

        uop.insn.exec(self.arch_state)

        if self.scalar_exu is not None:
            self.scalar_exu._total_instructions += 1
            if uop.insn.mnemonic != "delay":
                self.scalar_exu._busy_cycles += 1

        self._pending_scalar_completion = uop
        self.uop = None

    def _retire_pending_scalar(self) -> None:
        """Close the D and E stages opened last cycle and retire the uop."""
        uop = self._pending_scalar_completion
        if uop is None:
            return
        scalar_lane = (
            self.scalar_exu.lane_id if self.scalar_exu is not None else LaneType.DIU.value
        )
        self.logger.log_stage_end(
            uop.id, "D", lane=LaneType.DIU.value, cycle=self.cycle
        )
        self.logger.log_stage_end(
            uop.id, "E", lane=scalar_lane, cycle=self.cycle
        )
        self.logger.log_retire(uop.id)
        self._complete_count += 1
        self._pending_scalar_completion = None

    @property
    def complete_count(self) -> int:
        """Scalar instructions retired this cycle (executed inside decode)."""
        return self._complete_count

    def claim_uop(self, ifu_output: StageData[Uop | None]) -> None:
        """Claim a new uop from IFU"""
        assert self.uop is None

    def _is_control_flow_instruction(self, uop: Uop) -> bool:
        return isinstance(uop.insn, (SBType, UJType)) or uop.insn.mnemonic == "jalr"

    def _is_control_flow_delay_slot_violation(self, uop: Uop) -> bool:
        return (
            self._control_flow_delay_slots_remaining > 0
            and self._is_control_flow_instruction(uop)
        )

    def _consume_delay_slot_if_needed(self) -> None:
        if self._control_flow_delay_slots_remaining > 0:
            self._control_flow_delay_slots_remaining -= 1

    def check_backpressure(self, uop: Uop) -> bool:
        if isinstance(uop.insn, _DMA_WAIT_TYPES):
            if self.arch_state.check_flag(uop.insn.funct3):
                self._stalled = True
                return True
            else:
                self._stalled = False
                return False
        if uop.insn.exu == EXU.SCALAR:
            # Scalar instructions execute in-place in decode; no FU queue to
            # back up against.
            self._stalled = False
            return False
        target_exu = self.exu_map[uop.insn.exu]
        if self.outputs[target_exu].should_stall():
            # Don't end D stage - keep it active to show instruction is waiting
            # The D stage will end when we actually dispatch
            raise RuntimeError(
                f"Backpressure detected in IDU when running uop {uop.id} {str(uop.insn)} on cycle {self.cycle}"
            )

        # Backpressure cleared - if we were stalled, the D stage continues:
        self._stalled = False
        return False
