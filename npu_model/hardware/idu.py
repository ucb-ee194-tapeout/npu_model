from __future__ import annotations

from .hardware import Module
from .stage_data import StageData
from .exu import ExecutionUnit
from ..logging.logger import Logger, LaneType
from ..isa import IsaSpec, RType, SBType, UJType
from ..isa_types import EXU
from .arch_state import ArchState
from ..software.instruction import Uop


class InstructionDecode(Module):
    """RTL S1: decode, register read, execute and launch in the same cycle.

    Only DELAY and DMA.WAIT hold the frontend. Resource conflicts are software
    scheduling errors, not extra pipeline stages or implicit stalls.
    """

    def __init__(self, exus: list[ExecutionUnit], logger: Logger,
                 isa: type[IsaSpec], arch_state: ArchState) -> None:
        self.exus = exus
        self.logger = logger
        self.isa = isa
        self.arch_state = arch_state
        self.lane_id = LaneType.DIU.value
        self.exu_map = {EXU(type(exu).__name__): exu for exu in exus}
        self.reset()

    def reset(self) -> None:
        self.outputs = {exu: StageData(None) for exu in self.exus}
        self.uop: Uop | None = None
        self.issued_uop: Uop | None = None
        self.cycle = 0
        self.delay_counter = 0
        self._stalled = False
        self.stall_reason: str | None = None

    def is_finished(self) -> bool:
        return self.uop is None and self.delay_counter == 0 and all(
            not output.is_valid() for output in self.outputs.values()
        )

    @staticmethod
    def _is_control_flow_instruction(uop: Uop) -> bool:
        return isinstance(uop.insn, (SBType, UJType)) or uop.insn.mnemonic == "jalr"

    def tick(self, ifu_output: StageData[Uop | None]) -> None:
        self.cycle += 1
        self.issued_uop = None
        self._stalled = False
        self.stall_reason = None
        if self.arch_state.halted:
            self.uop = None
            self.delay_counter = 0
            return

        if self.uop is None:
            self.uop = ifu_output.claim()
            if self.uop is not None:
                self.logger.log_stage_end(self.uop.id, "F", lane=LaneType.IFU.value,
                                          cycle=self.cycle)
                self.logger.log_stage_start(self.uop.id, "D", lane=self.lane_id,
                                            cycle=self.cycle)

        # ScalarCore halt detection is combinational and precedes stall gating.
        if self.uop is not None:
            mnemonic = self.uop.insn.mnemonic
            if (self.arch_state.in_delay_slot
                    and self._is_control_flow_instruction(self.uop)):
                self.arch_state.halted = True
                self.arch_state.halt_reason = "illegal"
                self.arch_state.execute_pc = self.uop.pc
                raise RuntimeError(
                    f"Illegal control-flow instruction '{mnemonic}' decoded "
                    f"in a delay-slot position on cycle {self.cycle}"
                )
            if mnemonic in {"ecall", "ebreak"}:
                self.arch_state.halted = True
                self.arch_state.halt_reason = mnemonic
                self.arch_state.execute_pc = self.uop.pc
                self.logger.log_stage_end(self.uop.id, "D", lane=self.lane_id,
                                          cycle=self.cycle)
                self.uop = None
                self.delay_counter = 0
                return

        if self.delay_counter:
            self.delay_counter -= 1
            self._stalled = True
            self.stall_reason = "delay"
            return
        if self.uop is None:
            return
        insn = self.uop.insn
        if insn.mnemonic.startswith("dma.wait") and self.arch_state.check_flag(insn.funct3):
            self._stalled = True
            self.stall_reason = "dma.wait"
            return

        self.arch_state.execute_pc = self.uop.pc
        self.arch_state.in_delay_slot = False
        self.issued_uop = self.uop
        if insn.mnemonic.startswith("dma.wait"):
            self.logger.log_stage_end(self.uop.id, "D", lane=self.lane_id,
                                      cycle=self.cycle)
            self.logger.log_retire(self.uop.id)
        else:
            target_exu = self.exu_map[insn.exu]
            if self.outputs[target_exu].should_stall():
                raise RuntimeError(
                    f"Backpressure detected in IDU when running uop {self.uop.id} "
                    f"{insn} on cycle {self.cycle}"
                )
            self.outputs[target_exu].prepare(self.uop)
            if insn.exu == EXU.DMA and isinstance(insn, RType):
                assert not self.arch_state.check_flag(insn.funct3), (
                    f"Flag {insn.funct3} is already set, erroneous program"
                )
                self.arch_state.set_flag(insn.funct3)
        if insn.mnemonic == "delay":
            self.delay_counter = int(insn.imm) & 0xFFF
        self.uop = None

    @property
    def is_stalled(self) -> bool:
        return self._stalled

    def force_unstall(self) -> None:
        self._stalled = False
        self.stall_reason = None
