from typing import Callable, List
import torch

from npu_model.software.program import Program
from npu_model.software.instruction import Uop
from npu_model.logging.logger import Logger, LaneType
from npu_model.hardware.arch_state import ArchState
from npu_model.isa import RType, SBType, UJType
from npu_model.isa_types import EXU

from .hardware import Module
from .config import HardwareConfig
from .ifu import InstructionFetch
from .exu import ExecutionUnit

from .exu import ScalarExecutionUnit  # type: ignore # noqa: F401, F403
from .mxu import (
    MatrixExecutionUnitInner, # type: ignore 
    MatrixExecutionUnitSystolic, # type: ignore 
)  # noqa: F401, F403
from .dma import DmaExecutionUnit  # type: ignore # noqa: F401, F403
from .vpu import VectorExecutionUnit  # type: ignore # noqa: F401, F403
from .xlu import CrossLaneExecutionUnit  # noqa: F401
from .lsu import LoadStoreUnit  # type: ignore # noqa: F401, F403 


class Core(Module):
    """
    NPU Core.
    Orchestrates the two RTL stages: fetch (IFU), then S1, which decodes,
    reads registers, executes scalar ops and launches EXU commands in one
    cycle. Only DELAY and DMA.WAIT hold S1. Resource conflicts are software
    scheduling errors, not extra pipeline stages or implicit stalls.

    Each functional unit handles its own logging.
    """

    def __init__(
        self,
        config: HardwareConfig,
        logger: Logger,
    ) -> None:
        self.config = config
        self.logger = logger

        self.arch_state = ArchState(
            config=self.config.arch_state_config,
            logger=self.logger,
        )

        # Create execution units (each gets logger reference)
        self.exus: List[ExecutionUnit] = []

        for idx, (name, exu_class) in enumerate(self.config.execution_units.items()):
            self.exus.append(
                eval(exu_class)(
                    name,
                    logger=self.logger,
                    arch_state=self.arch_state,
                    lane_id=LaneType.EXU_BASE.value + idx,
                    config=self.config,
                )
            )

        # Create pipeline components (each gets logger reference)
        self.ifu = InstructionFetch(
            width=self.config.fetch_width,
            logger=self.logger,
            arch_state=self.arch_state,
        )
        self.exu_map = {EXU(type(exu).__name__): exu for exu in self.exus}

        self.ignore_runtime_errors = False
        self.runtime_error_reporter: Callable[[str, Exception], None] | None = None

        self.reset()

    def load_program(self, program: Program):
        self.ifu.load_program(program)
        if len(program.memory_regions) > 0:
            for base, arr in program.memory_regions:
                self.arch_state.write_dram(program.dram_base + base, arr.flatten().view(torch.uint8))

    def reset(self) -> None:
        """Reset all components."""
        self.arch_state.reset()
        self.ifu.reset()
        self._reset_s1()
        for exu in self.exus:
            exu.reset()
        self.cycle_count = 0
        self.last_cycle = {}
        self.total_completed = 0

    def tick(self) -> None:
        """Advance one edge, evaluating S1 against the previous edge's state.

        Fetch reads the old PC even on a redirect, preserving exactly one
        architectural delay slot. LSU writeback is last so register reads in
        this cycle observe the old value, as in the RTL (no load bypass).
        """
        self.logger.log_cycle(1)
        self.cycle_count += 1
        state = self.arch_state
        state.conflict_checker.begin_cycle(self.cycle_count)
        state.begin_csr_cycle()
        fetch_pc = state.pc
        s1 = self.s1_uop or self.ifu.output.peek()
        state.npc = (state.pc + 1) & 0xFFFFFFFF
        state.redirect_requested = False
        state.current_uop = None

        target: ExecutionUnit | None = None
        try:
            target = self._tick_s1()
        except Exception as exc:
            if not self._handle_runtime_error("S1", exc):
                raise
            self._recover_s1_fault()
        state.current_uop = self.issued_uop

        # Read operands / launch commands before scalar-load writeback.
        for exu in sorted(self.exus, key=lambda unit: isinstance(unit, LoadStoreUnit)):
            try:
                exu.tick(self.issued_uop if exu is target else None)
            except Exception as exc:
                if not self._handle_runtime_error(f"EXU {exu.name}", exc):
                    raise
                self._recover_exu_fault(exu)

        if self.issued_uop is not None:
            # RTL inst_retire counts S1 launches, not asynchronous completions.
            self.total_completed += 1
        if state.redirect_requested:
            state.in_delay_slot = True
        state.finish_csr_cycle(retired=self.issued_uop is not None)

        try:
            self.ifu.tick(stalled=self._s1_stalled)
        except Exception as exc:
            if not self._handle_runtime_error("IFU", exc):
                raise
            self._recover_ifu_fault()

        self.last_cycle = {
            "cycle": self.cycle_count,
            "fetch_pc": fetch_pc,
            "s1_pc": s1.pc if s1 is not None else None,
            "s1_valid": s1 is not None,
            "s1_fire": self.issued_uop is not None,
            "instruction": s1.insn.mnemonic if s1 is not None else None,
            "stall": self.stall_reason,
            "redirect": state.redirect_requested,
            "next_pc": state.pc,
            "halted": state.halted,
        }

    def _reset_s1(self) -> None:
        # s1_uop models ScalarCore's instruction hold register.
        self.s1_uop: Uop | None = None
        self.issued_uop: Uop | None = None
        self.delay_counter = 0
        self._s1_stalled = False
        self.stall_reason: str | None = None

    @staticmethod
    def _is_control_flow_instruction(uop: Uop) -> bool:
        return isinstance(uop.insn, (SBType, UJType)) or uop.insn.mnemonic == "jalr"

    def _tick_s1(self) -> ExecutionUnit | None:
        """Evaluate S1: halt detection, stall gating, then launch.

        Returns the EXU that receives the launched uop this cycle, if any.
        """
        state = self.arch_state
        self.issued_uop = None
        self._s1_stalled = False
        self.stall_reason = None
        if state.halted:
            self.s1_uop = None
            self.delay_counter = 0
            return None

        if self.s1_uop is None:
            self.s1_uop = self.ifu.output.claim()
            if self.s1_uop is not None:
                self.logger.log_stage_end(self.s1_uop.id, "F", lane=LaneType.IFU.value,
                                          cycle=self.cycle_count)

        # ScalarCore halt detection is combinational and precedes stall gating.
        if self.s1_uop is not None:
            mnemonic = self.s1_uop.insn.mnemonic
            if state.in_delay_slot and self._is_control_flow_instruction(self.s1_uop):
                state.halted = True
                state.halt_reason = "illegal"
                state.execute_pc = self.s1_uop.pc
                raise RuntimeError(
                    f"Illegal control-flow instruction '{mnemonic}' decoded "
                    f"in a delay-slot position on cycle {self.cycle_count}"
                )
            if mnemonic in {"ecall", "ebreak"}:
                state.execute_pc = self.s1_uop.pc
                self.s1_uop.insn.exec(state)
                self.s1_uop = None
                self.delay_counter = 0
                return None

        if self.delay_counter:
            self.delay_counter -= 1
            self._s1_stalled = True
            self.stall_reason = "delay"
            return None
        if self.s1_uop is None:
            return None
        insn = self.s1_uop.insn
        if insn.mnemonic.startswith("dma.wait") and state.check_flag(insn.funct3):
            self._s1_stalled = True
            self.stall_reason = "dma.wait"
            return None

        state.execute_pc = self.s1_uop.pc
        state.in_delay_slot = False
        self.issued_uop = self.s1_uop
        self.s1_uop = None
        target: ExecutionUnit | None = None
        if insn.mnemonic.startswith("dma.wait"):
            insn.exec(state)
            self.logger.log_retire(self.issued_uop.id)
        else:
            target = self.exu_map[insn.exu]
            if insn.exu == EXU.DMA and isinstance(insn, RType) and not insn.mnemonic.startswith("dma.config"):
                # Only transfers occupy a channel; dma.config writes dmaBaseReg at issue.
                assert not state.check_flag(insn.funct3), (
                    f"Flag {insn.funct3} is already set, erroneous program"
                )
                state.set_flag(insn.funct3)
        if insn.mnemonic == "delay":
            self.delay_counter = int(insn.imm) & 0xFFF
        return target

    def is_finished(self) -> bool:
        """Check if execution is complete."""
        if self.arch_state.halted:
            return True
        if not self.ifu.is_finished():
            return False
        if self.s1_uop is not None or self.delay_counter:
            return False
        for exu in self.exus:
            if exu.has_in_flight:
                return False
        return True

    def stop(self):
        # Flush any pending completions in EXUs
        for exu in self.exus:
            exu.flush_completions()

    def close(self) -> None:
        self.arch_state.close()
        self.exus.clear()

    def _handle_runtime_error(self, stage: str, exc: Exception) -> bool:
        if not self.ignore_runtime_errors:
            return False
        if self.runtime_error_reporter is not None:
            self.runtime_error_reporter(stage, exc)
        return True

    def _recover_exu_fault(self, exu: ExecutionUnit) -> None:
        if hasattr(exu, "abort"):
            exu.abort()
            return
        if hasattr(exu, "in_flight"):
            current = getattr(exu, "in_flight")
            if isinstance(current, list):
                setattr(exu, "in_flight", [])
            else:
                setattr(exu, "in_flight", None)
        if hasattr(exu, "_pending_completions"):
            getattr(exu, "_pending_completions").clear()
        if hasattr(exu, "_pending_completion_uop"):
            setattr(exu, "_pending_completion_uop", None)
        if hasattr(exu, "_complete_count"):
            setattr(exu, "_complete_count", 0)

    def _recover_s1_fault(self) -> None:
        self.s1_uop = None
        self.issued_uop = None
        self._s1_stalled = False
        self.stall_reason = None

    def _recover_ifu_fault(self) -> None:
        self.ifu.force_unstall()
