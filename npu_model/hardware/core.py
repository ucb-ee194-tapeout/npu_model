from typing import Callable, List
import torch

from npu_model.software.program import Program
from npu_model.software.instruction import Uop
from npu_model.logging.logger import Logger, LaneType
from npu_model.hardware.arch_state import ArchState
from npu_model.hardware.stage_data import StageData

from .hardware import Module
from .config import HardwareConfig
from .ifu import InstructionFetch
from .idu import InstructionDecode
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
    Orchestrates the two RTL stages: fetch, then decode/execute/writeback.
    Decode dispatches combinationally to the EXUs before the next fetch edge.

    Pipeline stages use StageData with claim-based handshaking:
    - Downstream stages claim data from upstream stages
    - Upstream stages stall if their data isn't claimed

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
        self.idu = InstructionDecode(
            exus=self.exus,
            logger=self.logger,
            arch_state=self.arch_state,
            isa=self.config.isa,
        )

        self.ignore_runtime_errors = False
        self.runtime_error_reporter: Callable[[str, Exception], None] | None = None

        self.reset()

    def load_program(self, program: Program):
        self.ifu.load_program(program)
        if len(program.memory_regions) > 0:
            for base, arr in program.memory_regions:
                self.arch_state.write_dram(base, arr.flatten().view(torch.uint8))

    def reset(self) -> None:
        """Reset all components."""
        self.arch_state.reset()
        self.ifu.reset()
        self.idu.reset()
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
        s1 = self.idu.uop or self.ifu.output.peek()
        state.npc = (state.pc + 1) & 0xFFFFFFFF
        state.redirect_requested = False
        state.current_uop = None

        try:
            self.idu.tick(self.ifu.output)
        except Exception as exc:
            if not self._handle_runtime_error("IDU", exc):
                raise
            self._recover_idu_fault()
        state.current_uop = self.idu.issued_uop

        # Read operands / launch commands before scalar-load writeback.
        for exu in sorted(self.exus, key=lambda unit: isinstance(unit, LoadStoreUnit)):
            idu_out = self.idu.outputs[exu]
            try:
                exu.tick(idu_output=idu_out)
            except Exception as exc:
                if not self._handle_runtime_error(f"EXU {exu.name}", exc):
                    raise
                self._recover_exu_fault(exu, idu_out)

        if self.idu.issued_uop is not None:
            # RTL inst_retire counts S1 launches, not asynchronous completions.
            self.total_completed += 1
        if state.redirect_requested:
            state.in_delay_slot = True
        state.finish_csr_cycle(retired=self.idu.issued_uop is not None)

        try:
            self.ifu.tick(stalled=self.idu.is_stalled)
        except Exception as exc:
            if not self._handle_runtime_error("IFU", exc):
                raise
            self._recover_ifu_fault()

        self.last_cycle = {
            "cycle": self.cycle_count,
            "fetch_pc": fetch_pc,
            "s1_pc": s1.pc if s1 is not None else None,
            "s1_valid": s1 is not None,
            "s1_fire": self.idu.issued_uop is not None,
            "instruction": s1.insn.mnemonic if s1 is not None else None,
            "stall": self.idu.stall_reason,
            "redirect": state.redirect_requested,
            "next_pc": state.pc,
            "halted": state.halted,
        }

    def is_finished(self) -> bool:
        """Check if execution is complete."""
        if self.arch_state.halted:
            return True
        if not self.ifu.is_finished():
            return False
        if not self.idu.is_finished():
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

    def _recover_exu_fault(self, exu: ExecutionUnit, idu_out: StageData[Uop | None]) -> None:
        idu_out.reset()
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

    def _recover_idu_fault(self) -> None:
        self.idu.uop = None
        self.idu.issued_uop = None
        self.idu.force_unstall()
        for output in self.idu.outputs.values():
            output.reset()

    def _recover_ifu_fault(self) -> None:
        self.ifu.force_unstall()
