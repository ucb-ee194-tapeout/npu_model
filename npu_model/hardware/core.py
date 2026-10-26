from typing import Callable, List, NoReturn
import torch

from npu_model.software.program import Program
from npu_model.software.instruction import Uop
from npu_model.logging.logger import Logger, LaneType
from npu_model.hardware.arch_state import ArchState
from npu_model.isa import TABLE, CSRType, Instruction, RType, Word
from npu_model.isa_types import EXU

from .hardware import Module
from .config import HardwareConfig
from .ifu import InstructionFetch
from .exu import ExecutionUnit

from .exu import ScalarExecutionUnit  # type: ignore # noqa: F401, F403
from .mxu import MatrixExecutionUnitInner  # type: ignore # noqa: F401, F403
from .dma import DmaExecutionUnit  # type: ignore # noqa: F401, F403
from .vpu import _TWO_INPUT, VectorExecutionUnit  # type: ignore # noqa: F401, F403
from .xlu import CrossLaneExecutionUnit  # noqa: F401
from .lsu import LoadStoreUnit  # type: ignore # noqa: F401, F403 


def _x_regs(insn: Instruction) -> dict[str, int]:
    """The fields of an instruction that hold an x-register (the table's x operands, which are
    ScalarCore's decode column xr), with their registers."""
    operands = TABLE.entries[insn.mnemonic].operands
    return {field: int(getattr(insn, field)) for field in ("rd", "rs1", "rs2") if operands.get(field) == "x"}


def _operand_illegal(insn: Instruction) -> str | None:
    """ScalarCore's operand rules that the encoding does not give: a two-input .bf16 VPU operation
    with two different source pairs in one physical bank (m_i and m_i+32), and vpack.bf16.fp8 with
    vd = vs2 + 1 (it writes m[vd] before it reads m[vs2+1])."""
    if insn.mnemonic in _TWO_INPUT and int(insn.vs1) % 32 == int(insn.vs2) % 32 and insn.vs1 != insn.vs2:
        return "two source pairs in one physical bank"
    if insn.mnemonic == "vpack.bf16.fp8" and int(insn.vd) == int(insn.vs2) + 1:
        return "vd = vs2 + 1"
    return None


class Core(Module):
    """
    NPU Core.
    Orchestrates the two RTL stages: fetch (IFU), then S1, which decodes,
    reads registers, executes scalar ops and launches EXU commands in one
    cycle. DELAY, DMA.WAIT and the scalar-load interlock hold S1. Other
    resource conflicts are software scheduling errors, not extra pipeline
    stages or implicit stalls.

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

        A taken branch or jump drops the instruction that fetch reads in its
        cycle (no delay slot). LSU writeback is last so register reads in this
        cycle observe the old value, as in the RTL (no load bypass).
        """
        self.logger.log_cycle(1)
        self.cycle_count += 1
        state = self.arch_state
        state.conflict_checker.begin_cycle(self.cycle_count)
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
            insn = self.s1_uop.insn
            mnemonic = insn.mnemonic
            if isinstance(insn, Word):
                self._illegal(f"Illegal instruction word {insn.word:#010x}")
            if isinstance(insn, CSRType) and state.csr_illegal(insn.imm, insn.writes):
                self._illegal(f"Illegal CSR access '{mnemonic}' to CSR {insn.imm:#x}")
            if (rule := _operand_illegal(insn)) is not None:
                self._illegal(f"Illegal operands of '{mnemonic}': {rule}")
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

        if self._x_reg_hazard(insn):
            self._s1_stalled = True
            self.stall_reason = "xreg"
            return None

        state.execute_pc = self.s1_uop.pc
        # A jump target with bit 1 set is illegal (RV32I without C), and so is a DMA transfer size
        # that the DMA engine does not take; the RTL checks both at issue.
        jump = insn.jump_target(state)
        if jump is not None and jump & 2:
            self._illegal(f"Illegal jump target {jump:#x} of '{insn.mnemonic}'")
        if insn.mnemonic in ("dma.load", "dma.store"):
            nbytes = state.read_xrf(insn.rs2)
            if self.exu_map[EXU.DMA].illegal_size(nbytes):
                self._illegal(f"Illegal DMA transfer size {nbytes} of '{insn.mnemonic}'")
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

    def _x_reg_hazard(self, insn: Instruction) -> bool:
        """ScalarCore's xRegHazard: an x-register that the instruction names has a pending scalar
        load, or a load response takes the register write port while the instruction writes x[rd]."""
        regs = _x_regs(insn)
        named = set(regs.values()) - {0}
        writes_rd = insn.exu == EXU.SCALAR and regs.get("rd", 0) != 0
        lsu = self.exu_map[EXU.LSU]
        return any(_x_regs(load).get("rd") in named or (writes_rd and now) for load, now in lsu.pending_loads())

    def _illegal(self, message: str) -> NoReturn:
        """Halt on the illegal instruction in S1, as ScalarCore does."""
        assert self.s1_uop is not None
        state = self.arch_state
        state.halted = True
        state.halt_reason = "illegal"
        state.execute_pc = self.s1_uop.pc
        raise RuntimeError(f"{message} on cycle {self.cycle_count}")

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
