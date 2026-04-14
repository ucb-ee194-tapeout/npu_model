from typing import Callable, List
import torch

from npu_model.software.program import Program
from npu_model.logging.logger import Logger, LaneType
from npu_model.hardware.arch_state import ArchState

from .hardware import Module
from .config import HardwareConfig
from .ifu import InstructionFetch
from .idu import InstructionDecode
from .exu import ExecutionUnit

from .exu import ScalarExecutionUnit  # noqa: F401, F403
from .mxu import (
    MatrixExecutionUnitInner,
    MatrixExecutionUnitSystolic,
)  # noqa: F401, F403
from .dma import DmaExecutionUnit  # noqa: F401, F403
from .vpu import VectorExecutionUnit  # noqa: F401, F403
from .lsu import LoadStoreUnit


class Core(Module):
    """
    NPU Core.
    Orchestrates the pipeline: IFU -> DIU -> EXUs.
    Ticking happens in reverse pipeline order to properly propagate values.

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
        # self.cycle_count = 0
        self.total_completed = 0

    def tick(self) -> None:
        """
        Execute one cycle.
        Tick in reverse pipeline order (downstream first):
        2. EXUs claim and consume from DIU outputs
        1. IDU claims from IFU and dispatches to EXU outputs
        3. IFU fetches new instructions (if not stalled)
        4. Log cycle advancement

        Each downstream stage claims from the previous stage's output.
        If a stage's output isn't claimed, it will stall on the next tick.
        """
        # 0. Log cycle advancement
        self.logger.log_cycle(1)

        # 1. Advance program counter
        self.arch_state.npc = self.arch_state.pc + 1

        # 2. Tick EXUs (claim from InstructionDecode outputs)
        for exu in self.exus:
            idu_out = self.idu.outputs[exu]
            try:
                exu.tick(idu_output=idu_out)
            except Exception as exc:
                if not self._handle_runtime_error(f"EXU {exu.name}", exc):
                    raise
                self._recover_exu_fault(exu, idu_out)
            else:
                self.total_completed += exu.complete_count

        # 3. Tick IDU (claim from InstructionFetch output, dispatch to EXU outputs)
        try:
            self.idu.tick(self.ifu.output)
        except Exception as exc:
            if not self._handle_runtime_error("IDU", exc):
                raise
            self._recover_idu_fault()

        # 4. Tick IFU (fetch new instructions if not stalled)
        try:
            self.ifu.tick()
        except Exception as exc:
            if not self._handle_runtime_error("IFU", exc):
                raise
            self._recover_ifu_fault()

    def is_finished(self) -> bool:
        """Check if execution is complete."""
        if not self.ifu.is_finished():
            return False
        if not self.idu.is_finished():
            return False
        for exu in self.exus:
            if exu.has_in_flight:
                print(f"EXU {exu.name} has in-flight instructions")
                return False
        return True

    def stop(self):
        # Flush any pending completions in EXUs
        for exu in self.exus:
            exu.flush_completions()

    def _handle_runtime_error(self, stage: str, exc: Exception) -> bool:
        if not self.ignore_runtime_errors:
            return False
        if self.runtime_error_reporter is not None:
            self.runtime_error_reporter(stage, exc)
        return True

    def _recover_exu_fault(self, exu: ExecutionUnit, idu_out) -> None:
        idu_out.reset()
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
        self.idu._stalled = False
        for output in self.idu.outputs.values():
            output.reset()

    def _recover_ifu_fault(self) -> None:
        self.ifu._stalled = False
