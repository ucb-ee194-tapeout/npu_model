from typing import List
import torch

from model_npu.software.program import Program
from model_npu.logging.logger import Logger, LaneType
from model_npu.hardware.arch_state import ArchState

from .hardware import Module
from .config import HardwareConfig
from .ifu import InstructionFetch
from .idu import InstructionDecode
from .exu import ExecutionUnit

from .exu import ScalarExecutionUnit  # noqa: F401, F403
from .mxu import MatrixExecutionUnit  # noqa: F401, F403
from .dma import DmaExecutionUnit  # noqa: F401, F403
from .vpu import VectorExecutionUnit  # noqa: F401, F403


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

        self.reset()

    def load_program(self, program: Program):
        self.ifu.load_program(program)
        if len(program.memory_regions) > 0:
            for base, arr in program.memory_regions:
                self.arch_state.write_memory(base, arr.flatten().view(torch.uint8))

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
            idu_output = self.idu.outputs[exu]
            exu.tick(idu_output)
            self.total_completed += exu.complete_count

        # 3. Tick IDU (claim from InstructionFetch output, dispatch to EXU outputs)
        self.idu.tick(self.ifu.output)

        # 4. Tick IFU (fetch new instructions if not stalled)
        self.ifu.tick()

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
