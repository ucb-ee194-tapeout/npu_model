from typing import Tuple
from model_npu.hardware.config import HardwareConfig, ArchStateConfig
from model_npu.isa import IsaSpec


class DefaultHardwareConfig(HardwareConfig):
    name: str = "SimpleNPU"

    fetch_width: int = 1
    isa: IsaSpec = IsaSpec
    arch_state_config: ArchStateConfig = ArchStateConfig(
        mrf_depth=64,  # each instruction is 64 cycles
        mrf_width=1*32*2,  # each cycle, we read out 64 B of activation row
        wb_width=16*32*2,  # each cycle, we read out 1024 B of weight
        num_x_registers=32,
        num_m_registers=64,
        num_wb_registers=2,
        memory_size=1048576,
    )
    execution_units: dict[str, str] = {
        "Scalar0": "ScalarExecutionUnit",
        "Matrix0": "MatrixExecutionUnit",
        "Vector0": "VectorExecutionUnit",
        "DMA0": "DmaExecutionUnit",
    }
