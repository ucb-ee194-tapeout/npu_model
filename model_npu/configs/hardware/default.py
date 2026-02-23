from typing import Tuple
from model_npu.hardware.config import HardwareConfig, ArchStateConfig
from model_npu.isa import IsaSpec


class DefaultHardwareConfig(HardwareConfig):
    name: str = "SimpleNPU"

    fetch_width: int = 1
    isa: IsaSpec = IsaSpec
    arch_state_config: ArchStateConfig = ArchStateConfig(
        mrf_depth=64,  # each instruction is 64 cycles
        mrf_width=1 * 32 * 1,  # 64 rows of 32 bytes in a tensor register
        wb_width=32 * 16 * 1,  # 512 16-bit elements fit in one weight buffer
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
