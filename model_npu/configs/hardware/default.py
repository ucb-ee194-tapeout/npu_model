from typing import Tuple
from model_npu.hardware.config import HardwareConfig
from model_npu.isa import IsaSpec


class DefaultHardwareConfig(HardwareConfig):
    name: str = "SimpleNPU"

    fetch_width: int = 1
    isa: IsaSpec = IsaSpec
    matrix_shape: Tuple[int, int] = (16, 16)
    memory_size: int = 16 * 16 * 4 * 2
    execution_units: dict[str, str] = {
        "Scalar0": "ScalarExecutionUnit",
        "Matrix0": "MatrixExecutionUnit",
        "DMA0": "DmaExecutionUnit",
    }
