from typing import Tuple
from model_npu.hardware.config import HardwareConfig
from model_npu.isa import IsaSpec


class DefaultHardwareConfig(HardwareConfig):
    name: str = "SimpleNPU"

    fetch_width: int = 1
    isa: IsaSpec = IsaSpec
    matrix_shape: Tuple[int, int] = (64, 32)
    weight_shape: Tuple[int, int] = (16, 32)
    memory_size: int = 1048576
    execution_units: dict[str, str] = {
        "Scalar0": "ScalarExecutionUnit",
        "Matrix0": "MatrixExecutionUnit",
        "Vector0": "VectorExecutionUnit",
        "DMA0": "DmaExecutionUnit",
    }
