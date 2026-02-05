
from typing import Tuple
from model_npu.isa import IsaSpec


class HardwareConfig:
    name: str
    fetch_width: int
    isa: IsaSpec = IsaSpec
    matrix_shape: Tuple[int, int]
    execution_units: dict[str, str]
    memory_size: int
