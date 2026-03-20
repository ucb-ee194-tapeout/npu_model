from typing import Tuple
from npu_model.hardware.config import HardwareConfig, ArchStateConfig
from npu_model.isa import IsaSpec


class DefaultHardwareConfig(HardwareConfig):
    name: str = "SimpleNPU"

    fetch_width: int = 1
    isa: IsaSpec = IsaSpec
    arch_state_config: ArchStateConfig = ArchStateConfig(
        mrf_depth=32,  # 32 rows per architectural tensor tile
        mrf_width=32,  # 32 bytes per row => 32x32 FP8 or 32x16 BF16 per register
        wb_width=32 * 32 * 1,  # one 32x32 FP8 weight tile per weight slot
        num_x_registers=32,
        num_m_registers=64,
        num_wb_registers=2,
        memory_size=1048576,
    )
    mxu0_matmul_latency_cycles: int = 32
    mxu1_matmul_latency_cycles: int = 32
    vpu_simple_op_latency_cycles: int = 2
    vpu_non_pipelineable_op_latency_cycles: int = 8
    xlu_transform_latency_cycles: int = 4
    vmem_bytes_per_cycle: int = 64
    execution_units: dict[str, str] = {
        "Scalar0": "ScalarExecutionUnit",
        "Matrix0": "MatrixExecutionUnitSystolic",
        "Matrix1": "MatrixExecutionUnitInner",
        "Vector0": "VectorExecutionUnit",
        "DMA0": "DmaExecutionUnit",
    }
