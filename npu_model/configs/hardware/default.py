from npu_model.hardware.config import HardwareConfig, ArchStateConfig
from npu_model.isa import IsaSpec


class DefaultHardwareConfig(HardwareConfig):
    name: str = "SimpleNPU"

    fetch_width: int = 1
    isa: type[IsaSpec] = IsaSpec
    arch_state_config: ArchStateConfig = ArchStateConfig(
        mrf_depth=32,  # 32 rows per architectural tensor tile
        mrf_width=32,  # 32 bytes per row => 32x32 FP8 or 32x16 BF16 per register
        wb_width=32 * 32 * 1,  # one 32x32 FP8 weight tile per weight slot
        num_x_registers=32,
        num_csrs=4096,  # cover the architectural 12-bit CSR index space
        num_e_registers=32,
        num_m_registers=64,
        num_wb_registers=2,
        dram_base=0x80000000,
        dram_size=1 * 1024 * 1024 * 1024,  # 1 GiB default simulation aperture
        vmem_base=0x20000000,
        vmem_size=1024 * 1024,
    )
    mxu0_matmul_latency_cycles: int = 32
    mxu1_matmul_latency_cycles: int = 32
    vpu_simple_op_latency_cycles: int = 4
    vpu_non_pipelineable_op_latency_cycles: int = 16
    xlu_transform_latency_cycles: int = 4
    offchip_link_width_bits: int = 32
    offchip_link_core_cycles_per_beat: int = 2
    dma_offchip_command_words: int = 2
    vmem_bus_width_bits: int = 512
    vmem_bus_core_cycles_per_beat: int = 1
    vmem_bytes_per_cycle: int = 64
    execution_units: dict[str, str] = {
        "Scalar0": "ScalarExecutionUnit",
        "Matrix0": "MatrixExecutionUnitSystolic",
        "Matrix1": "MatrixExecutionUnitInner",
        "Vector0": "VectorExecutionUnit",
        "DMA0": "DmaExecutionUnit",
        "LSU": "LoadStoreUnit"
    }


class FullDramHardwareConfig(DefaultHardwareConfig):
    name: str = "SimpleNPUFullDram"
    arch_state_config: ArchStateConfig = ArchStateConfig(
        mrf_depth=32,
        mrf_width=32,
        wb_width=32 * 32 * 1,
        num_x_registers=32,
        num_csrs=4096,
        num_e_registers=32,
        num_m_registers=64,
        num_wb_registers=2,
        dram_base=0x80000000,
        dram_size=16 * 1024 * 1024 * 1024,
        vmem_base=0x20000000,
        vmem_size=1024 * 1024,
    )
