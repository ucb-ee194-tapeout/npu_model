from dataclasses import dataclass

from npu_model.isa import IsaSpec


@dataclass
class ArchStateConfig:
    mrf_depth: int
    """ Read depth of a matrix register (number of rows). """
    mrf_width: int
    """ Read width of a matrix register in bytes. """
    wb_width: int
    """ Read width of a weight buffer entry in bytes. """
    num_x_registers: int
    """ Number of scalar registers. """
    num_csrs: int
    """ Number of control and status registers. """
    num_e_registers: int
    """Number of scaling factor registers."""
    num_m_registers: int
    """ Number of matrix registers. """
    num_wb_registers: int
    """ Number of weight buffer entries. """
    dram_size: int
    """ Size of dram in bytes. """
    vmem_size: int
    """ Size of vmem in bytes. """
    randomize_init: bool = False
    """ Initialize architectural storage with deterministic pseudo-random data. """
    init_seed: int = 42
    """ Seed used when randomize_init is enabled. """


class HardwareConfig:
    name: str
    fetch_width: int
    isa: type[IsaSpec] = IsaSpec
    arch_state_config: ArchStateConfig
    execution_units: dict[str, str]
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
