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
    imem_size: int
    """ Size of imem in bytes. """


class HardwareConfig:
    name: str
    fetch_width: int
    isa: type[IsaSpec] = IsaSpec
    arch_state_config: ArchStateConfig
    execution_units: dict[str, str]
    mxu0_matmul_latency_cycles: int = 32
    mxu1_matmul_latency_cycles: int = 32
    vpu_simple_op_latency_cycles: int = 2
    vpu_non_pipelineable_op_latency_cycles: int = 8
    xlu_transform_latency_cycles: int = 4
    vmem_bytes_per_cycle: int = 64
