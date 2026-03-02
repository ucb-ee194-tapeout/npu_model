from dataclasses import dataclass

from model_npu.isa import IsaSpec


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
    num_m_registers: int
    """ Number of matrix registers. """
    num_wb_registers: int
    """ Number of weight buffer entries. """
    memory_size: int
    """ Size of memory in bytes. """


class HardwareConfig:
    name: str
    fetch_width: int
    isa: IsaSpec = IsaSpec
    arch_state_config: ArchStateConfig
    execution_units: dict[str, str]
