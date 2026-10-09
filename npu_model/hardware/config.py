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
    """ Bytes of DRAM mapped from ``dram_base``. The EE290 simulation target maps
    64 GiB (``WithExtMemSize(0x10_0000_0000)``); pages are materialized on
    first touch, so the size costs nothing until used. """
    vmem_size: int
    """ Size of vmem in bytes. """
    dram_base: int = 0x8000_0000
    """ Physical address of the first DRAM byte (``AtlasMemMap.DRAM_BASE``).
    DMA addresses are the 64-bit ``{dma.base, x[rs]}`` the RTL forms; accesses
    outside ``[dram_base, dram_base + dram_size)`` raise. """
    randomize_init: bool = False
    """ Initialize architectural storage with deterministic pseudo-random data. """
    init_seed: int = 42
    """ Seed used when randomize_init is enabled. """
    numerics: str = "pytorch"
    """ Arithmetic used by Instruction.exec: "pytorch" or "rtl" (bit-exact with the RTL). """


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
    vmem_bank_bytes: int = 256 * 1024
    """Contiguous VMEM bank window (six banks in the default RTL)."""

    # DMA engine geometry (atlas/common/DmaParams.scala defaults).
    dma_beat_bytes: int = 32
    """TileLink beat, VMEM line and transfer granularity (DMA_ALIGN)."""
    dma_num_channels: int = 8
    """Channels, and therefore command slots (DMA_CHANNELS)."""
    dma_max_in_flight: int = 64
    """Outstanding TileLink beats; source IDs are 2^ceil(log2(this))."""
    dma_max_transfer_bytes: int = 4096
    """Largest single transfer."""

    # Off-chip memory behind the DMA port (hardware/memory_backend.py).
    dma_memory_backend: str = "curve"
    """Backend kind: "curve" (Mess-style bandwidth-latency curves; params
    ``curves`` names a file in ``npu_model/configs/memory_curves``) or "fixed"
    (rate-limited link with fixed latency, used by directed tests)."""
    dma_memory_params: dict = {"curves": "ee290sim_vcs_probe.json"}
    """Keyword arguments for the backend class. The default curve file was
    measured on VCS simulation of EE290SimConfig, where the DMA port reaches
    DRAMSim2 (DDR3, the Chipyard ``+dramsim`` default) through TileLink
    buffers and a 64-bit AXI memory port: ``scripts/gen_dma_probe_programs.py``
    generates the DMA-side Mess benchmark (a timed 32 B probe against
    background traffic at a sweep of sizes and load/store mixes),
    ``scripts/fit_dma_probe_curve.py`` turns the VCS logs into the file. The
    file gives a lead-off latency of about 46 cycles per Get and 42 per Put
    and a streaming rate of about 23 cycles per 32-byte beat. Measured with
    DRAMSim2 refresh disabled (``../baremetal/dramsim2_ini_norefresh``),
    because refresh adds up to about 130 cycles per transfer window depending
    on when the program starts, which no memory model can reproduce.
    Nothing is fitted; ``scripts/calibrate_dma_backend.py`` reports the
    residual against ``../baremetal/assembly/perf_dma_*.S``."""
    dma_memory_latency_cycles: int | None = None
    """"fixed": channel A fire to channel D valid. None derives the spec's command-overhead cycles."""
    dma_memory_cycles_per_beat: int | None = None
    """"fixed": minimum cycles between accepted beats. None derives the spec's serialized-link rate."""
