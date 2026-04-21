from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.hardware.config import ArchStateConfig


class CIHardwareConfig(DefaultHardwareConfig):
    """Lean test config for GitHub Actions runners.

    The current program suite only touches DRAM up to ~0x7000 and VMEM up to
    ~0xF000, so 64 KiB for each region preserves headroom while cutting the
    simulator's buffer footprint in CI.
    """

    arch_state_config: ArchStateConfig = ArchStateConfig(
        mrf_depth=DefaultHardwareConfig.arch_state_config.mrf_depth,
        mrf_width=DefaultHardwareConfig.arch_state_config.mrf_width,
        wb_width=DefaultHardwareConfig.arch_state_config.wb_width,
        num_x_registers=DefaultHardwareConfig.arch_state_config.num_x_registers,
        num_csrs=DefaultHardwareConfig.arch_state_config.num_csrs,
        num_e_registers=DefaultHardwareConfig.arch_state_config.num_e_registers,
        num_m_registers=DefaultHardwareConfig.arch_state_config.num_m_registers,
        num_wb_registers=DefaultHardwareConfig.arch_state_config.num_wb_registers,
        dram_size=256 * 1024,
        vmem_size=256 * 1024,
    )
