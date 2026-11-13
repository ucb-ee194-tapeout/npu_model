"""Resized Atlas geometries, as the RTL's ``WithAtlasTileDim(n)`` builds them.

Tile N means N x N tensor registers (N rows of N bytes), N x N MXU arrays and
weight slots, and N / 2 BF16 VPU lanes; every latency follows from N
(``npu_model/geometry.py``). VMEM, its 32-byte lines, the DMA and everything
else are the default config.
"""
from npu_model.hardware.config import scaled_arch_state

from .default import DefaultHardwareConfig


class Tile16HardwareConfig(DefaultHardwareConfig):
    """16 x 16 tiles: 256-byte tensor registers, 16 x 16 MXUs, 8 BF16 lanes."""

    name: str = "Tile16"
    arch_state_config = scaled_arch_state(DefaultHardwareConfig.arch_state_config, 16)


class Tile64HardwareConfig(DefaultHardwareConfig):
    """64 x 64 tiles: 4 KiB tensor registers, 64 x 64 MXUs, 32 BF16 lanes."""

    name: str = "Tile64"
    arch_state_config = scaled_arch_state(DefaultHardwareConfig.arch_state_config, 64)
