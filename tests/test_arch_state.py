import pytest
import torch

from npu_model.hardware.arch_state import ArchState
from npu_model.hardware.config import ArchStateConfig


def build_state() -> ArchState:
    cfg = ArchStateConfig(
        mrf_depth=32,
        mrf_width=32,
        wb_width=32 * 32,
        num_x_registers=32,
        num_csrs=4,
        num_e_registers=32,
        num_m_registers=64,
        num_wb_registers=2,
        dram_size=1048576,
        vmem_size=256 * 1024,
    )
    return ArchState(cfg)


def test_zero_byte_dram_write_at_end_of_memory() -> None:
    state = build_state()

    state.write_dram(state.cfg.dram_size, torch.zeros(0, dtype=torch.uint8))
    assert state.read_dram(state.cfg.dram_size, 0).numel() == 0


def test_non_empty_dram_write_past_end_fails() -> None:
    state = build_state()

    with pytest.raises(AssertionError):
        state.write_dram(state.cfg.dram_size, torch.tensor([1], dtype=torch.uint8))
