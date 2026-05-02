import pytest
import torch

from npu_model.hardware.arch_state import ArchState
from npu_model.hardware.config import ArchStateConfig


def build_state(*, randomize_init: bool = True) -> ArchState:
    cfg = ArchStateConfig(
        mrf_depth=32,
        mrf_width=32,
        wb_width=32 * 32,
        num_x_registers=32,
        num_csrs=4,
        num_e_registers=32,
        num_m_registers=64,
        num_wb_registers=2,
        dram_base=0x80000000,
        dram_size=1048576,
        vmem_base=0x20000000,
        vmem_size=256 * 1024,
        randomize_init=randomize_init,
        init_seed=42,
    )
    return ArchState(cfg)


def test_zero_byte_dram_write_at_end_of_memory() -> None:
    state = build_state()
    end = state.cfg.dram_base + state.cfg.dram_size

    state.write_dram(end, torch.zeros(0, dtype=torch.uint8))
    assert state.read_dram(end, 0).numel() == 0


def test_non_empty_dram_write_past_end_fails() -> None:
    state = build_state()
    end = state.cfg.dram_base + state.cfg.dram_size

    with pytest.raises(AssertionError):
        state.write_dram(end, torch.tensor([1], dtype=torch.uint8))


def test_zero_init_is_default_when_randomization_is_disabled() -> None:
    state = build_state(randomize_init=False)

    assert state.read_dram(state.cfg.dram_base, 16).sum().item() == 0
    assert state.read_vmem(state.cfg.vmem_base, 0, 16).sum().item() == 0
    assert state.xrf == [0] * len(state.xrf)


def test_random_init_is_deterministic_with_seed_42() -> None:
    a = build_state(randomize_init=True)
    b = build_state(randomize_init=True)

    assert torch.equal(a.read_dram(a.cfg.dram_base, 64), b.read_dram(b.cfg.dram_base, 64))
    assert torch.equal(a.read_vmem(a.cfg.vmem_base, 0, 64), b.read_vmem(b.cfg.vmem_base, 0, 64))
    assert a.xrf == b.xrf
    assert a.csrf == b.csrf
    assert a.read_dram(a.cfg.dram_base, 64).sum().item() != 0
    assert a.read_vmem(a.cfg.vmem_base, 0, 64).sum().item() != 0
