"""DRAM as the RTL maps it: 64 GiB at 0x8000_0000, addressed by {dma.base, x[rs]}."""
import pytest
import torch

from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.configs.isa_definition import *  # noqa: F401, F403
from npu_model.hardware.arch_state import ArchState
from npu_model.hardware.sparse_memory import PAGE_BYTES, SparseMemory
from npu_model.software import x
from npu_model.software.program import InstantiableProgram
from tests.helpers import run_simulation

DRAM_BASE = 0x8000_0000


def test_default_window_matches_ee290_sim_config():
    cfg = DefaultHardwareConfig().arch_state_config
    assert (cfg.dram_base, cfg.dram_size) == (DRAM_BASE, 0x10_0000_0000)
    state = ArchState(cfg)
    assert state.dram.numel() == cfg.dram_size and state.dram.pages() == 0
    state.write_dram(0x9000_0000, torch.arange(64, dtype=torch.uint8))
    assert torch.equal(state.read_dram(0x9000_0000, 64), torch.arange(64, dtype=torch.uint8))
    assert state.dram.pages() == 1                         # only the touched page exists
    assert state.read_dram(DRAM_BASE + cfg.dram_size - 32, 32).sum() == 0
    state.close()


def test_accesses_outside_the_window_are_rejected():
    state = ArchState(DefaultHardwareConfig().arch_state_config)
    with pytest.raises(AssertionError, match="outside the mapped window"):
        state.read_dram(0x7FFF_FFE0, 64)                   # straddles the base
    with pytest.raises(AssertionError, match="outside the mapped window"):
        state.write_dram(0, torch.zeros(1, dtype=torch.uint8))
    with pytest.raises(AssertionError, match="outside the mapped window"):
        state.read_dram(DRAM_BASE + state.cfg.dram_size, 1)
    assert state.read_dram(DRAM_BASE + state.cfg.dram_size, 0).numel() == 0
    state.close()


def test_dma_address_is_base_concatenated_with_the_operand():
    state = ArchState(DefaultHardwareConfig().arch_state_config)
    state.base = 0
    assert state.dma_dram_address(0x9000_0040) == 0x9000_0040
    state.base = 1
    assert state.dma_dram_address(0x40) == 0x1_0000_0040
    state.base = 0x1_0000_0003                             # dma.config takes 32 bits
    assert state.dma_dram_address(0xFFFF_FFFF) == 0x3_FFFF_FFFF
    state.close()


def test_dma_launch_outside_the_window_raises():
    cfg = DefaultHardwareConfig()
    program = InstantiableProgram([
        ADDI(rd=x(1), rs1=x(0), imm=0),                    # VMEM word 0
        ADDI(rd=x(2), rs1=x(0), imm=0x100),                # DRAM 0x100 with base 0: unmapped
        ADDI(rd=x(3), rs1=x(0), imm=64),
        DMA_CONFIG_CH0(rs1=x(0)),
        DMA_WAIT_CH0(),
        DMA_LOAD_CH0(rd=x(1), rs1=x(2), rs2=x(3)),
        DMA_WAIT_CH0(),
    ])
    program.memory_regions = []
    with pytest.raises(AssertionError, match="outside the mapped window"):
        run_simulation(program, cfg, max_cycles=64)


def test_program_regions_land_at_the_program_base_and_dma_reads_them_back():
    cfg = DefaultHardwareConfig()
    payload = torch.arange(64, dtype=torch.uint8)
    program = InstantiableProgram([
        ADDI(rd=x(1), rs1=x(0), imm=0),                    # VMEM word 0
        ADDI(rd=x(2), rs1=x(0), imm=0x100),                # offset into Program.dram_base
        ADDI(rd=x(3), rs1=x(0), imm=64),
        ADDI(rd=x(4), rs1=x(0), imm=1),                    # dma.base 1 = 4 GiB = Program.dram_base
        DMA_CONFIG_CH0(rs1=x(4)),
        DMA_WAIT_CH0(),
        DMA_LOAD_CH0(rd=x(1), rs1=x(2), rs2=x(3)),
        DMA_WAIT_CH0(),
    ])
    program.memory_regions = [(0x100, payload)]
    assert program.dram_base == 1 << 32
    sim = run_simulation(program, cfg, max_cycles=256)
    state = sim.core.arch_state
    assert torch.equal(state.read_dram((1 << 32) + 0x100, 64), payload)
    assert torch.equal(state.read_vmem(0, 0, 64), payload)
    sim.close()


def test_sparse_memory_slices_pages_and_randomizes_lazily():
    mem = SparseMemory(4 * PAGE_BYTES)
    assert mem.shape == (4 * PAGE_BYTES,) and len(mem) == 4 * PAGE_BYTES
    span = slice(PAGE_BYTES - 16, PAGE_BYTES + 16)         # crosses a page boundary
    mem[span] = torch.arange(32, dtype=torch.uint8)
    assert torch.equal(mem[span], torch.arange(32, dtype=torch.uint8))
    assert mem.pages() == 2 and mem[0:16].sum() == 0
    mem[PAGE_BYTES:PAGE_BYTES + 4] = 0xAB
    assert mem[PAGE_BYTES:PAGE_BYTES + 4].tolist() == [0xAB] * 4
    assert torch.equal(mem.dense()[span], mem[span])
    with pytest.raises(ValueError):
        mem[0:8] = torch.zeros(4, dtype=torch.uint8)
    a, b = SparseMemory(1 << 40), SparseMemory(1 << 40)
    a.randomize(42)
    b.randomize(42)
    assert torch.equal(a[0x9000_0000:0x9000_0040], b[0x9000_0000:0x9000_0040])
    assert a[0x9000_0000:0x9000_0040].sum() != 0
    assert a.pages() == 1
