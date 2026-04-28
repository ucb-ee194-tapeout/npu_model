import math
import struct
from dataclasses import dataclass
from typing import Callable

import pytest
import torch

from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.configs.isa_definition import *  # noqa: F401, F403
from npu_model.hardware.arch_state import ArchState
from npu_model.hardware.dma import dma_transfer_cycles
from npu_model.isa import Instruction
from npu_model.software import m, x
from npu_model.software.program import InstantiableProgram
from tests.helpers import run_simulation


TRANSFER_BYTES = 1024
VMEM_SRC_BASE = 0
VMEM_DST_BASE = 1024
DRAM_SRC_BASE = 0

STALE_WORD = 0x11223344
FRESH_WORD = 0xAABBCCDD


@dataclass
class Scenario:
    name: str
    latency_cycles: int
    program: InstantiableProgram
    seed_state: Callable[[ArchState], None]
    stale_reg: int
    fresh_reg: int
    expected_stale: int
    expected_fresh: int
    final_word_addr: int
    expect_violating_load_blocked: bool = False


def repeated_word_bytes(word: int, total_bytes: int) -> torch.Tensor:
    assert total_bytes % 4 == 0
    payload = struct.pack("<I", word) * (total_bytes // 4)
    return torch.tensor(list(payload), dtype=torch.uint8)


def read_word(data: torch.Tensor) -> int:
    return int.from_bytes(bytes(int(x) for x in data[:4]), "little")


def make_dma_load_visibility_scenario(cfg: DefaultHardwareConfig) -> Scenario:

    latency_cycles = dma_transfer_cycles(cfg, TRANSFER_BYTES)
    stale_tile = repeated_word_bytes(STALE_WORD, TRANSFER_BYTES)
    fresh_tile = repeated_word_bytes(FRESH_WORD, TRANSFER_BYTES)

    instrs: list[Instruction] = [
            ADDI(rd=x(1), rs1=x(0), imm=VMEM_DST_BASE),
            ADDI(rd=x(2), rs1=x(0), imm=DRAM_SRC_BASE),
            ADDI(rd=x(3), rs1=x(0), imm=TRANSFER_BYTES),
            DMA_CONFIG_CH0(rs1=x(0)),
            DMA_WAIT_CH0(),
            DELAY(imm=1),
            DMA_LOAD_CH0(rd=x(1), rs1=x(2), rs2=x(3)),
            LW(rd=x(10), imm=0, rs1=x(1)),
            DELAY(imm=latency_cycles),
            LW(rd=x(11), imm=0, rs1=x(1)),
    ]
    
    program = InstantiableProgram(instrs)
    program.memory_regions = [(DRAM_SRC_BASE, fresh_tile.clone())]

    def seed_state(state: ArchState) -> None:
        state.write_vmem(VMEM_DST_BASE, 0, stale_tile.clone())

    return Scenario(
        name="dma.load visibility",
        latency_cycles=latency_cycles,
        program=program,
        seed_state=seed_state,
        stale_reg=10,
        fresh_reg=11,
        expected_stale=STALE_WORD,
        expected_fresh=FRESH_WORD,
        final_word_addr=VMEM_DST_BASE,
    )


def make_vstore_visibility_scenario(cfg: DefaultHardwareConfig) -> Scenario:
    latency_cycles = 34
    stale_tile = repeated_word_bytes(STALE_WORD, TRANSFER_BYTES)
    fresh_tile = repeated_word_bytes(FRESH_WORD, TRANSFER_BYTES)

    instrs: list[Instruction] = [
        ADDI(rd=x(1), rs1=x(0), imm=VMEM_SRC_BASE),
        ADDI(rd=x(2), rs1=x(0), imm=VMEM_DST_BASE),
        ADDI(rd=x(10), rs1=x(0), imm=0),
        VLOAD(vd=m(0), imm=0, rs1=x(1)),
        DELAY(imm=latency_cycles),
        VSTORE(vd=m(0), imm=0, rs1=x(2)),
        LW(rd=x(10), imm=0, rs1=x(2)),
        DELAY(imm=latency_cycles),
        LW(rd=x(11), imm=0, rs1=x(2)),
    ]
    program = InstantiableProgram(instrs)

    def seed_state(state: ArchState) -> None:
        state.write_vmem(VMEM_SRC_BASE, 0, fresh_tile.clone())
        state.write_vmem(VMEM_DST_BASE, 0, stale_tile.clone())

    return Scenario(
        name="vstore visibility",
        latency_cycles=latency_cycles,
        program=program,
        seed_state=seed_state,
        stale_reg=10,
        fresh_reg=11,
        expected_stale=0,
        expected_fresh=FRESH_WORD,
        final_word_addr=VMEM_DST_BASE,
        expect_violating_load_blocked=True,
    )


@pytest.mark.parametrize(
    "scenario_factory",
    [make_dma_load_visibility_scenario, make_vstore_visibility_scenario],
    ids=["dma_load_visibility", "vstore_visibility"],
)
def test_memory_producers_commit_after_modeled_latency(scenario_factory) -> None:
    cfg = DefaultHardwareConfig()
    scenario = scenario_factory(cfg)
    sim = run_simulation(
        scenario.program,
        cfg,
        max_cycles=max(256, scenario.latency_cycles + 32),
        ignore_runtime_errors=True,
        before_run=lambda simulation: scenario.seed_state(simulation.core.arch_state),
    )

    stale_seen = sim.core.arch_state.read_xrf(scenario.stale_reg)
    fresh_seen = sim.core.arch_state.read_xrf(scenario.fresh_reg)
    final_word = read_word(
        sim.core.arch_state.read_vmem(scenario.final_word_addr, 0, 4)
    )

    if scenario.expect_violating_load_blocked:
        assert sim.runtime_errors, (
            f"{scenario.name}: expected scheduler/runtime rejection for the "
            "violating load, but none occurred"
        )
        assert stale_seen == scenario.expected_stale, (
            f"{scenario.name}: violating load should be blocked and leave the "
            f"destination register unchanged (0x{scenario.expected_stale:08X}), "
            f"got 0x{stale_seen:08X}"
        )
    else:
        assert stale_seen == scenario.expected_stale, (
            f"{scenario.name}: violating load should observe stale value "
            f"0x{scenario.expected_stale:08X}, got 0x{stale_seen:08X}"
        )

    assert fresh_seen == scenario.expected_fresh, (
        f"{scenario.name}: delayed load should observe fresh value "
        f"0x{scenario.expected_fresh:08X}, got 0x{fresh_seen:08X}"
    )
    assert final_word == scenario.expected_fresh, (
        f"{scenario.name}: final VMEM word should be fresh value "
        f"0x{scenario.expected_fresh:08X}, got 0x{final_word:08X}"
    )
