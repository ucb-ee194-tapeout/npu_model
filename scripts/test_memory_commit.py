#!/usr/bin/env python3
"""
Verify delayed commit behavior for multi-cycle memory producers.

The harness intentionally issues an `lw` too early after:
- `dma.load.ch<N>` writing DRAM data into VMEM
- `vstore` writing MRF data into VMEM

Those violating reads should observe the stale VMEM contents that were seeded before
execution. A second `lw` after the modeled latency should observe the fresh value.
"""

import math
import io
import struct
import sys
import tempfile
from dataclasses import dataclass
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable

import torch

# Add project root for imports when run as script
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.configs.isa_definition import *  # noqa: F401, F403
from npu_model.hardware.arch_state import ArchState
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation
from npu_model.software.instruction import Instruction
from npu_model.software.program import InstantiableProgram


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


def repeated_word_bytes(word: int, total_bytes: int) -> torch.Tensor:
    assert total_bytes % 4 == 0
    payload = struct.pack("<I", word) * (total_bytes // 4)
    return torch.tensor(list(payload), dtype=torch.uint8)


def read_word(data: torch.Tensor) -> int:
    return int.from_bytes(bytes(int(x) for x in data[:4]), "little")


def make_dma_load_visibility_scenario(cfg: DefaultHardwareConfig) -> Scenario:
    latency_cycles = max(1, math.ceil(TRANSFER_BYTES / cfg.vmem_bytes_per_cycle))
    stale_tile = repeated_word_bytes(STALE_WORD, TRANSFER_BYTES)
    fresh_tile = repeated_word_bytes(FRESH_WORD, TRANSFER_BYTES)

    program = InstantiableProgram(
        [
            Instruction("addi", ScalarArgs(rd=1, rs1=0, imm=VMEM_DST_BASE)),
            Instruction("addi", ScalarArgs(rd=2, rs1=0, imm=DRAM_SRC_BASE)),
            Instruction("addi", ScalarArgs(rd=3, rs1=0, imm=TRANSFER_BYTES)),
            Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=0)),
            Instruction("delay", ScalarArgs(imm=1)),
            Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=2, rs2=3, channel=0)),
            Instruction("lw", ScalarArgs(rd=10, rs1=1, imm=0)),
            Instruction("delay", ScalarArgs(imm=latency_cycles)),
            Instruction("lw", ScalarArgs(rd=11, rs1=1, imm=0)),
        ]
    )
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
    latency_cycles = max(1, math.ceil(TRANSFER_BYTES / cfg.vmem_bytes_per_cycle))
    stale_tile = repeated_word_bytes(STALE_WORD, TRANSFER_BYTES)
    fresh_tile = repeated_word_bytes(FRESH_WORD, TRANSFER_BYTES)

    program = InstantiableProgram(
        [
            Instruction("addi", ScalarArgs(rd=1, rs1=0, imm=VMEM_SRC_BASE)),
            Instruction("addi", ScalarArgs(rd=2, rs1=0, imm=VMEM_DST_BASE)),
            Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)),
            Instruction("delay", ScalarArgs(imm=latency_cycles)),
            Instruction("vstore", VectorArgs(vd=0, rs1=2, imm12=0)),
            Instruction("lw", ScalarArgs(rd=10, rs1=2, imm=0)),
            Instruction("delay", ScalarArgs(imm=latency_cycles)),
            Instruction("lw", ScalarArgs(rd=11, rs1=2, imm=0)),
        ]
    )

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
        expected_stale=STALE_WORD,
        expected_fresh=FRESH_WORD,
        final_word_addr=VMEM_DST_BASE,
    )


def run_scenario(scenario: Scenario, cfg: DefaultHardwareConfig) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        trace_path = f.name

    try:
        sim = Simulation(
            hardware_config=cfg,
            logger_config=LoggerConfig(filename=trace_path),
            program=scenario.program,
            verbose=False,
        )
        scenario.seed_state(sim.core.arch_state)
        with redirect_stdout(io.StringIO()):
            sim.run(max_cycles=256)

        stale_seen = sim.core.arch_state.read_xrf(scenario.stale_reg)
        fresh_seen = sim.core.arch_state.read_xrf(scenario.fresh_reg)
        final_word = read_word(
            sim.core.arch_state.read_vmem(scenario.final_word_addr, 0, 4)
        )

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

        print(
            f"OK  {scenario.name:<20} latency={scenario.latency_cycles:>2} cycles  "
            f"early=0x{stale_seen:08X}  late=0x{fresh_seen:08X}"
        )
    finally:
        Path(trace_path).unlink(missing_ok=True)


def main() -> int:
    cfg = DefaultHardwareConfig()
    scenarios = [
        make_dma_load_visibility_scenario(cfg),
        make_vstore_visibility_scenario(cfg),
    ]

    for scenario in scenarios:
        run_scenario(scenario, cfg)

    print(f"Verified {len(scenarios)} delayed-commit scenarios.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
