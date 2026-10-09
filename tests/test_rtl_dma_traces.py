"""Replay the actual DmaEngine + Vmem trace with a scripted memory and LSU traffic.

The scenario, responder rule and LSU bank schedule mirror
tests/rtl/scala/dma/NpuModelDmaTraceTest.scala. Every per-cycle observation of
the Python engine (channel A/D fires, VMEM port requests and grants, channel
busy) must match the RTL cycle for cycle.
"""
import json
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.configs.isa_definition import (
    DMA_LOAD_CH0, DMA_LOAD_CH2, DMA_LOAD_CH4, DMA_STORE_CH1, DMA_STORE_CH3, DMA_STORE_CH5,
)
from npu_model.hardware.arch_state import ArchState
from npu_model.hardware.dma import DmaExecutionUnit
from npu_model.hardware.memory_backend import FixedLatencyBackend
from npu_model.logging.logger import Logger
from npu_model.software import x
from npu_model.software.instruction import Uop

LINE = 8   # words per 32-byte VMEM line
# (issue cycle, instruction, VMEM word address, DRAM address, bytes); x operands:
# rd/rs1 carry the VMEM and DRAM addresses, rs2 the size, as the ISA defines.
COMMANDS = [
    (1, DMA_LOAD_CH0, 0 * LINE, 0x1000, 128),
    (2, DMA_STORE_CH1, (8192 + 4) * LINE, 0x2000, 96),
    (3, DMA_LOAD_CH2, 8190 * LINE, 0x3000, 256),
    (4, DMA_STORE_CH3, 0 * LINE, 0x4000, 64),
    (30, DMA_LOAD_CH4, 16384 * LINE, 0x5000, 64),
    (31, DMA_STORE_CH5, 16388 * LINE, 0x6000, 128),
]


def latency(request):
    return 5 + 2 * (request.source % 4)


def a_ready(cycle):
    return cycle % 5 != 0


def lsu_banks(cycle):
    banks = {}
    if 7 <= cycle <= 12:
        banks[0] = "vload"
    elif 45 <= cycle <= 47:
        banks[2] = "vload"
    if cycle in (5, 6, 14):
        banks[1] = "sw"
    return banks


@pytest.fixture
def dma():
    cfg = DefaultHardwareConfig()
    cfg.arch_state_config = replace(cfg.arch_state_config, dram_base=0, dram_size=1 << 20, numerics="rtl")
    state = ArchState(cfg.arch_state_config)
    generator = torch.Generator().manual_seed(1)
    state.dram[:] = torch.randint(0, 256, state.dram.shape, generator=generator, dtype=torch.uint8)
    unit = DmaExecutionUnit("DMA0", Mock(spec=Logger), state, config=cfg)
    unit.backend = FixedLatencyBackend(latency=1, latency_fn=latency, accept_fn=a_ready)
    yield unit
    state.close()


def test_dma_engine_rtl_trace(dma):
    trace = json.loads((Path(__file__).parent / "rtl/dma_traces.json").read_text())
    state = dma.arch_state
    launches = {cycle: (insn, vmem, dram, size) for cycle, insn, vmem, dram, size in COMMANDS}
    for expected in trace:
        cycle = expected["cycle"]
        state.conflict_checker.announce_vmem_ports(cycle, lsu_banks(cycle))
        uop = None
        if cycle in launches:
            insn_cls, vmem, dram, size = launches[cycle]
            store = insn_cls.mnemonic.startswith("dma.store")
            state.write_xrf(1, dram if store else vmem)
            state.write_xrf(2, vmem if store else dram)
            state.write_xrf(3, size)
            insn = insn_cls(rd=x(1), rs1=x(2), rs2=x(3))
            state.set_flag(insn.funct3)
            uop = Uop(insn)
        dma.tick(uop)
        actual = asdict(dma.last_cycle)
        assert actual == expected, cycle
    assert not dma.has_in_flight
