"""Instruction.exec defines results; execution units only schedule them."""
from dataclasses import replace
from io import StringIO
from unittest.mock import Mock

import pytest
import torch

from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.configs.isa_definition import (
    DMA_CONFIG_CH0, DMA_LOAD_CH0, DMA_STORE_CH1, LB, LBU, LH, LHU, LW, SB, SELD, SH, SW,
    VADD_BF16, VLI_ALL, VLI_COL, VLOAD, VMATMUL_MXU0, VMATPUSH_WEIGHT_MXU0, VSTORE,
)
from npu_model.configs.numerics import NUMERICS
from npu_model.hardware.arch_state import ArchState
from npu_model.hardware.dma import DmaExecutionUnit
from npu_model.hardware.lsu import LoadStoreUnit
from npu_model.hardware.mxu import MXU_OP_LATENCIES, MatrixExecutionUnitInner, MatrixExecutionUnitSystolic
from npu_model.hardware.vpu import VPU_OP_LATENCIES, VectorExecutionUnit
from npu_model.hardware.xlu import CrossLaneExecutionUnit
from npu_model.logging.logger import Logger
from npu_model.software import acc, e, m, w, x
from npu_model.software.instruction import Uop
from npu_model.util.converter import stream_to_instrs

UNITS = {"VPU": VectorExecutionUnit, "MXU0": MatrixExecutionUnitSystolic,
         "MXU1": MatrixExecutionUnitInner, "XLU": CrossLaneExecutionUnit,
         "LSU": LoadStoreUnit, "DMA": DmaExecutionUnit}
TWO_INPUT = {"vadd.bf16", "vsub.bf16", "vmul.bf16", "vminimum.bf16", "vmaximum.bf16"}
MXU_OPERANDS = {
    "vmatpush.weight": "w1, m8", "vmatpush.acc.fp8": "acc1, m8", "vmatpush.acc.bf16": "acc1, m2",
    "vmatpop.fp8.acc": "m4, acc1, e0", "vmatpop.bf16.acc": "m4, acc1",
    "vmatmul": "acc1, m8, w1", "vmatmul.acc": "acc1, m8, w1",
}


def make(unit="VPU", numerics="rtl"):
    cfg = DefaultHardwareConfig()
    cfg.arch_state_config = replace(cfg.arch_state_config, dram_size=4096, numerics=numerics)
    state = ArchState(cfg.arch_state_config)
    generator = torch.Generator().manual_seed(0)
    random = lambda shape, scale=4: torch.randn(shape, generator=generator) * scale
    for bank in range(8):
        state.mrf[bank].view(torch.bfloat16)[:] = random(512).to(torch.bfloat16)
    for bank in range(8, 16):
        state.mrf[bank].view(torch.float8_e4m3fn)[:] = random(1024).to(torch.float8_e4m3fn)
    for mxu in state.acc:
        for slot in range(2):
            state.acc[mxu][slot][:] = random((32, 32)).to(torch.bfloat16)
            state.wb[mxu][slot].view(torch.float8_e4m3fn)[:] = random(1024, 1).to(torch.float8_e4m3fn)
    state.vmem[:] = torch.randint(0, 256, state.vmem.shape, generator=generator, dtype=torch.uint8)
    state.dram[:] = torch.randint(0, 256, state.dram.shape, generator=generator, dtype=torch.uint8)
    state.write_erf(0, 129)
    # Unaligned scalar address, VLS word address of byte 0x1000, a store value
    # above 2**31, a DMA length, DRAM and VMEM addresses, and a DMA base.
    for reg, value in zip(range(1, 8), (0x1003, 0x400, 0xDEADBEEF, 64, 0x100, 0x2000, 1)):
        state.write_xrf(reg, value)
    return UNITS[unit](unit, Mock(spec=Logger), state, config=cfg)


def asm(name):
    if name.startswith("vli."):
        return f"{name} m4, 0x3f80"
    if name in {"vpack.bf16.fp8", "vunpack.fp8.bf16"}:
        return f"{name} m4, m2, e0"
    if name == "vtrpose.xlu":
        return f"{name} m4, m8"
    if ".mxu" in name:
        return f"{name} {MXU_OPERANDS[name.rsplit('.', 1)[0]]}"
    return f"{name} m4, m0, m2" if name in TWO_INPUT else f"{name} m4, m0"


def run(unit, insn):
    """Issue ``insn``, drain the unit, and return its MREG write schedule."""
    writes = []
    original = unit.arch_state.conflict_checker.access_mreg

    def record(cycle, bank, row, write, owner):
        original(cycle, bank, row, write, owner)
        if write:
            writes.append((cycle, bank, row))

    unit.arch_state.conflict_checker.access_mreg = record
    unit.tick(Uop(insn))
    while unit.has_in_flight:
        unit.tick(None)
    return writes, unit.cycle


def assert_same_state(got, expected):
    for bank, (a, b) in enumerate(zip(got.mrf, expected.mrf)):
        assert torch.equal(a, b), f"m{bank}"
    assert (got.xrf, got.erf, got.base) == (expected.xrf, expected.erf, expected.base)
    assert torch.equal(got.vmem, expected.vmem) and torch.equal(got.dram, expected.dram)
    for mxu in got.acc:
        for slot in range(2):
            assert torch.equal(got.acc[mxu][slot].view(torch.int16), expected.acc[mxu][slot].view(torch.int16)), (mxu, slot)
            assert torch.equal(got.wb[mxu][slot], expected.wb[mxu][slot]), (mxu, slot)


MEMORY_CASES = [
    ("LSU", insn) for insn in (
        LB(rd=x(10), rs1=x(1), imm=0), LBU(rd=x(10), rs1=x(1), imm=0),
        LH(rd=x(10), rs1=x(1), imm=0), LHU(rd=x(10), rs1=x(1), imm=0),
        LW(rd=x(10), rs1=x(1), imm=1), SELD(rd=e(1), rs1=x(1), imm=0),
        SB(rs1=x(1), rs2=x(3), imm=0), SH(rs1=x(1), rs2=x(3), imm=0), SW(rs1=x(1), rs2=x(3), imm=0),
        VLOAD(vd=m(20), rs1=x(2), imm=0), VSTORE(vd=m(0), rs1=x(2), imm=8),
    )
] + [
    ("DMA", insn) for insn in (
        DMA_LOAD_CH0(rd=x(6), rs1=x(5), rs2=x(4)), DMA_STORE_CH1(rd=x(5), rs1=x(6), rs2=x(4)),
        DMA_CONFIG_CH0(rs1=x(7)),
    )
]
CASES = ([("VPU", name) for name in sorted(VPU_OP_LATENCIES)]
         + [(f"MXU{name[-1]}", name) for name in sorted(MXU_OP_LATENCIES)]
         + [("XLU", "vtrpose.xlu")] + MEMORY_CASES)


@pytest.mark.parametrize("unit,name", CASES, ids=lambda case: getattr(case, "mnemonic", case))
def test_unit_commits_exec_results_on_a_numerics_independent_schedule(unit, name):
    insn = name if not isinstance(name, str) else stream_to_instrs(StringIO(asm(name)))[0]
    schedules = []
    for numerics in ("rtl", "pytorch"):
        exu, reference = make(unit, numerics), make(unit, numerics).arch_state
        insn.exec(reference)
        schedules.append(run(exu, insn))
        assert_same_state(exu.arch_state, reference)
    assert schedules[0] == schedules[1]


def test_numerics_backend_selects_rounding():
    a = torch.tensor([1.0], dtype=torch.bfloat16)
    b = torch.tensor([1.5 * 2 ** -8], dtype=torch.bfloat16)
    assert NUMERICS["rtl"].add(a, b).item() == 1.0  # AddSubSumVec truncates
    assert NUMERICS["pytorch"].add(a, b).item() == 1.0078125


def test_source_overwritten_mid_read_uses_rows_as_sampled():
    unit, reference = make(), make().arch_state
    unit.tick(Uop(VADD_BF16(vd=m(4), vs1=m(0), vs2=m(2))))
    for _ in range(19):
        unit.tick(None)
    # As a VLOAD may: m0 row 3 was already read, row 25 and m1 are read later.
    late = [(0, 25)] + [(1, row) for row in range(32)]
    for state, rows in ((unit.arch_state, [(0, 3)] + late), (reference, late)):
        for bank, row in rows:
            state.mrf[bank][row * 32:(row + 1) * 32] = 0x40
    while unit.has_in_flight:
        unit.tick(None)
    VADD_BF16(vd=m(4), vs1=m(0), vs2=m(2)).exec(reference)
    assert torch.equal(unit.arch_state.mrf[4], reference.mrf[4])
    assert torch.equal(unit.arch_state.mrf[5], reference.mrf[5])


def test_weight_push_leading_a_matmul_matches_sequential_exec():
    unit, reference = make("MXU0"), make("MXU0").arch_state
    push = VMATPUSH_WEIGHT_MXU0(vd=w(0), vs1=m(9))
    matmul = VMATMUL_MXU0(vd=acc(0), vs1=m(8), vs2=w(0))
    unit.tick(Uop(push))
    unit.tick(Uop(matmul))  # Rows of w0 land while the wavefront advances.
    while unit.has_in_flight:
        unit.tick(None)
    push.exec(reference)
    matmul.exec(reference)
    assert_same_state(unit.arch_state, reference)


def test_weights_changed_under_the_wavefront_are_rejected():
    unit = make("MXU0")
    unit.tick(Uop(VMATMUL_MXU0(vd=acc(0), vs1=m(8), vs2=w(0))))
    for _ in range(9):
        unit.tick(None)
    unit.arch_state.read_wb_u8("mxu0", 0)[0, 0] ^= 0x01  # Row 0 already used it.
    with pytest.raises(RuntimeError, match="weights changed while a matmul was using them"):
        unit.tick(None)


def test_load_after_store_sees_the_stored_word():
    unit = make("LSU")
    unit.tick(Uop(SW(rs1=x(1), rs2=x(3), imm=1)))
    unit.tick(Uop(LW(rd=x(10), rs1=x(1), imm=1)))  # Issues before the store lands.
    while unit.has_in_flight:
        unit.tick(None)
    assert unit.arch_state.read_xrf(10) == 0xDEADBEEF


def test_dma_takes_registers_at_launch_and_memory_at_completion():
    unit, reference = make("DMA"), make("DMA").arch_state
    load = DMA_LOAD_CH0(rd=x(6), rs1=x(5), rs2=x(4))
    unit.tick(Uop(load))
    unit.arch_state.write_xrf(6, 0x3000)  # Retargeting x6 mid-transfer has no effect,
    for state in (unit.arch_state, reference):
        state.dram[0x100:0x140] = 7  # but the source is read as the transfer completes.
    while unit.has_in_flight:
        unit.tick(None)
    load.exec(reference)
    assert torch.equal(unit.arch_state.vmem, reference.vmem)


def test_exec_writing_outside_the_unit_schedule_is_rejected(monkeypatch):
    monkeypatch.setattr(VLI_COL, "exec", VLI_ALL.exec)
    with pytest.raises(RuntimeError, match="vli.col exec wrote m5"):
        make().tick(Uop(VLI_COL(vd=m(4), imm=0x3F80)))
