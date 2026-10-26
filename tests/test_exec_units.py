"""Instruction.exec defines results; execution units only schedule them."""
from dataclasses import replace
from io import StringIO
from unittest.mock import Mock

import pytest
import torch

from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.configs.isa_definition import (
    DMA_CONFIG, DMA_LOAD, DMA_STORE, GELU_C1, GELU_C2, LB, LBU, LH, LHU, LW, SB, SELD, SH, SW,
    VADD_BF16, VLI_ALL, VLI_COL, VLOAD, VMATMUL_MXU1, VMATPOP_BF16_ACC_MXU1,
    VMATPUSH_ACC_FP8_MXU1, VSTORE,
)
from npu_model.configs.numerics import NUMERICS
from npu_model.hardware.arch_state import ArchState
from npu_model.hardware.dma import DmaExecutionUnit
from npu_model.hardware.lsu import LoadStoreUnit
from npu_model.hardware.mxu import MXU_MNEMONICS, MatrixExecutionUnitInner
from npu_model.hardware.vpu import VPU_MNEMONICS, VectorExecutionUnit
from npu_model.hardware.xlu import CrossLaneExecutionUnit
from npu_model.logging.logger import Logger
from npu_model.software import acc, e, m, w, x
from npu_model.software.instruction import Uop
from npu_model.util.converter import stream_to_instrs

UNITS = {"VPU": VectorExecutionUnit, "MXU1": MatrixExecutionUnitInner, "XLU": CrossLaneExecutionUnit,
         "LSU": LoadStoreUnit, "DMA": DmaExecutionUnit}
TWO_INPUT = {"vadd.bf16", "vsub.bf16", "vmul.bf16", "vminimum.bf16", "vmaximum.bf16"}
MXU_OPERANDS = {
    "vmatpush.weight": "w1, m8", "vmatpush.acc.fp8": "acc1, m8", "vmatpush.acc.bf16": "acc1, m2",
    "vmatpush.acc.fp32": "acc1, m4",
    "vmatpop.fp8.acc": "m4, acc1, e0", "vmatpop.bf16.acc": "m4, acc1", "vmatpop.bf16.scaled.acc": "m4, acc1, e0",
    "vmatpop.fp32.acc": "m12, acc1",
    "vmatmul": "acc1, m8, w1", "vmatmul.acc": "acc1, m8, w1",
}


def make(unit="VPU", numerics="rtl"):
    cfg = DefaultHardwareConfig()
    cfg.arch_state_config = replace(cfg.arch_state_config, dram_base=0, dram_size=4096, numerics=numerics)
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
    # above 2**31, a DMA length, a DRAM address, the DMA VMEM word address of
    # byte 0x2000, and a DMA base.
    for reg, value in zip(range(1, 8), (0x1003, 0x400, 0xDEADBEEF, 64, 0x100, 0x800, 1)):
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
    if name.endswith(".vx"):
        return f"{name} m4, m0, x3"
    return f"{name} m4, m0, m2" if name in TWO_INPUT or name.endswith(".vr") else f"{name} m4, m0"


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
    assert torch.equal(got.vmem, expected.vmem) and torch.equal(got.dram.dense(), expected.dram.dense())
    for mxu in got.acc:
        for slot in range(2):
            assert torch.equal(got.acc[mxu][slot].view(torch.int16), expected.acc[mxu][slot].view(torch.int16)), (mxu, slot)
            assert torch.equal(got.wb[mxu][slot], expected.wb[mxu][slot]), (mxu, slot)


MEMORY_CASES = [
    ("LSU", insn) for insn in (
        LB(rd=x(10), rs1=x(1), imm=0), LBU(rd=x(10), rs1=x(1), imm=0),
        LH(rd=x(10), rs1=x(1), imm=0), LHU(rd=x(10), rs1=x(1), imm=0),
        LW(rd=x(10), rs1=x(1), imm=1), SELD(rd=e(1), rs1=x(1)),
        SB(rs1=x(1), rs2=x(3), imm=0), SH(rs1=x(1), rs2=x(3), imm=0), SW(rs1=x(1), rs2=x(3), imm=0),
        VLOAD(vd=m(20), rs1=x(2), imm=0), VSTORE(vd=m(0), rs1=x(2), imm=8),
    )
] + [
    ("DMA", insn) for insn in (
        DMA_LOAD(ch=0, rd=x(6), rs1=x(5), rs2=x(4)), DMA_STORE(ch=1, rd=x(5), rs1=x(6), rs2=x(4)),
        DMA_CONFIG(ch=0, rs1=x(7)),
    )
]
CASES = ([("VPU", name) for name in sorted(VPU_MNEMONICS)]
         + [(f"MXU{name[-1]}", name) for name in sorted(MXU_MNEMONICS)]
         + [("XLU", "vtrpose.xlu")] + MEMORY_CASES)


@pytest.mark.parametrize("rows", [32, 3])
@pytest.mark.parametrize("unit,name", CASES, ids=lambda case: getattr(case, "mnemonic", case))
def test_unit_commits_exec_results_on_a_numerics_independent_schedule(unit, name, rows):
    insn = name if not isinstance(name, str) else stream_to_instrs(StringIO(asm(name)))[0]
    schedules = []
    for numerics in ("rtl", "pytorch"):
        exu, reference = make(unit, numerics), make(unit, numerics).arch_state
        exu.arch_state.rows = reference.rows = rows
        insn.exec(reference)
        schedules.append(run(exu, insn))
        assert_same_state(exu.arch_state, reference)
    assert schedules[0] == schedules[1]


@pytest.mark.parametrize("name", sorted(TWO_INPUT))
def test_broadcast_forms_equal_the_bf16_op_on_a_broadcast_tile(name):
    """.vx and .vr give the .bf16 instruction's result, bit for bit, with their operand as a tile."""
    state = make().arch_state
    specials = [0x0000, 0x8000, 0x0001, 0x8001, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x3F80, 0x7F7F]
    for bank in (0, 3):
        state.mrf[bank].view(torch.uint16)[:len(specials)] = torch.tensor(specials, dtype=torch.uint16)
    execute = lambda text: stream_to_instrs(StringIO(text))[0].exec(state)
    for scalar in specials + [0xC0A0]:
        state.write_xrf(9, scalar << 16 | 0x1234)
        execute(f"{name}.vx m4, m0, x9")
        for bank in (6, 7):
            state.mrf[bank].view(torch.uint16)[:] = scalar
        execute(f"{name} m8, m0, m6")
        assert torch.equal(state.mrf[4], state.mrf[8]) and torch.equal(state.mrf[5], state.mrf[9]), hex(scalar)
    execute(f"{name}.vr m4, m0, m2")
    for bank in (6, 7):
        state.mrf[bank].view(torch.uint16).view(32, 16)[:] = state.mrf[bank - 4].view(torch.uint16)[:16]
    execute(f"{name} m8, m0, m6")
    assert torch.equal(state.mrf[4], state.mrf[8]) and torch.equal(state.mrf[5], state.mrf[9])


@pytest.mark.parametrize("numerics", ["rtl", "pytorch"])
def test_vgelu_equals_its_eight_bf16_instructions(numerics):
    """vgelu.bf16 gives, bit for bit, what vcube; vmul c1; vadd x; vmul c2; vtanh; vadd 1; vmul x;
    vmul 0.5 give, with special values among the inputs."""
    state = make(numerics=numerics).arch_state
    specials = [0x0000, 0x8000, 0x0001, 0x8001, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x3F80, 0xC0A0,
                0x7F7F, 0xFF7F, 0x5F00, 0xDF00, 0x0080, 0x8080]
    state.mrf[1].view(torch.uint16)[:len(specials)] = torch.tensor(specials, dtype=torch.uint16)
    execute = lambda text: stream_to_instrs(StringIO(text))[0].exec(state)
    execute("vgelu.bf16 m4, m0")
    for reg, bits in zip(range(9, 13), (GELU_C1, GELU_C2, 0x3F80, 0x3F00)):
        state.write_xrf(reg, bits << 16)
    for text in ("vcube.bf16 m8, m0", "vmul.bf16.vx m8, m8, x9", "vadd.bf16 m8, m8, m0",
                 "vmul.bf16.vx m8, m8, x10", "vtanh.bf16 m8, m8", "vadd.bf16.vx m8, m8, x11",
                 "vmul.bf16 m8, m0, m8", "vmul.bf16.vx m8, m8, x12"):
        execute(text)
    assert torch.equal(state.mrf[4], state.mrf[8]) and torch.equal(state.mrf[5], state.mrf[9])


def test_numerics_backend_selects_rounding():
    x = torch.tensor([[1.0], [1.5 * 2 ** -8]], dtype=torch.bfloat16)
    assert NUMERICS["rtl"].column_sum(x).item() == 1.0  # ColAddVec truncates
    assert NUMERICS["pytorch"].column_sum(x).item() == 1.0078125


def test_rtl_add_and_sub_round_to_nearest_even():
    one = torch.tensor([1.0, 1.0, 1.0], dtype=torch.bfloat16)
    b = torch.tensor([1.5 * 2 ** -8, 2 ** -8, 3 * 2 ** -8], dtype=torch.bfloat16)
    assert NUMERICS["rtl"].add(one, b).tolist() == [1.0078125, 1.0, 1.015625]  # ties to even
    assert NUMERICS["rtl"].sub(one, b).tolist() == [0.9921875, 0.99609375, 0.98828125]


def test_rtl_square_and_cube_round_the_exact_power_once():
    x = torch.tensor([1.4140625, 2 ** -64, -2 ** -44, float("nan")], dtype=torch.bfloat16)
    bits = lambda t: t.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF).tolist()
    # 1.4140625^2 rounds up to 2.0 (v1 truncated); 2^-128 and -2^-132 are subnormal results.
    assert bits(NUMERICS["rtl"].unary("square", x)) == [0x4000, 0x0020, 0x1380, 0x7FC0]
    assert bits(NUMERICS["rtl"].unary("cube", x)) == [0x4035, 0x0000, 0x8002, 0x7FC0]


def test_rtl_nan_results_are_canonical_whatever_torch_emits(monkeypatch):
    """x86 torch writes NaN as 0xffff when it converts to BF16; the RTL emits 0x7fc0."""
    def x86(result):
        result = result.clone()
        result.view(torch.int16)[torch.isnan(result)] = -1
        return result
    mul, to = torch.Tensor.__mul__, torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "__mul__", lambda self, other: x86(mul(self, other)))
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: x86(to(self, *a, **k))
                        if self.dtype != torch.bfloat16 and to(self, *a, **k).dtype == torch.bfloat16
                        else to(self, *a, **k))
    nan = torch.full((2, 32), float("nan"), dtype=torch.bfloat16)
    rtl = NUMERICS["rtl"]
    for result in (rtl.mul(nan, torch.ones_like(nan)), rtl.row_sum(nan)):
        assert result.view(torch.int16).eq(0x7FC0).all()


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


def test_accumulator_read_trailing_a_push_by_two_cycles_sees_it():
    exu, reference = make("MXU1"), make("MXU1").arch_state
    push = VMATPUSH_ACC_FP8_MXU1(vd=acc(0), vs1=m(8))
    pop = VMATPOP_BF16_ACC_MXU1(vd=m(20), vs2=acc(0))
    exu.tick(Uop(push))
    exu.tick(None)
    exu.tick(Uop(pop))  # Gap 2, the minimum: row r is read one cycle after it lands.
    while exu.has_in_flight:
        exu.tick(None)
    push.exec(reference)
    pop.exec(reference)
    assert_same_state(exu.arch_state, reference)


def test_weights_changed_under_the_wavefront_are_rejected():
    unit = make("MXU1")
    unit.tick(Uop(VMATMUL_MXU1(vd=acc(0), vs1=m(8), vs2=w(0))))
    for _ in range(9):
        unit.tick(None)
    unit.arch_state.read_wb_u8("mxu1", 0)[0, 0] ^= 0x01  # Row 0 already used it.
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
    load = DMA_LOAD(ch=0, rd=x(6), rs1=x(5), rs2=x(4))
    unit.tick(Uop(load))
    unit.arch_state.write_xrf(6, 0xC00)  # Retargeting x6 mid-transfer has no effect,
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
