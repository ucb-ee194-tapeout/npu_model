from typing import List, Tuple

import pytest
import torch

from npu_model.configs.hardware import DefaultHardwareConfig
from npu_model.configs.isa_definition import *  # noqa: F401, F403
from npu_model.hardware.bank_conflict import BankConflictError
from npu_model.isa import Instruction
from npu_model.software import Program, acc, m, w, x
from tests.helpers import run_simulation


class _MrfConflictProgram(Program):
    instructions: list[Instruction] = [
        VADD_BF16(vd=m(4), vs1=m(0), vs2=m(0)),
        VMATMUL_MXU0(vd=acc(0), vs1=m(4), vs2=w(0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _VmemConflictProgram(Program):
    instructions: list[Instruction] = [
        VLOAD(vd=m(0), imm=0, rs1=x(0)),
        VSTORE(vd=m(2), imm=0, rs1=x(0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0, torch.zeros(1024, dtype=torch.uint8)),
    ]


class _WeightBufConflictProgram(Program):
    instructions: list[Instruction] = [
        VMATPUSH_WEIGHT_MXU0(vd=w(0), vs1=m(0)),
        VMATMUL_MXU0(vd=acc(0), vs1=m(0), vs2=w(0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _AccBufConflictProgram(Program):
    instructions: list[Instruction] = [
        VMATMUL_MXU0(vd=acc(0), vs1=m(0), vs2=w(0)),
        VMATPOP_BF16_ACC_MXU0(vd=m(4), vs2=acc(0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoMrfConflictProgram(Program):
    instructions: list[Instruction] = [
        VADD_BF16(vd=m(6), vs1=m(0), vs2=m(2)),
        DELAY(imm=0),
        VMATMUL_MXU0(vd=acc(0), vs1=m(4), vs2=w(0)),
        DELAY(imm=0),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoVpuMrfConflictProgram(Program):
    instructions: list[Instruction] = [
        VADD_BF16(vd=m(6), vs1=m(0), vs2=m(2)),
        # Let the VPU finish reading m0 before MXU0 reads the same bank.
        DELAY(imm=30),
        VMATMUL_MXU0(vd=acc(0), vs1=m(0), vs2=w(0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoXluMrfConflictProgram(Program):
    instructions: list[Instruction] = [
        VTRPOSE_XLU(vd=m(6), vs1=m(0)),
        # XLU starts its MRF read phase one cycle after issue.
        DELAY(imm=31),
        VMATMUL_MXU0(vd=acc(0), vs1=m(0), vs2=w(0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoVmemConflictProgram(Program):
    instructions: list[Instruction] = [
        ADDI(rd=x(2), rs1=x(0), imm=1024),
        ADDI(rd=x(3), rs1=x(0), imm=1024),
        DMA_CONFIG_CH0(rs1=x(0)),
        DMA_WAIT_CH0(),
        DMA_LOAD_CH0(rd=x(0), rs1=x(0), rs2=x(2)),
        VLOAD(vd=m(0), imm=0, rs1=x(3)),
        DMA_WAIT_CH0(),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0, torch.zeros(1024, dtype=torch.uint8)),
        (1024, torch.zeros(1024, dtype=torch.uint8)),
    ]


class _NoWeightBufConflictProgram(Program):
    instructions: list[Instruction] = [
        VMATPUSH_WEIGHT_MXU0(vd=w(0), vs1=m(0)),
        # Both commands read m0 through the same physical MRF bank. The
        # minimum delay that avoids overlapping those reads is 30 cycles.
        DELAY(imm=30),
        VMATMUL_MXU0(vd=acc(0), vs1=m(0), vs2=w(0)),
        DELAY(imm=0),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoAccBufConflictProgram(Program):
    instructions: list[Instruction] = [
        VMATMUL_MXU0(vd=acc(0), vs1=m(0), vs2=w(0)),
        # Wait until the first accumulator row is available to the pop.
        DELAY(imm=62),
        VMATPOP_BF16_ACC_MXU0(vd=m(4), vs2=acc(0)),
        DELAY(imm=0),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


@pytest.mark.parametrize(
    ("program", "expected_exception"),
    [
        (_MrfConflictProgram(), BankConflictError),
        (_VmemConflictProgram(), BankConflictError),
        (_WeightBufConflictProgram(), RuntimeError),
        (_AccBufConflictProgram(), RuntimeError),
    ],
    ids=[
        "MrfBankConflict",
        "VmemBankConflict",
        "WeightBufSchedulingViolation",
        "AccBufSchedulingViolation",
    ],
)
def test_conflicting_programs_fail(program: Program, expected_exception) -> None:
    with pytest.raises(expected_exception, match=".*"):
        run_simulation(program, DefaultHardwareConfig(), max_cycles=500)


@pytest.mark.parametrize(
    "program",
    [
        _NoMrfConflictProgram(),
        _NoVpuMrfConflictProgram(),
        _NoXluMrfConflictProgram(),
        _NoVmemConflictProgram(),
        _NoWeightBufConflictProgram(),
        _NoAccBufConflictProgram(),
    ],
    ids=[
        "NoMrfConflict",
        "NoVpuMrfConflict",
        "NoXluMrfConflict",
        "NoVmemConflict",
        "NoWeightBufConflict",
        "NoAccBufConflict",
    ],
)
def test_non_conflicting_programs_execute(program: Program) -> None:
    run_simulation(program, DefaultHardwareConfig(), max_cycles=500)


@pytest.mark.parametrize(
    ("program", "delay_index", "too_short"),
    [
        (_NoWeightBufConflictProgram(), 1, 29),
        (_NoAccBufConflictProgram(), 1, 61),
        (_NoVpuMrfConflictProgram(), 1, 29),
        (_NoXluMrfConflictProgram(), 1, 30),
    ],
    ids=[
        "WeightPushNeeds30Cycles",
        "AccumulatorPopNeeds62Cycles",
        "VpuReadNeeds30Cycles",
        "XluReadNeeds31Cycles",
    ],
)
def test_minimum_conflict_avoidance_delay(
    program: Program, delay_index: int, too_short: int
) -> None:
    program.instructions = list(program.instructions)
    program.instructions[delay_index] = DELAY(imm=too_short)
    with pytest.raises(RuntimeError, match=".*"):
        run_simulation(program, DefaultHardwareConfig(), max_cycles=500)
