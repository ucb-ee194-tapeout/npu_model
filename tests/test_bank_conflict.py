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
        VADD_BF16(vd=m(5), vs1=m(0), vs2=m(0)),
        VMATMUL_MXU0(vd=acc(0), vs1=m(0), vs2=w(0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _VmemConflictProgram(Program):
    instructions: list[Instruction] = [
        ADDI(rd=x(2), rs1=x(0), imm=1024),
        DMA_CONFIG_CH0(rs1=x(0)),
        DMA_WAIT_CH0(),
        DMA_LOAD_CH0(rd=x(0), rs1=x(0), rs2=x(2)),
        VLOAD(vd=m(0), imm=0, rs1=x(0)),
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
        DELAY(imm=5),
        VMATMUL_MXU0(vd=acc(0), vs1=m(4), vs2=w(0)),
        DELAY(imm=32),
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
        DELAY(imm=64),
        VMATMUL_MXU0(vd=acc(0), vs1=m(0), vs2=w(0)),
        DELAY(imm=64),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoAccBufConflictProgram(Program):
    instructions: list[Instruction] = [
        VMATMUL_MXU0(vd=acc(0), vs1=m(0), vs2=w(0)),
        DELAY(imm=128),
        VMATPOP_BF16_ACC_MXU0(vd=m(4), vs2=acc(0)),
        DELAY(imm=64),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


@pytest.mark.parametrize(
    ("program", "expected_exception"),
    [
        (_MrfConflictProgram(), BankConflictError),
        (_VmemConflictProgram(), BankConflictError),
        (_AccBufConflictProgram(), RuntimeError),
    ],
    ids=[
        "MrfBankConflict",
        "VmemBankConflict",
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
        _NoVmemConflictProgram(),
        _WeightBufConflictProgram(),
        _NoWeightBufConflictProgram(),
        _NoAccBufConflictProgram(),
    ],
    ids=[
        "NoMrfConflict",
        "NoVmemConflict",
        "WeightPushToMatmulOverlap",
        "NoWeightBufConflict",
        "NoAccBufConflict",
    ],
)
def test_non_conflicting_programs_execute(program: Program) -> None:
    run_simulation(program, DefaultHardwareConfig(), max_cycles=500)
