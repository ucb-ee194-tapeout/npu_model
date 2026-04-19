from typing import Any, List, Tuple

import pytest
import torch

from npu_model.configs.hardware import DefaultHardwareConfig
from npu_model.configs.isa_definition import *  # noqa: F401, F403
from npu_model.hardware.bank_conflict import BankConflictError
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs
from npu_model.software import Instruction, Program
from tests.helpers import run_simulation


class _MrfConflictProgram(Program):
    instructions: List[Instruction[Any]] = [
        Instruction("vadd.bf16", VectorArgs(vd=5, vs1=0, vs2=0)),
        Instruction("vmatmul.mxu0", MatrixArgs(vd=0, vs1=0, vs2=0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _VmemConflictProgram(Program):
    instructions: List[Instruction[Any]] = [
        Instruction("addi", ScalarArgs(rd=2, rs1=0, imm=1024)),
        Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=0, rs1=0, rs2=2, channel=0)),
        Instruction("vload", VectorArgs(vd=0, rs1=0, imm12=0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0, torch.zeros(1024, dtype=torch.uint8)),
    ]


class _WeightBufConflictProgram(Program):
    instructions: List[Instruction[Any]] = [
        Instruction("vmatpush.weight.mxu0", VectorArgs(vd=0, vs1=0)),
        Instruction("vmatmul.mxu0", MatrixArgs(vd=0, vs1=0, vs2=0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _AccBufConflictProgram(Program):
    instructions: List[Instruction[Any]] = [
        Instruction("vmatmul.mxu0", MatrixArgs(vd=0, vs1=0, vs2=0)),
        Instruction("vmatpop.bf16.acc.mxu0", VectorArgs(vd=4, vs1=0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoMrfConflictProgram(Program):
    instructions: List[Instruction[Any]] = [
        Instruction("vadd.bf16", VectorArgs(vd=6, vs1=0, vs2=2)),
        Instruction("delay", ScalarArgs(imm=5)),
        Instruction("vmatmul.mxu0", MatrixArgs(vd=0, vs1=4, vs2=0)),
        Instruction("delay", ScalarArgs(imm=32)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoVmemConflictProgram(Program):
    instructions: List[Instruction[Any]] = [
        Instruction("addi", ScalarArgs(rd=2, rs1=0, imm=1024)),
        Instruction("addi", ScalarArgs(rd=3, rs1=0, imm=1024)),
        Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=0, rs1=0, rs2=2, channel=0)),
        Instruction("vload", VectorArgs(vd=0, rs1=3, imm12=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0, torch.zeros(1024, dtype=torch.uint8)),
        (1024, torch.zeros(1024, dtype=torch.uint8)),
    ]


class _NoWeightBufConflictProgram(Program):
    instructions: List[Instruction[Any]] = [
        Instruction("vmatpush.weight.mxu0", VectorArgs(vd=0, vs1=0)),
        Instruction("delay", ScalarArgs(imm=64)),
        Instruction("vmatmul.mxu0", MatrixArgs(vd=0, vs1=0, vs2=0)),
        Instruction("delay", ScalarArgs(imm=64)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoAccBufConflictProgram(Program):
    instructions: List[Instruction[Any]] = [
        Instruction("vmatmul.mxu0", MatrixArgs(vd=0, vs1=0, vs2=0)),
        Instruction("delay", ScalarArgs(imm=128)),
        Instruction("vmatpop.bf16.acc.mxu0", VectorArgs(vd=4, vs1=0)),
        Instruction("delay", ScalarArgs(imm=64)),
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
        _NoVmemConflictProgram(),
        _NoWeightBufConflictProgram(),
        _NoAccBufConflictProgram(),
    ],
    ids=[
        "NoMrfConflict",
        "NoVmemConflict",
        "NoWeightBufConflict",
        "NoAccBufConflict",
    ],
)
def test_non_conflicting_programs_execute(program: Program) -> None:
    run_simulation(program, DefaultHardwareConfig(), max_cycles=500)
