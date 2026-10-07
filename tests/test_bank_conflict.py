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
        VMATPUSH_WEIGHT_MXU1(vd=w(0), vs1=m(0)),
        # MXU1 rejects a matmul that reads this slot while its push is active.
        VMATMUL_MXU1(vd=acc(0), vs1=m(2), vs2=w(0)),
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
        VMATMUL_MXU0(vd=acc(0), vs1=m(4), vs2=w(0)),
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
        # MXU0 can read the same weight slot while it is still being filled.
        # Use a different MRF bank so this only tests weight-buffer forwarding.
        VMATMUL_MXU0(vd=acc(0), vs1=m(2), vs2=w(0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoWeightPushMrfConflictProgram(Program):
    instructions: list[Instruction] = [
        VMATPUSH_WEIGHT_MXU0(vd=w(0), vs1=m(0)),
        # Both commands read m0. The push reads it at ages 0..31; DELAY 30
        # puts the matmul 32 issue cycles after the push. DELAY 29 overlaps.
        DELAY(imm=30),
        VMATMUL_MXU0(vd=acc(0), vs1=m(0), vs2=w(0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoMxu1WeightBufConflictProgram(Program):
    instructions: list[Instruction] = [
        VMATPUSH_WEIGHT_MXU1(vd=w(0), vs1=m(0)),
        DELAY(imm=30),
        VMATMUL_MXU1(vd=acc(0), vs1=m(2), vs2=w(0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoAccBufConflictProgram(Program):
    instructions: list[Instruction] = [
        VMATMUL_MXU0(vd=acc(0), vs1=m(0), vs2=w(0)),
        # Wait until the first accumulator row is available to the pop.
        DELAY(imm=62),
        VMATPOP_BF16_ACC_MXU0(vd=m(4), vs2=acc(0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoAccReadAfterWriteConflictProgram(Program):
    instructions: list[Instruction] = [
        VMATMUL_MXU0(vd=acc(0), vs1=m(0), vs2=w(0)),
        DELAY(imm=62),
        VMATMUL_ACC_MXU0(vd=acc(0), vs1=m(2), vs2=w(0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoVpuWriteReadConflictProgram(Program):
    instructions: list[Instruction] = [
        VADD_BF16(vd=m(4), vs1=m(0), vs2=m(2)),
        DELAY(imm=64),
        VMATMUL_MXU0(vd=acc(0), vs1=m(4), vs2=w(0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoXluWriteReadConflictProgram(Program):
    instructions: list[Instruction] = [
        VTRPOSE_XLU(vd=m(4), vs1=m(0)),
        DELAY(imm=64),
        VMATMUL_MXU0(vd=acc(0), vs1=m(4), vs2=w(0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoWeightPushOverlapProgram(Program):
    instructions: list[Instruction] = [
        VMATPUSH_WEIGHT_MXU0(vd=w(0), vs1=m(0)),
        # Weight pushes share one write path; the next waits for the last row.
        DELAY(imm=30),
        VMATPUSH_WEIGHT_MXU0(vd=w(1), vs1=m(2)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoAccPushOverlapProgram(Program):
    instructions: list[Instruction] = [
        VMATPUSH_ACC_FP8_MXU1(vd=acc(0), vs1=m(0)),
        # Accumulator pushes share one write path, even to different accumulators.
        DELAY(imm=30),
        VMATPUSH_ACC_FP8_MXU1(vd=acc(1), vs1=m(2)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoAccReadAfterPushProgram(Program):
    instructions: list[Instruction] = [
        VMATPUSH_ACC_FP8_MXU0(vd=acc(0), vs1=m(0)),
        # Trailing the push by one more cycle reads each row after it lands.
        DELAY(imm=0),
        VMATMUL_ACC_MXU0(vd=acc(0), vs1=m(2), vs2=w(0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoPopAfterPushProgram(Program):
    instructions: list[Instruction] = [
        VMATPUSH_ACC_FP8_MXU1(vd=acc(0), vs1=m(0)),
        DELAY(imm=0),
        VMATPOP_BF16_ACC_MXU1(vd=m(4), vs2=acc(0)),
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
        _NoWeightPushMrfConflictProgram(),
        _NoMxu1WeightBufConflictProgram(),
        _NoAccBufConflictProgram(),
        _NoAccReadAfterWriteConflictProgram(),
        _NoVpuWriteReadConflictProgram(),
        _NoXluWriteReadConflictProgram(),
        _NoWeightPushOverlapProgram(),
        _NoAccPushOverlapProgram(),
        _NoAccReadAfterPushProgram(),
        _NoPopAfterPushProgram(),
    ],
    ids=[
        "NoMrfConflict",
        "NoVpuMrfConflict",
        "NoXluMrfConflict",
        "NoVmemConflict",
        "NoWeightBufConflict",
        "NoWeightPushMrfConflict",
        "NoMxu1WeightBufConflict",
        "NoAccBufConflict",
        "NoAccReadAfterWriteConflict",
        "NoVpuWriteReadConflict",
        "NoXluWriteReadConflict",
        "NoWeightPushOverlap",
        "NoAccPushOverlap",
        "NoAccReadAfterPush",
        "NoPopAfterPush",
    ],
)
def test_non_conflicting_programs_execute(program: Program) -> None:
    run_simulation(program, DefaultHardwareConfig(), max_cycles=500)


@pytest.mark.parametrize(
    ("program", "delay_index", "too_short", "expected_exception", "message"),
    [
        (_NoMxu1WeightBufConflictProgram(), 1, 29, RuntimeError, "active push"),
        (
            _NoWeightPushMrfConflictProgram(),
            1,
            29,
            BankConflictError,
            "MRF bank conflict",
        ),
        (_NoAccBufConflictProgram(), 1, 61, RuntimeError, "row-0 writeback"),
        (_NoAccReadAfterWriteConflictProgram(), 1, 61, RuntimeError, "row-0 writeback"),
        (_NoVpuMrfConflictProgram(), 1, 29, BankConflictError, "MRF bank conflict"),
        (_NoXluMrfConflictProgram(), 1, 30, BankConflictError, "MRF bank conflict"),
        (
            _NoVpuWriteReadConflictProgram(),
            1,
            63,
            BankConflictError,
            "MRF bank conflict",
        ),
        (
            _NoXluWriteReadConflictProgram(),
            1,
            63,
            BankConflictError,
            "MRF bank conflict",
        ),
        (
            _NoWeightPushOverlapProgram(),
            1,
            29,
            RuntimeError,
            "another weight push is still writing",
        ),
        (
            _NoAccPushOverlapProgram(),
            1,
            29,
            RuntimeError,
            "another accumulator push is still writing",
        ),
    ],
    ids=[
        "Mxu1WeightBufferNeeds30Cycles",
        "WeightPushMrfReadNeeds30Cycles",
        "AccumulatorPopNeeds62Cycles",
        "AccumulatorReuseNeeds62Cycles",
        "VpuMrfReadNeeds30Cycles",
        "XluMrfReadNeeds31Cycles",
        "VpuWriteThenMrfReadNeeds64Cycles",
        "XluWriteThenMrfReadNeeds64Cycles",
        "WeightPushOverlapNeeds32Cycles",
        "AccPushOverlapNeeds32Cycles",
    ],
)
def test_minimum_conflict_avoidance_delay(
    program: Program,
    delay_index: int,
    too_short: int,
    expected_exception: type[Exception],
    message: str,
) -> None:
    program.instructions = list(program.instructions)
    program.instructions[delay_index] = DELAY(imm=too_short)
    with pytest.raises(expected_exception, match=message):
        run_simulation(program, DefaultHardwareConfig(), max_cycles=500)


@pytest.mark.parametrize(
    "program",
    [_NoAccReadAfterPushProgram(), _NoPopAfterPushProgram()],
    ids=["MatmulAccDirectlyAfterPush", "PopDirectlyAfterPush"],
)
def test_accumulator_read_cannot_directly_follow_push(program: Program) -> None:
    program.instructions = [
        insn for insn in program.instructions if not isinstance(insn, DELAY)
    ]
    with pytest.raises(
        RuntimeError, match="accumulator read issued the cycle after a push"
    ):
        run_simulation(program, DefaultHardwareConfig(), max_cycles=500)
