"""Request/write edges from VectorFSM, lane valid flops and XluEngine."""
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.configs.isa_definition import (
    VADD_BF16, VMOV, VLI_ALL, VLI_ROW, VLI_COL, VLI_ONE, VREDSUM_BF16,
    VREDSUM_ROW_BF16, VREDMAX_ROW_BF16, VPACK_BF16_FP8, VUNPACK_FP8_BF16,
    VSQUARE_BF16, VCUBE_BF16, VTRPOSE_XLU,
)
from npu_model.hardware.arch_state import ArchState
from npu_model.hardware.bank_conflict import BankConflictError
from npu_model.hardware.stage_data import StageData
from npu_model.hardware.vpu import VectorExecutionUnit, pack_row, unpack_row
from npu_model.hardware.xlu import CrossLaneExecutionUnit
from npu_model.logging.logger import Logger
from npu_model.software import e, m
from npu_model.software.instruction import Uop


@pytest.fixture
def vpu():
    cfg = DefaultHardwareConfig()
    cfg.arch_state_config = replace(cfg.arch_state_config, dram_size=4096)
    state = ArchState(cfg.arch_state_config)
    unit = VectorExecutionUnit("VPU", Mock(spec=Logger), state, config=cfg)
    yield unit
    state.close()


def tick(unit, insn=None):
    stage = StageData(None)
    if insn is not None:
        stage.prepare(Uop(insn))
    unit.tick(stage)
    assert not stage.is_valid()


def row(state, bank, index):
    return state.mrf[bank][index * 32:(index + 1) * 32]


def fill_pair(state, bank, value):
    for index in (bank, bank + 1):
        state.mrf[index].view(torch.bfloat16).fill_(value)


@pytest.mark.parametrize("op,first,last,count", [
    (VLI_ALL(vd=m(4), imm=0x3F80), 1, 64, 64),
    (VLI_ROW(vd=m(4), imm=0x3F80), 1, 64, 64),
    (VLI_COL(vd=m(5), imm=0x3F80), 1, 32, 32),
    (VLI_ONE(vd=m(5), imm=0x3F80), 1, 32, 32),
    (VMOV(vd=m(4), vs1=m(0)), 2, 65, 64),
    (VADD_BF16(vd=m(4), vs1=m(0), vs2=m(2)), 2, 65, 64),
    (VREDMAX_ROW_BF16(vd=m(4), vs1=m(0)), 2, 33, 64),
    (VREDSUM_ROW_BF16(vd=m(4), vs1=m(0)), 7, 38, 64),
    (VREDSUM_BF16(vd=m(4), vs1=m(0)), 66, 129, 64),
    (VPACK_BF16_FP8(vd=m(4), vs2=m(0), es1=e(0)), 3, 65, 32),
    (VUNPACK_FP8_BF16(vd=m(4), vs2=m(0), es1=e(0)), 3, 66, 64),
])
def test_operation_write_edges(vpu, op, first, last, count):
    state = vpu.arch_state
    fill_pair(state, 0, 1)
    fill_pair(state, 2, 2)
    state.write_erf(0, 127)
    writes = []
    original = state.conflict_checker.access_mreg

    def record(cycle, bank, index, write, owner):
        original(cycle, bank, index, write, owner)
        if write:
            writes.append((cycle - 1, bank, index))

    state.conflict_checker.access_mreg = record
    tick(vpu, op)
    for age in range(1, last + 1):
        assert vpu.has_in_flight
        tick(vpu)
    assert not vpu.has_in_flight
    assert writes[0][0] == first
    assert writes[-1][0] == last
    assert len(writes) == count
    assert len({(bank, index) for _, bank, index in writes}) == count


def test_vli_raw_immediate_and_pair_semantics(vpu):
    state = vpu.arch_state
    tick(vpu, VLI_ROW(vd=m(4), imm=0xBF80))
    for _ in range(64):
        tick(vpu)
    for bank in (4, 5):
        assert torch.all(row(state, bank, 0).view(torch.bfloat16) == -1)
        assert torch.count_nonzero(state.mrf[bank][32:]) == 0


def test_move_samples_each_source_row_at_its_read_edge(vpu):
    state = vpu.arch_state
    fill_pair(state, 0, 1)
    tick(vpu, VMOV(vd=m(4), vs1=m(0)))
    row(state, 0, 0).view(torch.bfloat16).fill_(9)
    row(state, 0, 1).view(torch.bfloat16).fill_(2)
    tick(vpu)
    assert torch.count_nonzero(state.mrf[4]) == 0
    tick(vpu)
    assert torch.all(row(state, 4, 0).view(torch.bfloat16) == 1)
    assert torch.count_nonzero(row(state, 4, 1)) == 0
    tick(vpu)
    assert torch.all(row(state, 4, 1).view(torch.bfloat16) == 2)


def test_column_sum_accumulates_rtl_pair_stream_and_broadcasts(vpu):
    state = vpu.arch_state
    state.mrf[0].view(torch.bfloat16).fill_(1)
    state.mrf[1].view(torch.bfloat16).fill_(2)
    tick(vpu, VREDSUM_BF16(vd=m(4), vs1=m(0)))
    for _ in range(129):
        tick(vpu)
    assert torch.all(state.mrf[4].view(torch.bfloat16) == 96)
    assert torch.all(state.mrf[5].view(torch.bfloat16) == 96)


def test_independent_single_input_operations_overlap(vpu):
    fill_pair(vpu.arch_state, 0, 2)
    fill_pair(vpu.arch_state, 4, 3)
    tick(vpu, VMOV(vd=m(2), vs1=m(0)))
    tick(vpu, VSQUARE_BF16(vd=m(6), vs1=m(4)))
    assert len(vpu.operations) == 2
    for _ in range(65):
        tick(vpu)
    assert torch.all(vpu.arch_state.mrf[2].view(torch.bfloat16) == 2)
    assert torch.all(vpu.arch_state.mrf[6].view(torch.bfloat16) == 9)
    assert not vpu.has_in_flight


@pytest.mark.parametrize("next_op", [
    VMOV(vd=m(6), vs1=m(4)),
    VADD_BF16(vd=m(6), vs1=m(4), vs2=m(8)),
])
def test_issue_busy_rejects_shared_logic_or_double_read(vpu, next_op):
    tick(vpu, VMOV(vd=m(2), vs1=m(0)))
    with pytest.raises(RuntimeError, match="issue-busy"):
        tick(vpu, next_op)


def test_square_cube_share_lane_logic(vpu):
    tick(vpu, VSQUARE_BF16(vd=m(2), vs1=m(0)))
    with pytest.raises(RuntimeError, match="issue-busy"):
        tick(vpu, VCUBE_BF16(vd=m(6), vs1=m(4)))


def test_lane_logic_accepts_next_operation_on_last_write(vpu):
    tick(vpu, VMOV(vd=m(2), vs1=m(0)))
    for _ in range(64):
        tick(vpu)
    tick(vpu, VMOV(vd=m(6), vs1=m(4)))
    assert len(vpu.operations) == 1
    assert vpu.operations[0].uop.insn.vd == 6


def test_source_reservation_released_before_result_drain(vpu):
    tick(vpu, VMOV(vd=m(2), vs1=m(0)))
    for _ in range(63):
        tick(vpu)
    tick(vpu, VLI_COL(vd=m(0), imm=0x3F80))
    assert len(vpu.operations) == 2


def test_destination_reservation_lasts_through_final_write(vpu):
    tick(vpu, VMOV(vd=m(2), vs1=m(0)))
    for _ in range(64):
        tick(vpu)
    with pytest.raises(BankConflictError, match="MRF"):
        tick(vpu, VSQUARE_BF16(vd=m(6), vs1=m(2)))


def test_mirrored_binary_operand_uses_one_physical_read_port(vpu):
    fill_pair(vpu.arch_state, 0, 2)
    tick(vpu, VADD_BF16(vd=m(4), vs1=m(0), vs2=m(0)))
    for _ in range(65):
        tick(vpu)
    assert torch.all(vpu.arch_state.mrf[4].view(torch.bfloat16) == 4)


def test_fp8_conversion_has_rtl_scale_and_flush_behavior():
    values = torch.tensor([1, 2, -1, float("inf"), float("nan"), 2 ** -7], dtype=torch.bfloat16)
    assert pack_row(values, 127).tolist() == [0x38, 0x40, 0xB8, 0x7E, 0, 0]
    assert pack_row(values[:3], 128).tolist() == [0x30, 0x38, 0xB0]
    bits = unpack_row(torch.tensor([0x38, 0x40, 0x81, 0x7F], dtype=torch.uint8), 128)
    assert bits[:2].tolist() == [2, 4]
    assert bits.view(torch.uint16)[2:].tolist() == [0x8000, 0]


def test_xlu_transposes_with_independent_read_and_write_phases(vpu):
    state = vpu.arch_state
    xlu = CrossLaneExecutionUnit("XLU", Mock(spec=Logger), state, config=vpu.config)
    source = torch.arange(1024, dtype=torch.int32).to(torch.uint8).reshape(32, 32)
    state.mrf[8][:] = source.flatten()
    tick(xlu, VTRPOSE_XLU(vd=m(10), vs1=m(8)))
    for _ in range(33):
        tick(xlu)
    assert torch.count_nonzero(state.mrf[10]) == 0
    tick(xlu)
    assert torch.equal(row(state, 10, 0), source[:, 0])
    for _ in range(31):
        tick(xlu)
    assert torch.equal(state.mrf[10].reshape(32, 32), source.T)
    assert not xlu.has_in_flight


def test_xlu_and_vpu_run_concurrently(vpu):
    state = vpu.arch_state
    xlu = CrossLaneExecutionUnit("XLU", Mock(spec=Logger), state, config=vpu.config)
    tick(vpu, VLI_ALL(vd=m(0), imm=0x3F80))
    tick(xlu, VTRPOSE_XLU(vd=m(10), vs1=m(8)))
    for _ in range(65):
        tick(vpu)
        tick(xlu)
    assert not xlu.has_in_flight
    assert not vpu.has_in_flight
