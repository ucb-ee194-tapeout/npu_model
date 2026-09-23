"""Clock-edge schedules derived from ScalarCore.scala and LSU.scala."""
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.configs.isa_definition import ADDI, LB, LH, LHU, LW, SELD, SELI, SW, VLOAD, VSTORE
from npu_model.hardware.arch_state import ArchState
from npu_model.hardware.bank_conflict import BankConflictError
from npu_model.hardware.lsu import LoadStoreUnit
from npu_model.hardware.stage_data import StageData
from npu_model.logging.logger import Logger
from npu_model.software import e, m, x
from npu_model.software.instruction import Uop


@pytest.fixture
def lsu():
    cfg = DefaultHardwareConfig()
    cfg.arch_state_config = replace(cfg.arch_state_config, dram_size=4096)
    state = ArchState(cfg.arch_state_config)
    unit = LoadStoreUnit("LSU", Mock(spec=Logger), state, config=cfg)
    yield unit
    state.close()


def tick(unit, insn=None, *, scalar=None):
    stage = StageData(None)
    uop = Uop(insn) if insn is not None else None
    if uop is not None:
        stage.prepare(uop)
    unit.arch_state.current_uop = Uop(scalar) if scalar is not None else uop
    unit.tick(stage)
    assert not stage.is_valid()


def word(state, address, value):
    state.vmem[address:address + 4] = torch.tensor(list(value.to_bytes(4, "little")), dtype=torch.uint8)


def test_scalar_load_command_response_capture_and_writeback(lsu):
    state = lsu.arch_state
    word(state, 0, 0x12345678)
    tick(lsu, LW(rd=x(1), rs1=x(0), imm=0))  # T: scalar issue
    assert state.read_xrf(x(1)) == 0
    tick(lsu)                              # T+1: command / SRAM read
    word(state, 0, 0xDEADBEEF)
    assert state.read_xrf(x(1)) == 0
    tick(lsu)                              # T+2: response capture
    assert state.read_xrf(x(1)) == 0
    tick(lsu)                              # T+3: register writeback
    assert state.read_xrf(x(1)) == 0x12345678
    assert not lsu.has_in_flight


def test_store_address_and_data_are_latched_at_issue(lsu):
    state = lsu.arch_state
    state.write_xrf(x(1), 32)
    state.write_xrf(x(2), 0x11223344)
    tick(lsu, SW(rs1=x(1), rs2=x(2), imm=4))
    assert state.vmem.sum() == 0
    state.write_xrf(x(1), 64)
    state.write_xrf(x(2), 0xFFEEDDCC)
    tick(lsu)
    assert state.vmem[36:40].tolist() == [0x44, 0x33, 0x22, 0x11]
    assert state.vmem[68:72].tolist() == [0, 0, 0, 0]


def test_store_can_issue_while_scalar_load_is_pending(lsu):
    state = lsu.arch_state
    word(state, 0, 123)
    state.write_xrf(x(2), 456)
    tick(lsu, LW(rd=x(1), rs1=x(0), imm=0))
    tick(lsu, SW(rs1=x(0), rs2=x(2), imm=32))
    tick(lsu)
    assert state.vmem[32:36].tolist() == [200, 1, 0, 0]
    assert state.read_xrf(x(1)) == 0
    tick(lsu)
    assert state.read_xrf(x(1)) == 123


@pytest.mark.parametrize("gap", [1, 2])
def test_load_cannot_reissue_before_response_capture(lsu, gap):
    tick(lsu, LW(rd=x(1), rs1=x(0), imm=0))
    for _ in range(gap - 1):
        tick(lsu)
    with pytest.raises(RuntimeError, match="prior load response pending"):
        tick(lsu, LW(rd=x(2), rs1=x(0), imm=0))


def test_load_reissues_during_previous_writeback(lsu):
    word(lsu.arch_state, 0, 7)
    tick(lsu, LW(rd=x(1), rs1=x(0), imm=0))
    tick(lsu)
    tick(lsu)
    tick(lsu, LW(rd=x(2), rs1=x(0), imm=0))
    assert lsu.arch_state.read_xrf(x(1)) == 7
    for _ in range(3):
        tick(lsu)
    assert lsu.arch_state.read_xrf(x(2)) == 7


@pytest.mark.parametrize("scalar", [ADDI(rd=x(2), rs1=x(0), imm=1), SELI(rd=e(2), imm=1)])
def test_scalar_load_writeback_collisions_assert(lsu, scalar):
    tick(lsu, LW(rd=x(1), rs1=x(0), imm=0))
    tick(lsu)
    tick(lsu)
    with pytest.raises(RuntimeError, match="response collides"):
        tick(lsu, scalar=scalar)


@pytest.mark.parametrize(("insn", "value"), [
    (LB(rd=x(1), rs1=x(0), imm=1), 0xFFFFFF80),
    (LH(rd=x(1), rs1=x(0), imm=3), 0xFFFFFFFE),
    (LHU(rd=x(1), rs1=x(0), imm=3), 0xFFFE),
    (LW(rd=x(1), rs1=x(0), imm=3), 0xFFFE8012),
])
def test_scalar_lane_selection_matches_rtl(lsu, insn, value):
    word(lsu.arch_state, 0, 0xFFFE8012)
    tick(lsu, insn)
    for _ in range(3):
        tick(lsu)
    assert lsu.arch_state.read_xrf(x(1)) == value


def test_seld_uses_low_byte_of_selected_word(lsu):
    word(lsu.arch_state, 0, 0x88776655)
    tick(lsu, SELD(rd=e(1), rs1=x(0), imm=3))
    for _ in range(3):
        tick(lsu)
    assert lsu.arch_state.read_erf(e(1)) == 0x55


@pytest.mark.parametrize("load", [True, False])
def test_vector_rows_stream_from_t_plus_3_through_t_plus_34(lsu, load):
    state = lsu.arch_state
    source = state.vmem[:1024] if load else state.mrf[m(0)]
    destination = state.mrf[m(0)] if load else state.vmem[:1024]
    source.fill_(7)
    tick(lsu, (VLOAD if load else VSTORE)(vd=m(0), rs1=x(0), imm=0))
    tick(lsu)  # first SRAM request
    source[:32].fill_(9)  # first row already sampled
    tick(lsu)  # SRAM response / ingress register
    assert destination.count_nonzero() == 0
    tick(lsu)  # first destination write
    assert destination[:32].tolist() == [7] * 32
    assert destination[32:].count_nonzero() == 0
    for age in range(4, 35):
        tick(lsu)
        assert destination[:(age - 2) * 32].count_nonzero() == (age - 2) * 32
        assert destination[(age - 2) * 32:].count_nonzero() == 0
    assert not lsu.has_in_flight
    assert lsu.complete_count == 1


def test_vector_paths_can_overlap_on_different_banks(lsu):
    state = lsu.arch_state
    bank = lsu.config.vmem_bank_bytes
    state.write_xrf(x(1), bank // 4)
    state.vmem[:1024].fill_(1)
    state.mrf[m(1)].fill_(2)
    tick(lsu, VLOAD(vd=m(0), rs1=x(0), imm=0))
    tick(lsu, VSTORE(vd=m(1), rs1=x(1), imm=0))
    for _ in range(34):
        tick(lsu)
    assert state.mrf[m(0)].tolist() == [1] * 1024
    assert state.vmem[bank:bank + 1024].tolist() == [2] * 1024


def test_disjoint_lines_in_same_physical_vmem_bank_conflict(lsu):
    tick(lsu, VLOAD(vd=m(0), rs1=x(0), imm=0))
    tick(lsu, SW(rs1=x(0), rs2=x(0), imm=1024))
    with pytest.raises(BankConflictError, match="VMEM bank conflict"):
        tick(lsu)


def test_scalar_can_access_vmem_while_vector_drains_without_bank_access(lsu):
    tick(lsu, VLOAD(vd=m(0), rs1=x(0), imm=0))
    for _ in range(31):
        tick(lsu)
    tick(lsu, SW(rs1=x(0), rs2=x(0), imm=1024))
    tick(lsu)  # T+33 scalar write; only MREG writes remain for VLOAD.
    tick(lsu)
    assert not lsu.has_in_flight


def test_vector_reissue_waits_until_final_write_has_completed(lsu):
    tick(lsu, VLOAD(vd=m(0), rs1=x(0), imm=0))
    for _ in range(33):
        tick(lsu)
    with pytest.raises(RuntimeError, match="VLOAD path is busy"):
        tick(lsu, VLOAD(vd=m(1), rs1=x(0), imm=0))
