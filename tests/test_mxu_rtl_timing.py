"""Default-geometry RTL command, port, wavefront, and row-write schedules."""
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.configs import isa_definition as isa
from npu_model.hardware.arch_state import ArchState
from npu_model.hardware.bank_conflict import BankConflictError
from npu_model.hardware.mxu import MatrixExecutionUnitInner
from npu_model.logging.logger import Logger
from npu_model.software import acc, m, w
from npu_model.software.instruction import Uop


@pytest.fixture
def mxu():
    cfg = DefaultHardwareConfig()
    cfg.arch_state_config = replace(cfg.arch_state_config, dram_size=4096, vmem_size=4096, numerics="rtl")
    state = ArchState(cfg.arch_state_config)
    unit = MatrixExecutionUnitInner("Matrix1", Mock(spec=Logger), state, config=cfg)
    yield unit
    state.close()


def instruction(unit, name, **kwargs):
    return getattr(isa, f"{name}_MXU{unit.mxu[-1]}")(**kwargs)


def tick(unit, insn=None):
    unit.tick(Uop(insn) if insn is not None else None)


def put_fp8(state, reg, values):
    state.write_mrf_fp8(reg, values.to(torch.float8_e4m3fn))


def test_push_weight_streams_t_plus_1_to_t_plus_32(mxu):
    state = mxu.arch_state
    put_fp8(state, m(0), torch.ones(32, 32))
    tick(mxu, instruction(mxu, "VMATPUSH_WEIGHT", vd=w(0), vs1=m(0)))
    assert state.read_wb_u8(mxu.mxu, w(0)).count_nonzero() == 0
    put_fp8(state, m(0), torch.full((32, 32), 2))
    tick(mxu)
    assert state.read_wb_fp8(mxu.mxu, w(0))[0].float().tolist() == [1] * 32
    assert state.read_wb_u8(mxu.mxu, w(0))[1:].count_nonzero() == 0
    for _ in range(31):
        tick(mxu)
    assert state.read_wb_fp8(mxu.mxu, w(0))[1:].float().tolist() == [[2] * 32] * 31
    assert not mxu.has_in_flight


def test_compute_streams_first_and_last_row_at_rtl_latency(mxu):
    state = mxu.arch_state
    put_fp8(state, m(0), torch.ones(32, 32))
    state.write_wb_fp8(mxu.mxu, w(0), torch.eye(32).to(torch.float8_e4m3fn))
    tick(mxu, instruction(mxu, "VMATMUL", vd=acc(0), vs1=m(0), vs2=w(0)))
    for _ in range(mxu.compute_first - 1):
        tick(mxu)
    assert state.acc[mxu.mxu][0].count_nonzero() == 0
    for row in range(32):
        tick(mxu)
        assert state.acc[mxu.mxu][0][:row + 1].float().tolist() == [[1] * 32] * (row + 1)
        assert state.acc[mxu.mxu][0][row + 1:].count_nonzero() == 0
    assert not mxu.has_in_flight
    assert mxu.complete_count == 1


def test_compute_uses_weight_rows_as_output_lanes(mxu):
    state = mxu.arch_state
    activation = torch.zeros(32, 32)
    activation[:, 3] = 1
    weights = torch.zeros(32, 32)
    weights[7, 3] = 2
    put_fp8(state, m(0), activation)
    state.write_wb_fp8(mxu.mxu, w(0), weights.to(torch.float8_e4m3fn))
    tick(mxu, instruction(mxu, "VMATMUL", vd=acc(0), vs1=m(0), vs2=w(0)))
    for _ in range(mxu.compute_first + 31):
        tick(mxu)
    expected = torch.zeros(32, 32)
    expected[:, 7] = 2
    assert torch.equal(state.acc[mxu.mxu][0].float(), expected)


def test_pop_can_follow_first_result_while_compute_drains(mxu):
    state = mxu.arch_state
    put_fp8(state, m(0), torch.ones(32, 32))
    state.write_wb_fp8(mxu.mxu, w(0), torch.eye(32).to(torch.float8_e4m3fn))
    tick(mxu, instruction(mxu, "VMATMUL", vd=acc(0), vs1=m(0), vs2=w(0)))
    for _ in range(max(mxu.compute_first, 31)):
        tick(mxu)
    # Row zero must be written and compute's shared accumulator read port
    # must be free (the latter is the limiting condition for IPT).
    tick(mxu, instruction(mxu, "VMATPOP_BF16_ACC", vd=m(2), vs2=acc(0)))
    for _ in range(32):
        tick(mxu)
    assert state.read_mrf_bf16_tile(m(2)).float().tolist() == [[1] * 32] * 32


def test_overwrite_compute_may_follow_pop_on_same_accumulator(mxu):
    """AccumulationBuffers has one read port per buffer, but only an
    accumulate-mode compute reads it; an overwrite compute may start the
    cycle after a pop of the same accumulator (perf_mm_* baremetal tests)."""
    state = mxu.arch_state
    put_fp8(state, m(0), torch.ones(32, 32))
    put_fp8(state, m(1), torch.ones(32, 32))
    state.acc[mxu.mxu][0][:] = torch.full((32, 32), 3.0)
    tick(mxu, instruction(mxu, "VMATPUSH_WEIGHT", vd=w(0), vs1=m(1)))
    for _ in range(mxu.compute_first + 32):
        tick(mxu)
    tick(mxu, instruction(mxu, "VMATPOP_BF16_ACC", vd=m(2), vs2=acc(0)))
    tick(mxu, instruction(mxu, "VMATMUL", vd=acc(0), vs1=m(0), vs2=w(0)))
    assert len(mxu.in_flight) == 2
    for _ in range(mxu.compute_first + 32):
        tick(mxu)
    assert not mxu.has_in_flight
    assert state.acc[mxu.mxu][0].float().tolist() == [[32] * 32] * 32
    assert state.read_mrf_bf16_tile(m(2)).float().tolist() == [[3] * 32] * 32


def test_accumulate_compute_during_pop_on_same_accumulator_is_rejected(mxu):
    tick(mxu, instruction(mxu, "VMATPOP_BF16_ACC", vd=m(2), vs2=acc(0)))
    with pytest.raises(RuntimeError, match="same accumulator"):
        tick(mxu, instruction(mxu, "VMATMUL_ACC", vd=acc(0), vs1=m(0), vs2=w(0)))


def test_second_compute_issues_on_last_feed_cycle(mxu):
    state = mxu.arch_state
    put_fp8(state, m(0), torch.ones(32, 32))
    put_fp8(state, m(1), torch.full((32, 32), 2))
    state.write_wb_fp8(mxu.mxu, w(0), torch.eye(32).to(torch.float8_e4m3fn))
    tick(mxu, instruction(mxu, "VMATMUL", vd=acc(0), vs1=m(0), vs2=w(0)))
    for _ in range(31):
        tick(mxu)
    tick(mxu, instruction(mxu, "VMATMUL", vd=acc(1), vs1=m(1), vs2=w(0)))
    assert len(mxu.in_flight) == 2
    for _ in range(mxu.compute_first + 31):
        tick(mxu)
    assert state.acc[mxu.mxu][0].float().tolist() == [[1] * 32] * 32
    assert state.acc[mxu.mxu][1].float().tolist() == [[2] * 32] * 32


def test_compute_before_read_port_boundary_is_rejected(mxu):
    tick(mxu, instruction(mxu, "VMATMUL", vd=acc(0), vs1=m(0), vs2=w(0)))
    for _ in range(30):
        tick(mxu)
    with pytest.raises(RuntimeError, match="read port"):
        tick(mxu, instruction(mxu, "VMATMUL", vd=acc(1), vs1=m(1), vs2=w(1)))


def test_compute_and_weight_push_use_independent_read_ports(mxu):
    tick(mxu, instruction(mxu, "VMATMUL", vd=acc(0), vs1=m(0), vs2=w(0)))
    tick(mxu, instruction(mxu, "VMATPUSH_WEIGHT", vd=w(1), vs1=m(1)))
    assert len(mxu.in_flight) == 2
    for _ in range(mxu.compute_first + 32):
        tick(mxu)
    assert not mxu.has_in_flight


def test_same_accumulator_reuse_before_row_zero_is_rejected(mxu):
    tick(mxu, instruction(mxu, "VMATMUL", vd=acc(0), vs1=m(0), vs2=w(0)))
    for _ in range(mxu.compute_first - 1):
        tick(mxu)
    with pytest.raises(RuntimeError, match="row-0"):
        tick(mxu, instruction(mxu, "VMATPOP_BF16_ACC", vd=m(2), vs2=acc(0)))


def test_weight_push_then_compute_same_slot_obeys_engine_schedule(mxu):
    tick(mxu, instruction(mxu, "VMATPUSH_WEIGHT", vd=w(0), vs1=m(1)))
    with pytest.raises(RuntimeError, match="active push"):
        tick(mxu, instruction(mxu, "VMATMUL", vd=acc(0), vs1=m(0), vs2=w(0)))


def test_weight_push_after_compute_uses_engine_specific_drain(mxu):
    tick(mxu, instruction(mxu, "VMATMUL", vd=acc(0), vs1=m(0), vs2=w(0)))
    for _ in range(31):
        tick(mxu)
    tick(mxu, instruction(mxu, "VMATPUSH_WEIGHT", vd=w(0), vs1=m(1)))
    for _ in range(32):
        tick(mxu)
    assert not mxu.has_in_flight


def test_fp32_push_writes_columns_0_15_then_16_31(mxu):
    """Step s writes row s % 32 at T+1+s: columns 0-15 from m[r], m[r+1], then 16-31 from m[r+2], m[r+3]."""
    state = mxu.arch_state
    for q in range(4):
        state.write_mrf_f32(m(4 + q), torch.full((32, 8), float(q + 1)))
    tick(mxu, instruction(mxu, "VMATPUSH_ACC_FP32", vd=acc(0), vs1=m(4)))
    want = torch.tensor([float(j // 8 + 1) for j in range(32)])
    for step in range(64):
        tick(mxu)
        row, half = step % 32, slice(16 * (step // 32), 16 * (step // 32) + 16)
        assert torch.equal(state.acc[mxu.mxu][0][row, half], want[half])
        if step < 32:
            assert state.acc[mxu.mxu][0][:, 16:].count_nonzero() == 0
    assert torch.equal(state.acc[mxu.mxu][0], want.expand(32, 32))
    assert not mxu.has_in_flight


def test_fp32_pop_writes_m_r_r1_then_r2_r3(mxu):
    state = mxu.arch_state
    state.acc[mxu.mxu][1][:] = torch.arange(1024, dtype=torch.float32).reshape(32, 32)
    tick(mxu, instruction(mxu, "VMATPOP_FP32_ACC", vd=m(8), vs2=acc(1)))
    for _ in range(32):
        tick(mxu)
    assert [state.read_mrf_f32(m(8 + q)).count_nonzero().item() for q in range(4)] == [255, 256, 0, 0]
    for _ in range(32):
        tick(mxu)
    assert torch.equal(torch.cat([state.read_mrf_f32(m(8 + q)) for q in range(4)], 1), state.acc[mxu.mxu][1])
    assert not mxu.has_in_flight


def test_pop_waits_for_the_fp32_push_second_write_of_row_0(mxu):
    tick(mxu, instruction(mxu, "VMATPUSH_ACC_FP32", vd=acc(0), vs1=m(4)))
    for _ in range(32):
        tick(mxu)
    with pytest.raises(RuntimeError, match="after a push"):
        tick(mxu, instruction(mxu, "VMATPOP_BF16_ACC", vd=m(8), vs2=acc(0)))


def test_accumulator_writes_wait_for_the_fp32_pop_first_phase(mxu):
    tick(mxu, instruction(mxu, "VMATPOP_FP32_ACC", vd=m(8), vs2=acc(0)))
    for _ in range(31):
        tick(mxu)
    with pytest.raises(RuntimeError, match="first half of an FP32 pop"):
        tick(mxu, instruction(mxu, "VMATPUSH_ACC_BF16", vd=acc(0), vs1=m(2)))


def test_fp32_transfer_frees_its_first_pair_after_32_steps(mxu):
    checker = mxu.arch_state.conflict_checker
    tick(mxu, instruction(mxu, "VMATPUSH_ACC_FP32", vd=acc(0), vs1=m(4)))
    for _ in range(32):
        tick(mxu)
    checker.begin_cycle(mxu.cycle + 1)
    checker.reserve_mreg("writer", reads=frozenset(), writes=frozenset({4, 5}))
    with pytest.raises(BankConflictError):
        checker.reserve_mreg("writer", reads=frozenset(), writes=frozenset({6}))


def test_a_pop_of_one_row_holds_its_registers_until_after_its_write(mxu):
    """At M = 1 a pop writes row 0 in its last step, and the RTL holds a reader of its registers
    until the next cycle: no read of row 0 in the cycle that the pop writes it."""
    checker = mxu.arch_state.conflict_checker
    mxu.arch_state.rows = 1
    tick(mxu, instruction(mxu, "VMATPOP_BF16_ACC", vd=m(8), vs2=acc(0)))
    tick(mxu)
    with pytest.raises(BankConflictError):
        checker.reserve_mreg("reader", reads=frozenset({8}), writes=frozenset())
    checker.begin_cycle(mxu.cycle + 1)
    checker.reserve_mreg("reader", reads=frozenset({8, 9}), writes=frozenset())
