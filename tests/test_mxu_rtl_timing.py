"""Default-geometry RTL command, port, wavefront, and row-write schedules."""
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.configs import isa_definition as isa
from npu_model.hardware.arch_state import ArchState
from npu_model.hardware.mxu import MatrixExecutionUnitSystolic, MatrixExecutionUnitInner
from npu_model.hardware.stage_data import StageData
from npu_model.logging.logger import Logger
from npu_model.software import acc, m, w
from npu_model.software.instruction import Uop


@pytest.fixture(params=[0, 1], ids=["SA", "IPT"])
def mxu(request):
    cfg = DefaultHardwareConfig()
    cfg.arch_state_config = replace(cfg.arch_state_config, dram_size=4096, vmem_size=4096)
    state = ArchState(cfg.arch_state_config)
    cls = (MatrixExecutionUnitSystolic, MatrixExecutionUnitInner)[request.param]
    unit = cls(f"Matrix{request.param}", Mock(spec=Logger), state, config=cfg)
    yield unit
    state.close()


def instruction(unit, name, **kwargs):
    return getattr(isa, f"{name}_MXU{unit.mxu[-1]}")(**kwargs)


def tick(unit, insn=None):
    stage = StageData(None)
    if insn is not None:
        stage.prepare(Uop(insn))
    unit.tick(stage)
    assert not stage.is_valid(), "MXU commands consume their issue slot immediately"


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
    expected = torch.zeros(32, 32, dtype=torch.bfloat16)
    expected[:, 7] = 2
    assert torch.equal(state.acc[mxu.mxu][0], expected)


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
    state = mxu.arch_state
    put_fp8(state, m(0), torch.ones(32, 32))
    put_fp8(state, m(1), torch.ones(32, 32))
    tick(mxu, instruction(mxu, "VMATPUSH_WEIGHT", vd=w(0), vs1=m(1)))
    command = instruction(mxu, "VMATMUL", vd=acc(0), vs1=m(0), vs2=w(0))
    if mxu.mxu == "mxu1":
        with pytest.raises(RuntimeError, match="active push"):
            tick(mxu, command)
        return
    # SA's column-by-column weight push precedes the compute wavefront;
    # starting compute before the push finishes is legal and uses new weights.
    tick(mxu, command)
    for _ in range(mxu.compute_first + 31):
        tick(mxu)
    assert state.acc[mxu.mxu][0].float().tolist() == [[32] * 32] * 32


def test_weight_push_after_compute_uses_engine_specific_drain(mxu):
    tick(mxu, instruction(mxu, "VMATMUL", vd=acc(0), vs1=m(0), vs2=w(0)))
    allowed_age = 63 if mxu.mxu == "mxu0" else 32
    for _ in range(allowed_age - 1):
        tick(mxu)
    tick(mxu, instruction(mxu, "VMATPUSH_WEIGHT", vd=w(0), vs1=m(1)))
    for _ in range(32):
        tick(mxu)
    assert not mxu.has_in_flight
