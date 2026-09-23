"""Edge-by-edge contracts from ScalarCore.scala and PcControl.scala.

Cycle 1 captures IMEM[0]; cycle 2 executes it. Register assertions below
observe the state after the indicated edge.
"""
from dataclasses import replace

import pytest

from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.configs.isa_definition import (
    ADDI, AUIPC, BEQ, BNE, BLT, BGE, JAL, JALR, DELAY, DMA_WAIT_CH0,
    ECALL, EBREAK, SLL, SLT, SRLI,
)
from npu_model.hardware.core import Core
from npu_model.logging import Logger, LoggerConfig
from npu_model.software import x
from npu_model.software.program import InstantiableProgram


@pytest.fixture
def make_core(tmp_path):
    cores = []
    def make(instructions):
        config = DefaultHardwareConfig()
        config.arch_state_config = replace(config.arch_state_config, dram_size=4096)
        logger = Logger(LoggerConfig(filename=str(tmp_path / f"trace{len(cores)}.json")))
        core = Core(config, logger)
        core.load_program(InstantiableProgram(instructions))
        cores.append(core)
        return core
    yield make
    for core in cores:
        core.close()
        core.logger.close()


def step(core, count=1):
    traces = []
    for _ in range(count):
        core.tick()
        traces.append(dict(core.last_cycle))
    return traces


def test_fetch_then_combined_decode_execute_writeback(make_core):
    core = make_core([ADDI(x(1), x(0), 7), ADDI(x(2), x(1), 3)])
    step(core)
    assert core.arch_state.xrf[1:3] == [0, 0]
    step(core)
    assert core.arch_state.xrf[1:3] == [7, 0]
    step(core)
    assert core.arch_state.xrf[1:3] == [7, 10]
    assert core.is_finished()
    assert core.total_completed == 2


def test_one_delay_slot_and_word_relative_target(make_core):
    core = make_core([
        BEQ(x(0), x(0), 8),  # decoded imm 8 -> target word 4
        ADDI(x(1), x(0), 11),
        ADDI(x(2), x(0), 22),
        ADDI(x(3), x(0), 33),
        ADDI(x(4), x(0), 44),
    ])
    trace = step(core, 4)
    assert [t['s1_pc'] for t in trace] == [None, 0, 1, 4]
    assert core.arch_state.xrf[1:5] == [11, 0, 0, 44]
    assert trace[1]['redirect']


def test_not_taken_branch_allows_immediately_following_branch(make_core):
    core = make_core([BNE(x(0), x(0), 4), JAL(x(1), 4), DELAY(0), ADDI(x(2), x(0), 9)])
    trace = step(core, 5)
    assert [t['s1_pc'] for t in trace] == [None, 0, 1, 2, 3]
    assert core.arch_state.read_xrf(1) == 2
    assert core.arch_state.read_xrf(2) == 9


def test_taken_branch_rejects_branch_in_only_delay_slot(make_core):
    core = make_core([BEQ(x(0), x(0), 6), JAL(x(0), 4), DELAY(0), DELAY(0)])
    step(core, 2)
    with pytest.raises(RuntimeError, match='delay-slot'):
        step(core)
    assert core.arch_state.halted


@pytest.mark.parametrize('delay', [0, 1, 4, 4095])
def test_delay_retires_then_stalls_following_instruction(make_core, delay):
    core = make_core([DELAY(delay), ADDI(x(1), x(0), 9)])
    step(core, 2)
    assert core.total_completed == 1
    assert core.idu.delay_counter == delay
    assert core.arch_state.pc == 2
    for _ in range(delay):
        step(core)
        assert core.arch_state.read_xrf(1) == 0
        assert core.arch_state.pc == 2
        assert core.last_cycle['stall'] == 'delay'
    step(core)
    assert core.arch_state.read_xrf(1) == 9
    assert core.cycle_count == delay + 3


def test_pc_relative_instruction_after_stall_uses_held_pc(make_core):
    core = make_core([DELAY(3), JAL(x(1), 6), ADDI(x(2), x(0), 2), ADDI(x(3), x(0), 3), AUIPC(x(4), 1)])
    trace = step(core, 8)
    assert [t['s1_pc'] for t in trace if t['s1_fire']] == [0, 1, 2, 4]
    assert core.arch_state.xrf[1:5] == [2, 2, 0, 4100]


def test_delay_in_branch_slot_holds_target(make_core):
    core = make_core([JAL(x(1), 6), DELAY(2), ADDI(x(2), x(0), 2), ADDI(x(3), x(0), 3)])
    trace = step(core, 6)
    assert [t['s1_pc'] for t in trace if t['s1_fire']] == [0, 1, 3]
    assert core.arch_state.read_xrf(2) == 0
    assert core.arch_state.read_xrf(3) == 3


def test_jalr_same_source_destination_uses_old_source_and_word_link(make_core):
    core = make_core([ADDI(x(1), x(0), 3), JALR(x(1), x(1), 1), DELAY(0), ADDI(x(2), x(0), 2), ADDI(x(3), x(0), 3)])
    trace = step(core, 5)
    assert [t['s1_pc'] for t in trace] == [None, 0, 1, 2, 4]
    assert core.arch_state.read_xrf(1) == 2
    assert core.arch_state.xrf[2:4] == [0, 3]


def test_dma_wait_holds_frontend_until_busy_is_clear(make_core):
    core = make_core([DMA_WAIT_CH0(), ADDI(x(1), x(0), 1)])
    core.arch_state.set_flag(0)
    trace = step(core, 4)
    assert all(t['stall'] == 'dma.wait' for t in trace[1:])
    assert core.arch_state.pc == 1
    core.arch_state.clear_flag(0)
    step(core)
    assert core.total_completed == 1
    assert core.arch_state.read_xrf(1) == 0
    step(core)
    assert core.arch_state.read_xrf(1) == 1


@pytest.mark.parametrize('halt', [ECALL, EBREAK])
def test_halt_detection_precedes_delay_stall_and_does_not_retire(make_core, halt):
    core = make_core([DELAY(10), halt(), ADDI(x(1), x(0), 1)])
    step(core, 3)
    assert core.arch_state.halted
    assert core.total_completed == 1
    assert core.arch_state.read_xrf(1) == 0
    assert core.cycle_count == 3


def test_scalar_rv32_wrap_signed_compare_and_shift_mask(make_core):
    core = make_core([ADDI(x(1), x(0), -1), SLT(x(2), x(1), x(0)), ADDI(x(3), x(0), 32), SLL(x(4), x(1), x(3)), SRLI(x(5), x(1), 1)])
    step(core, 6)
    assert core.arch_state.xrf[1:6] == [0xFFFFFFFF, 1, 32, 0xFFFFFFFF, 0x7FFFFFFF]


def test_csr_counters_read_pre_edge_and_writes_override_increment(make_core):
    from npu_model.configs.isa_definition import CSRRS, CSRRW
    core = make_core([
        CSRRS(x(1), x(0), 0xC00),  # reads 1; write of old value overrides increment
        CSRRS(x(2), x(0), 0xC01),  # reads one retired instruction, suppresses own increment
        CSRRW(x(0), x(1), 0xC10),
        CSRRS(x(3), x(0), 0xC10),
        ECALL(),
    ])
    step(core, 6)
    assert core.arch_state.xrf[1:4] == [1, 1, 1]
    assert core.arch_state.read_csrf(0xC01) == 3
    assert core.arch_state.read_csrf(0xC02) == 5  # halted / ecall
    assert core.arch_state.read_csrf(0xABC) == core.arch_state.read_csrf(0xC00)
