from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from npu_model.configs.isa_definition import ADDI
from npu_model.hardware.ifu import ImemRequest, InstructionFetch, InstructionMemory
from npu_model.logging.logger import Logger
from npu_model.software import x
from npu_model.software.program import InstantiableProgram


def program(*values: int) -> InstantiableProgram:
    return InstantiableProgram([ADDI(rd=x(1), rs1=x(0), imm=v) for v in values])


def fetch_unit(*values: int):
    state = SimpleNamespace(pc=0, npc=1, halted=False)
    state.set_pc = lambda pc: setattr(state, "pc", pc)
    ifu = InstructionFetch(1, Mock(spec=Logger), state)
    ifu.load_program(program(*values))
    return ifu, state


def test_program_counter_indexes_words() -> None:
    image = program(10, 20, 30)
    assert image.get_instruction(1).imm == 20
    assert not image.is_finished(2)
    assert image.is_finished(3)


def test_single_bank_capacity_and_program_snapshot() -> None:
    memory = InstructionMemory()
    image = program(10, 20)
    memory.load_program(image)
    image.instructions[0].imm = 99
    memory.tick(fetch_active=True, fetch_address=0)
    assert memory.fetch_data.imm == 10
    with pytest.raises(ValueError, match="RTL IMEM holds 32768 words"):
        memory.load_program(InstantiableProgram([image[0]] * 32769))


def test_synchronous_fetch_and_address_truncation() -> None:
    memory = InstructionMemory()
    memory.load_program(program(10, 20))
    assert memory.fetch_data is None
    memory.tick(fetch_active=True, fetch_address=0)
    assert memory.fetch_data.imm == 10
    memory.tick(fetch_active=True, fetch_address=InstructionMemory.WORDS + 1)
    assert memory.fetch_data.imm == 20


def test_host_get_is_blocked_while_fetch_owns_read_port() -> None:
    memory = InstructionMemory()
    memory.load_program(program(10, 20))
    request = ImemRequest(memory.BASE + 4)
    for _ in range(3):
        cycle = memory.tick(fetch_active=True, fetch_address=0, request=request)
        assert not cycle.request_ready
        assert cycle.response is None
        assert memory.fetch_data.imm == 10
    cycle = memory.tick(fetch_active=False, request=request)
    assert cycle.request_ready
    assert memory.response is None
    memory.tick(fetch_active=False)
    assert memory.response.is_get
    assert memory.response.data == memory.words[1].to_bytecode()


def test_get_response_is_buffered_and_stable_during_backpressure() -> None:
    memory = InstructionMemory()
    memory.load_program(program(10, 20))
    request = ImemRequest(memory.BASE, source=7, size=2)
    # E1 accepts A and performs the SRAM read; E2 captures SRAM data in D.
    memory.tick(fetch_active=False, request=request, response_ready=False)
    assert memory.response is None
    cycle = memory.tick(fetch_active=True, fetch_address=1, response_ready=False)
    assert not cycle.request_ready
    expected = memory.response
    assert expected.source == 7
    assert expected.size == 2
    assert expected.data == memory.words[0].to_bytecode()
    for _ in range(3):
        cycle = memory.tick(
            fetch_active=True, fetch_address=1, response_ready=False
        )
        assert cycle.response == expected
        assert memory.response == expected
        assert memory.fetch_data.imm == 20
    cycle = memory.tick(fetch_active=False, request=request)
    assert cycle.response == expected
    assert not cycle.request_ready  # Cannot accept A on the D handshake edge.
    assert memory.response is None
    assert memory.tick(fetch_active=False, request=request).request_ready


def test_host_put_updates_live_bank_without_interrupting_other_fetch() -> None:
    memory = InstructionMemory()
    memory.load_program(program(10, 20))
    replacement = program(99)[0]
    request = ImemRequest(memory.BASE + 4, replacement, source=3)
    cycle = memory.tick(fetch_active=True, fetch_address=0, request=request)
    assert cycle.request_ready
    assert memory.fetch_data.imm == 10
    assert memory.response is not None  # Put response after its acceptance edge.
    assert not memory.response.is_get
    assert memory.response.source == 3
    assert memory.response.data == 0
    memory.tick(fetch_active=True, fetch_address=1)
    assert memory.fetch_data.imm == 99  # No bank-swap command or extra read stage.


def test_same_word_read_write_is_explicitly_undefined() -> None:
    memory = InstructionMemory()
    memory.load_program(program(10))
    with pytest.raises(RuntimeError, match="Undefined IMEM read/write collision"):
        memory.tick(
            fetch_active=True,
            fetch_address=0,
            request=ImemRequest(memory.BASE, program(99)[0]),
        )


def test_reset_preserves_sram_contents_and_clears_response_state() -> None:
    memory = InstructionMemory()
    memory.load_program(program(10))
    memory.tick(fetch_active=False, request=ImemRequest(memory.BASE, program(99)[0]))
    assert memory.response is not None
    memory.reset()
    assert memory.response is None
    assert memory.fetch_data is None
    memory.tick(fetch_active=True, fetch_address=0)
    assert memory.fetch_data.imm == 99


def test_ifu_has_one_fetch_register_and_holds_instruction_and_pc_on_stall() -> None:
    ifu, state = fetch_unit(10, 20, 30)
    ifu.tick()
    first = ifu.output.peek()
    assert first.pc == 0
    assert first.insn.imm == 10
    assert state.pc == 1
    for _ in range(3):
        state.npc = 2
        ifu.tick(stalled=True)
        assert ifu.output.peek() is first
        assert ifu.memory.fetch_data.imm == 20  # SRAM keeps reading while held.
        assert state.pc == 1
    assert ifu.output.claim() is first
    ifu.tick()
    assert ifu.output.peek().pc == 1
    assert ifu.output.peek().insn.imm == 20
    assert state.pc == 2


def test_host_get_stays_blocked_during_frontend_stall_then_runs_when_halted() -> None:
    ifu, state = fetch_unit(10, 20)
    ifu.tick()
    request = ImemRequest(ifu.memory.BASE)
    assert not ifu.tick(stalled=True, host_request=request).request_ready
    state.halted = True
    assert ifu.tick(host_request=request).request_ready
    assert not ifu.output.is_valid()
    assert state.pc == 1
    ifu.tick()
    assert ifu.memory.response.data == ifu.memory.words[0].to_bytecode()


def test_fetches_one_delay_slot_before_redirect_target() -> None:
    ifu, state = fetch_unit(10, 20, 30, 40)
    ifu.tick()  # Fetch branch at word zero.
    assert ifu.output.claim().pc == 0
    state.npc = 3  # Branch resolves while word one is being fetched.
    ifu.tick()
    assert ifu.output.claim().pc == 1
    assert state.pc == 3
    state.npc = 4
    ifu.tick()
    assert ifu.output.claim().pc == 3


def test_redirect_at_program_end_is_not_lost() -> None:
    ifu, state = fetch_unit(10)
    ifu.tick()
    ifu.output.claim()
    assert state.pc == 1
    state.npc = 0
    ifu.tick()
    assert state.pc == 0
    assert not ifu.is_finished()
    state.npc = 1
    ifu.tick()
    assert ifu.output.peek().pc == 0


def test_ifu_rejects_non_rtl_fetch_width() -> None:
    with pytest.raises(ValueError, match="exactly one instruction"):
        InstructionFetch(2, Mock(spec=Logger), Mock())
