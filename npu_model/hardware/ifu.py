from copy import deepcopy
from dataclasses import dataclass

from .hardware import Module
from .stage_data import StageData
from ..software.program import Program
from ..software.instruction import Instruction, Uop
from ..logging.logger import Logger, LaneType
from ..hardware.arch_state import ArchState


@dataclass(frozen=True)
class ImemRequest:
    """Host word request; ``instruction=None`` means TileLink Get.

    Addresses are physical byte addresses. Writes carry decoded instructions,
    since this simulator executes instruction objects rather than decoding
    arbitrary binary words. Both PutFullData and PutPartialData in the current
    RTL write the entire word; this interface therefore exposes word writes.
    """

    address: int
    instruction: Instruction | None = None
    source: int = 0
    size: int = 2


@dataclass(frozen=True)
class ImemResponse:
    is_get: bool
    data: int | None
    source: int
    size: int


@dataclass(frozen=True)
class ImemCycle:
    """Host outputs immediately before the edge advanced by ``tick``."""

    request_ready: bool
    response: ImemResponse | None


class InstructionMemory:
    """The single 128 KiB, 1R1W SRAM in ``InstrMem.scala``.

    ``tick`` advances one rising edge. Its returned host outputs describe the
    pre-edge handshake; ``fetch_data`` contains the synchronous read result
    after that edge. A Put makes D valid after one edge, a Get after two.
    Responses remain stable until accepted, with no new A request accepted on
    the edge consuming D. Fetch blocks host Get throughout execution, including
    frontend stalls; host Put uses the independent write port.

    Uninitialized words are represented by None. Same-address read/write is
    undefined by SyncReadMem and is rejected rather than given invented data.
    The instruction objects are the model's representation of 32-bit words.
    There are no alternate program banks or bank-switch operation.
    """

    BASE = 0x0002_0000
    WORDS = 32768
    SIZE = WORDS * 4

    def __init__(self) -> None:
        self.words: list[Instruction | None] = [None] * self.WORDS
        self.reset()

    def reset(self) -> None:
        # SyncReadMem contents survive reset; only control state is reset.
        self.fetch_data: Instruction | None = None
        self._read_data: Instruction | None = None
        self._state = "idle"
        self._request: ImemRequest | None = None
        self._response: ImemResponse | None = None

    def load_program(self, program: Program) -> None:
        if len(program) > self.WORDS:
            raise ValueError(
                f"Program has {len(program)} words; RTL IMEM holds {self.WORDS} words"
            )
        self.words[:] = [None] * self.WORDS
        self.words[:len(program)] = deepcopy(program.instructions)

    @property
    def response(self) -> ImemResponse | None:
        """Current D-channel response, valid until a ready edge consumes it."""
        return self._response if self._state == "response" else None

    def tick(
        self,
        *,
        fetch_active: bool,
        fetch_address: int = 0,
        request: ImemRequest | None = None,
        response_ready: bool = True,
    ) -> ImemCycle:
        blocked_get = (
            fetch_active and request is not None and request.instruction is None
        )
        request_ready = self._state == "idle" and not blocked_get
        outputs = ImemCycle(request_ready, self.response)
        accepted = request if request_ready else None
        word_address = None
        if accepted is not None:
            if not self.BASE <= accepted.address < self.BASE + self.SIZE:
                raise ValueError("Host IMEM address is outside the RTL IMEM window")
            word_address = (accepted.address - self.BASE) // 4

        read_address = None
        if fetch_active:
            # ScalarCore connects only the low IMEM_ADDR_BITS of its word PC.
            read_address = fetch_address & (self.WORDS - 1)
        elif accepted is not None and accepted.instruction is None:
            read_address = word_address

        if (
            accepted is not None
            and accepted.instruction is not None
            and read_address == word_address
        ):
            raise RuntimeError("Undefined IMEM read/write collision at the same word")

        next_read_data = self.words[read_address] if read_address is not None else None
        if self._state == "idle" and accepted is not None:
            self._request = accepted
            if accepted.instruction is None:
                self._state = "read_response"
            else:
                assert word_address is not None
                self.words[word_address] = deepcopy(accepted.instruction)
                self._response = ImemResponse(False, 0, accepted.source, accepted.size)
                self._state = "response"
        elif self._state == "read_response":
            assert self._request is not None
            data = self._read_data.to_bytecode() if self._read_data is not None else None
            self._response = ImemResponse(
                True, data, self._request.source, self._request.size
            )
            self._state = "response"
        elif self._state == "response" and response_ready:
            self._state = "idle"
            self._request = None
            self._response = None

        self._read_data = next_read_data
        self.fetch_data = next_read_data
        return outputs


class InstructionFetch(Module):
    """One synchronous IMEM fetch stage feeding RTL decode/execute.

    ``output`` is the instruction currently in stage 1. During a frontend
    stall it models ScalarCore's instruction hold register, while the SRAM
    continues reading the held fetch address. PCs use RTL word indices.
    """

    def __init__(
        self,
        width: int,
        logger: Logger,
        arch_state: ArchState,
    ) -> None:
        if width != 1:
            raise ValueError("RTL IMEM fetch width is exactly one instruction")
        self.width = width
        self.logger = logger
        self.arch_state = arch_state
        self.program: Program | None = None
        self.memory = InstructionMemory()
        self.reset()

    def load_program(self, program: Program) -> None:
        self.memory.load_program(program)
        self.program = program

    def reset(self) -> None:
        self.output: StageData[Uop | None] = StageData(None)
        self.arch_state.set_pc(0)
        self.memory.reset()
        self.cycle = 0
        self._stalled = False

    def is_finished(self) -> bool:
        return (
            self.program is not None
            and self.memory.words[self.arch_state.pc & (self.memory.WORDS - 1)] is None
            and not self.output.is_valid()
        )

    def tick(
        self,
        *,
        stalled: bool = False,
        host_request: ImemRequest | None = None,
        host_response_ready: bool = True,
    ) -> ImemCycle:
        if self.program is None:
            raise RuntimeError("Attempted to tick while no program is loaded.")

        self.cycle += 1
        pc = self.arch_state.pc
        bus = self.memory.tick(
            fetch_active=not self.arch_state.halted,
            fetch_address=pc,
            request=host_request,
            response_ready=host_response_ready,
        )
        if self.arch_state.halted:
            self.output.reset()
            self._stalled = False
            return bus

        if stalled or self.output.should_stall():
            if not self._stalled:
                uop = self.output.peek()
                if uop is not None:
                    self.logger.log_stage_end(
                        uop.id, "F", lane=LaneType.IFU.value, cycle=self.cycle
                    )
            self._stalled = True
            return bus

        self._stalled = False
        fetched_instruction = self.memory.fetch_data
        # Program exhaustion is a simulator convenience (hardware requires a
        # halt instruction). Still apply redirects at the end of a program.
        if fetched_instruction is None:
            self.output.prepare(None)
        else:
            uop = Uop(fetched_instruction, pc=pc)
            self.logger.log_insn(uop.id, str(uop.insn))
            self.logger.log_stage_start(
                uop.id, "F", lane=LaneType.IFU.value, cycle=self.cycle
            )
            self.output.prepare(uop)

        self.arch_state.set_pc(self.arch_state.npc)
        return bus

    @property
    def is_stalled(self) -> bool:
        return self._stalled

    def force_unstall(self) -> None:
        self._stalled = False
