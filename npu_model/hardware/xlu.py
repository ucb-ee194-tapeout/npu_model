"""Independent XluEngine read-buffer-transpose-write state machine.

Values come from Instruction.exec; this unit decides only when rows move.
"""
from .exu import ExecutionUnit, StagedExecution
from ..software.instruction import Uop
from ..isa import EXU


class CrossLaneExecutionUnit(ExecutionUnit):
    def __init__(self, name, logger, arch_state, lane_id=0, config=None):
        super().__init__(name, logger, arch_state, lane_id, config)
        if arch_state.cfg.mrf_depth != 32 or arch_state.cfg.mrf_width != 32:
            raise ValueError("XluEngine requires a 32 by 32 byte MREG")
        self.reset()

    def reset(self) -> None:
        self.cycle = 0
        self.in_flight: Uop | None = None
        self.issued = 0
        self.owner = ""
        self.staged: StagedExecution | None = None
        self._pending_completions: list[Uop] = []
        self._complete_count = 0
        self._total_instructions = 0
        self._busy_cycles = 0

    def can_handle(self, uop: Uop) -> bool:
        return uop.insn.exu == EXU.XLU

    def abort(self) -> None:
        if self.owner:
            self.arch_state.conflict_checker.release_mreg(self.owner)
        self.in_flight = None
        self._pending_completions.clear()
        self._complete_count = 0

    def tick(self, uop: Uop | None) -> None:
        self.cycle += 1
        checker = self.arch_state.conflict_checker
        checker.begin_cycle(self.cycle)
        self.flush_completions()
        self._complete_count = 0
        if uop is not None:
            if self.in_flight is not None:
                raise RuntimeError("XLU command issued while transpose engine is busy")
            self.in_flight = uop
            self.issued = self.cycle
            self.owner = f"{self.name}:{uop.id}:vtrpose.xlu"
            source, destination = int(uop.insn.vs1), int(uop.insn.vd)
            checker.reserve_mreg(self.owner, frozenset({source}), frozenset({destination}))
            self.staged = StagedExecution(uop, self.arch_state, [("mrf", source)],
                                          [("mrf", destination)], "XluEngine")
            self._total_instructions += 1
            uop.execute_delay = 66
            self.logger.log_stage_start(uop.id, "E", lane=self.lane_id, cycle=self.cycle)
        if self.in_flight is None:
            return
        self._busy_cycles += 1
        age = self.cycle - self.issued
        insn = self.in_flight.insn
        if 1 <= age <= 32:
            row = age - 1
            checker.access_mreg(self.cycle, int(insn.vs1), row, False, self.owner)
            # Capture the value at the request edge; the response appears one
            # cycle later. The last response changes ReadMreg to WriteMreg.
            source = int(insn.vs1)
            self.staged.sample(("mrf", source), row, self.arch_state.read_mrf_u8(source)[row])
        if age == 33:
            checker.release_mreg(self.owner, reads=True, writes=False)
        if 34 <= age <= 65:
            row = age - 34
            destination = int(insn.vd)
            checker.access_mreg(self.cycle, destination, row, True, self.owner)
            self.arch_state.read_mrf_u8(destination)[row] = self.staged.result(("mrf", destination))[row]
        self.in_flight.execute_delay = max(0, 65 - age)
        if age == 65:
            checker.release_mreg(self.owner)
            self._pending_completions.append(self.in_flight)
            self.in_flight = None
            self._complete_count = 1

    def flush_completions(self) -> None:
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id, cycle=self.cycle)
            self.logger.log_retire(uop.id)
        self._pending_completions.clear()

    @property
    def has_in_flight(self) -> bool:
        return self.in_flight is not None

    @property
    def complete_count(self) -> int:
        return self._complete_count

    @property
    def total_instructions(self) -> int:
        return self._total_instructions

    @property
    def busy_cycles(self) -> int:
        return self._busy_cycles
