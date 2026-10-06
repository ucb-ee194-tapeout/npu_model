"""VectorFSM row schedules and its two software-scheduled issue slots.

Timing follows the registered MREG read and lane-box valid pipelines. Values
come from Instruction.exec; this unit decides only which rows are read and
written on which cycle.
"""
from dataclasses import dataclass

import torch

from .exu import ExecutionUnit, StagedExecution
from ..software.instruction import Uop
from ..isa import EXU

_TWO_INPUT = {"vadd.bf16", "vsub.bf16", "vmul.bf16", "vminimum.bf16", "vmaximum.bf16"}
_ROW_REDUCE = {"vredsum.row.bf16", "vredmin.row.bf16", "vredmax.row.bf16"}
_COL_REDUCE = {"vredsum.bf16", "vredmin.bf16", "vredmax.bf16"}
_GROUPS = (
    {"vadd.bf16", "vsub.bf16", "vredsum.row.bf16"},
    {"vexp.bf16", "vexp2.bf16"},
    {"vsin.bf16", "vcos.bf16"},
    {"vsquare.bf16", "vcube.bf16"},
    {"vmaximum.bf16", "vredmax.bf16"},
    {"vminimum.bf16", "vredmin.bf16"},
    {"vli.all", "vli.row", "vli.col", "vli.one"},
)

# ScalarISA numbering (zero is VPU_NONE; sixteen is reserved FP8).
_OP_ORDER = (
    "vadd.bf16", "vsub.bf16", "vmul.bf16", "vrecip.bf16", "vsqrt.bf16",
    "vsin.bf16", "vcos.bf16", "vtanh.bf16", "vlog2.bf16", "vexp.bf16",
    "vexp2.bf16", "vsquare.bf16", "vcube.bf16", "vredsum.row.bf16",
    "vredsum.bf16", "reserved.fp8", "vpack.bf16.fp8", "vunpack.fp8.bf16",
    "vrelu.bf16", "vredmax.row.bf16", "vredmin.row.bf16", "vredmax.bf16",
    "vredmin.bf16", "vmaximum.bf16", "vminimum.bf16", "vmov",
    "vli.one", "vli.col", "vli.row", "vli.all",
)

# Inclusive issue-to-last-write cycles for the default 32-row geometry.
VPU_OP_LATENCIES = {
    **dict.fromkeys(_TWO_INPUT | _COL_REDUCE | {
        "vmov", "vrecip.bf16", "vexp.bf16", "vexp2.bf16", "vpack.bf16.fp8",
        "vrelu.bf16", "vsin.bf16", "vcos.bf16", "vtanh.bf16", "vlog2.bf16",
        "vsqrt.bf16", "vsquare.bf16", "vcube.bf16",
    }, 66),
    **dict.fromkeys(_COL_REDUCE, 130),
    "vredsum.row.bf16": 39,
    "vredmin.row.bf16": 34,
    "vredmax.row.bf16": 34,
    "vli.all": 65,
    "vli.row": 65,
    "vli.col": 33,
    "vli.one": 33,
    "vunpack.fp8.bf16": 67,
}


@dataclass
class _VectorOperation:
    uop: Uop
    issued: int
    slot: int
    owner: str
    reads: frozenset[int]
    writes: frozenset[int]
    read_last: int
    write_last: int
    staged: StagedExecution


class VectorExecutionUnit(ExecutionUnit):
    """Two single-input slots, or one instruction using both read ports."""

    def __init__(self, name, logger, arch_state, lane_id=0, config=None):
        super().__init__(name, logger, arch_state, lane_id, config)
        if arch_state.cfg.mrf_depth != 32 or arch_state.cfg.mrf_width != 32:
            raise ValueError("VectorFSM requires 32 rows of 32 bytes per MREG")
        self.reset()

    def reset(self) -> None:
        self.cycle = 0
        self.operations: list[_VectorOperation] = []
        self._pending_completions: list[Uop] = []
        self._complete_count = 0
        self._total_instructions = 0
        self._busy_cycles = 0
        self._read_cache: dict[tuple[int, int], torch.Tensor] = {}

    @property
    def in_flight(self) -> Uop | None:
        return self.operations[0].uop if self.operations else None

    def abort(self) -> None:
        for op in self.operations:
            self.arch_state.conflict_checker.release_mreg(op.owner)
        self.operations.clear()
        self._pending_completions.clear()
        self._complete_count = 0

    @staticmethod
    def _double(mnemonic: str) -> bool:
        return mnemonic in _TWO_INPUT | _ROW_REDUCE

    @staticmethod
    def _share_logic(left: str, right: str) -> bool:
        return left == right or any(left in group and right in group for group in _GROUPS)

    def _issue_blocked(self, name: str, cycle: int) -> bool:
        live = [op for op in self.operations if cycle - op.issued < op.write_last]
        return bool(live and (len(live) == 2 or self._double(name)
                    or any(self._double(op.uop.insn.mnemonic)
                           or self._share_logic(name, op.uop.insn.mnemonic) for op in live)))

    @property
    def issue_busy_mask(self) -> int:
        """RTL issueBusy for the next tick, before its command is accepted."""
        return sum(1 << index for index, name in enumerate(_OP_ORDER, 1)
                   if self._issue_blocked(name, self.cycle + 1))

    def can_handle(self, uop: Uop) -> bool:
        return uop.insn.exu == EXU.VECTOR

    def _execution_latency(self, uop: Uop) -> int:
        return VPU_OP_LATENCIES[uop.insn.mnemonic]

    def _accept(self, uop: Uop) -> None:
        insn = uop.insn
        name = insn.mnemonic
        # VectorFSM's done includes the final write in this cycle.
        live = [op for op in self.operations if self.cycle - op.issued < op.write_last]
        if self._issue_blocked(name, self.cycle):
            raise RuntimeError(f"VPU command {name} issued while its issue-busy bit is set")
        slot = next(index for index in (0, 1) if all(op.slot != index for op in live))
        vli = name.startswith("vli.")
        pair_write = name not in {"vpack.bf16.fp8", "vli.col", "vli.one"}
        if pair_write and int(insn.vd) & 1:
            raise RuntimeError("VPU pair-write destination bank must be even")
        writes = frozenset({int(insn.vd), int(insn.vd) + 1} if pair_write else {int(insn.vd)})
        reads = set()
        if not vli:
            source = int(insn.vs2 if name in {"vpack.bf16.fp8", "vunpack.fp8.bf16"} else insn.vs1)
            reads.add(source)
            if name != "vunpack.fp8.bf16":
                if source & 1:
                    raise RuntimeError("VPU pair-read primary bank must be even")
                reads.add(source + 1)
            if name in _TWO_INPUT:
                source2 = int(insn.vs2)
                if source2 & 1:
                    raise RuntimeError("VPU pair-read secondary bank must be even")
                reads.update({source2, source2 + 1})
        read_last = -1 if vli else (31 if name in _ROW_REDUCE | {"vunpack.fp8.bf16"}
                                    else 127 if name in _COL_REDUCE else 63)
        owner = f"{self.name}:{uop.id}:{name}"
        self.arch_state.conflict_checker.reserve_mreg(owner, frozenset(reads), writes)
        latency = self._execution_latency(uop)
        staged = StagedExecution(uop, self.arch_state, [("mrf", bank) for bank in reads],
                                 [("mrf", bank) for bank in writes], "VectorFSM")
        op = _VectorOperation(uop, self.cycle, slot, owner, frozenset(reads), writes,
                              read_last, latency - 1, staged)
        self.operations.append(op)
        uop.execute_delay = latency
        self._total_instructions += 1
        self.logger.log_stage_start(uop.id, "E", lane=self.lane_id, cycle=self.cycle)

    def _read(self, op: _VectorOperation, bank: int, row: int, *, sample: bool) -> None:
        key = (bank, row)
        if key not in self._read_cache:
            self.arch_state.conflict_checker.access_mreg(self.cycle, bank, row, False, op.owner)
            self._read_cache[key] = self.arch_state.mrf[bank][row * 32:(row + 1) * 32].clone()
        if sample:
            op.staged.sample(("mrf", bank), row, self._read_cache[key])

    @staticmethod
    def _reads(op: _VectorOperation, age: int) -> list[tuple[int, int]]:
        """(bank, row) requests VectorFSM issues at this age."""
        if not 0 <= age <= op.read_last:
            return []
        insn, name = op.uop.insn, op.uop.insn.mnemonic
        if name in _ROW_REDUCE:
            return [(int(insn.vs1), age), (int(insn.vs1) + 1, age)]
        if name == "vunpack.fp8.bf16":
            return [(int(insn.vs2), age)]
        # Column reductions keep reading for 128 cycles, wrapping the pair.
        source = int(insn.vs2 if name == "vpack.bf16.fp8" else insn.vs1)
        reads = [(source + ((age // 32) & 1), age % 32)]
        if name in _TWO_INPUT:
            reads.append((int(insn.vs2) + age // 32, age % 32))
        return reads

    @staticmethod
    def _writes(op: _VectorOperation, age: int) -> list[tuple[int, int]]:
        """(bank, row) results committed at this age."""
        name, vd = op.uop.insn.mnemonic, int(op.uop.insn.vd)
        if name in _ROW_REDUCE:
            # Both halves of row r land together, after the reduction tree.
            row = age - (7 if name == "vredsum.row.bf16" else 2)
            return [(vd, row), (vd + 1, row)] if 0 <= row < 32 else []
        if name == "vpack.bf16.fp8":
            # FP8 row k packs BF16 rows 2k and 2k + 1, two cycles after the second.
            return [(vd, (age - 3) // 2)] if 3 <= age <= 65 and age % 2 else []
        # The rest write one row per cycle across the pair, starting at `first`.
        first = (1 if name.startswith("vli.") else 3 if name == "vunpack.fp8.bf16"
                 else 66 if name in _COL_REDUCE else 2)
        index = age - first
        count = 32 if name in {"vli.col", "vli.one"} else 64
        return [(vd + index // 32, index % 32)] if 0 <= index < count else []

    def _advance(self, op: _VectorOperation) -> None:
        age = self.cycle - op.issued
        for bank, row in self._reads(op, age):
            # Only the first pass over the pair feeds the result.
            self._read(op, bank, row, sample=age < 64)
        if age == op.read_last:
            self.arch_state.conflict_checker.release_mreg(op.owner, reads=True, writes=False)
        for bank, row in self._writes(op, age):
            self.arch_state.conflict_checker.access_mreg(self.cycle, bank, row, True, op.owner)
            self.arch_state.read_mrf_u8(bank)[row] = op.staged.result(("mrf", bank))[row]
        op.uop.execute_delay = max(0, op.write_last - age)
        if age == op.write_last:
            self.arch_state.conflict_checker.release_mreg(op.owner)
            self._pending_completions.append(op.uop)
            self._complete_count += 1

    def tick(self, uop: Uop | None) -> None:
        self.cycle += 1
        self.arch_state.conflict_checker.begin_cycle(self.cycle)
        self.flush_completions()
        self._complete_count = 0
        self._read_cache = {}
        if uop is not None:
            self._accept(uop)
        if self.operations:
            self._busy_cycles += 1
        for op in self.operations:
            self._advance(op)
        self.operations = [op for op in self.operations if self.cycle - op.issued < op.write_last]

    def flush_completions(self) -> None:
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id, cycle=self.cycle)
            self.logger.log_retire(uop.id)
        self._pending_completions.clear()

    def is_busy(self) -> bool:
        return bool(self.operations)

    @property
    def has_in_flight(self) -> bool:
        return bool(self.operations)

    @property
    def complete_count(self) -> int:
        return self._complete_count

    @property
    def total_instructions(self) -> int:
        return self._total_instructions

    @property
    def busy_cycles(self) -> int:
        return self._busy_cycles
