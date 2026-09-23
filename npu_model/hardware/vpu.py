"""VectorFSM row schedules and its two software-scheduled issue slots.

Timing follows the registered MREG read and lane-box valid pipelines. Numerical
transcendentals still use PyTorch, rather than the RTL approximation tables.
"""
from dataclasses import dataclass, field

import torch

from .exu import ExecutionUnit
from ..logging.logger import LaneType
from ..software.instruction import Uop
from ..isa import EXU
from .stage_data import StageData

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


def _truncated_bf16(value: torch.Tensor) -> torch.Tensor:
    """AddSubSumVec/ColAddVec take the upper 16 bits of FP32 results."""
    return (value.float().contiguous().view(torch.int32) >> 16).to(torch.int16).view(torch.bfloat16)


def pack_row(value: torch.Tensor, scale: int) -> torch.Tensor:
    """FP8Pack's BF16/E8M0 conversion, including flush and saturation."""
    output = []
    shift = min(127, max(-128, int(scale) - 127))
    for raw in value.contiguous().view(torch.int16).tolist():
        raw &= 0xFFFF
        sign, exp, mant = raw >> 15, (raw >> 7) & 255, raw & 127
        if exp == 0 or (exp == 255 and mant):
            out = 0
        elif exp == 255:
            out = (sign << 7) | 0x7E
        else:
            adjusted = exp - 127 - shift
            sig = 128 | mant
            rounded = (sig >> 4) + int(bool((sig & 8) and ((sig & 7) or ((sig >> 4) & 1))))
            if rounded == 16:
                adjusted += 1
                rounded = 8
            frac = (rounded - 8) & 7
            if adjusted > 8 or (adjusted == 8 and frac == 7):
                out = (sign << 7) | 0x7E
            elif adjusted < -6:
                out = 0
            else:
                out = (sign << 7) | ((adjusted + 7) << 3) | frac
        output.append(out)
    return torch.tensor(output, dtype=torch.uint8)


def unpack_row(value: torch.Tensor, scale: int) -> torch.Tensor:
    """FP8Unpack's E4M3/E8M0 conversion, preserving signed flushed zero."""
    output = []
    shift = min(127, max(-128, int(scale) - 127))
    for raw in value.tolist():
        sign, exp, mant = raw >> 7, (raw >> 3) & 15, raw & 7
        adjusted = exp - 7 + shift + 127
        if exp == 0 or (exp == 15 and mant == 7) or adjusted <= 0:
            out = sign << 15
        elif adjusted >= 255:
            out = (sign << 15) | 0x7F7F
        else:
            out = (sign << 15) | (adjusted << 7) | (mant << 4)
        output.append(out)
    return torch.tensor(output, dtype=torch.uint16).view(torch.bfloat16)


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
    scale: int = 127
    pending: dict[int, list[tuple[int, int, torch.Tensor]]] = field(default_factory=dict)
    packed_low: torch.Tensor | None = None
    reduction: torch.Tensor | None = None


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

    @staticmethod
    def _double(mnemonic: str) -> bool:
        return mnemonic in _TWO_INPUT | _ROW_REDUCE

    @staticmethod
    def _share_logic(left: str, right: str) -> bool:
        return left == right or any(left in group and right in group for group in _GROUPS)

    def can_handle(self, uop: Uop) -> bool:
        return uop.insn.exu == EXU.VECTOR

    def _execution_latency(self, uop: Uop) -> int:
        return VPU_OP_LATENCIES[uop.insn.mnemonic]

    def _accept(self, uop: Uop) -> None:
        insn = uop.insn
        name = insn.mnemonic
        # VectorFSM's done includes the final write in this cycle.
        live = [op for op in self.operations if self.cycle - op.issued < op.write_last]
        if live and (len(live) == 2 or self._double(name)
                     or any(self._double(op.uop.insn.mnemonic)
                            or self._share_logic(name, op.uop.insn.mnemonic) for op in live)):
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
        scale = self.arch_state.read_erf(insn.es1) if hasattr(insn, "es1") else 127
        latency = self._execution_latency(uop)
        op = _VectorOperation(uop, self.cycle, slot, owner, frozenset(reads), writes,
                              read_last, latency - 1, scale)
        self.operations.append(op)
        uop.execute_delay = latency
        self._total_instructions += 1
        self.logger.log_stage_end(uop.id, "D", lane=LaneType.DIU.value, cycle=self.cycle)
        self.logger.log_stage_start(uop.id, "E", lane=self.lane_id, cycle=self.cycle)

    def _read(self, op: _VectorOperation, bank: int, row: int) -> torch.Tensor:
        key = (bank, row)
        if key not in self._read_cache:
            self.arch_state.conflict_checker.access_mreg(self.cycle, bank, row, False, op.owner)
            self._read_cache[key] = self.arch_state.mrf[bank][row * 32:(row + 1) * 32].clone()
        return self._read_cache[key]

    def _queue(self, op: _VectorOperation, age: int, index: int, data: torch.Tensor,
               *, bank: int | None = None) -> None:
        if bank is None:
            bank = int(op.uop.insn.vd) + index // 32
        op.pending.setdefault(age, []).append((bank, index % 32, data.contiguous().view(torch.uint8).clone()))

    def _elementwise(self, name: str, a: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
        if name == "vadd.bf16":
            return _truncated_bf16(a.float() + b.float())
        if name == "vsub.bf16":
            return _truncated_bf16(a.float() - b.float())
        if name == "vmul.bf16":
            return a * b
        if name == "vminimum.bf16":
            return torch.minimum(a, b)
        if name == "vmaximum.bf16":
            return torch.maximum(a, b)
        functions = {
            "vmov": lambda x: x,
            "vrecip.bf16": torch.reciprocal, "vexp.bf16": torch.exp,
            "vexp2.bf16": torch.exp2, "vrelu.bf16": torch.relu,
            "vsin.bf16": torch.sin, "vcos.bf16": torch.cos,
            "vtanh.bf16": torch.tanh, "vlog2.bf16": torch.log2,
            "vsqrt.bf16": torch.sqrt, "vsquare.bf16": lambda x: x * x,
            "vcube.bf16": lambda x: x * x * x,
        }
        return functions[name](a).to(torch.bfloat16)

    def _advance(self, op: _VectorOperation) -> None:
        age = self.cycle - op.issued
        insn = op.uop.insn
        name = insn.mnemonic
        if name.startswith("vli.") and 1 <= age <= op.write_last:
            index = age - 1
            raw = torch.zeros(16, dtype=torch.uint16)
            if name == "vli.all" or (name == "vli.row" and index % 32 == 0):
                raw.fill_(int(insn.imm) & 0xFFFF)
            elif name == "vli.col" or (name == "vli.one" and index == 0):
                raw[0] = int(insn.imm) & 0xFFFF
            self._queue(op, age, index, raw)
        elif 0 <= age <= op.read_last:
            if name in _ROW_REDUCE:
                lo = self._read(op, int(insn.vs1), age).view(torch.bfloat16)
                hi = self._read(op, int(insn.vs1) + 1, age).view(torch.bfloat16)
                values = torch.cat((lo, hi)).float()
                if name == "vredsum.row.bf16":
                    # Balanced FP32 tree mirrors ReduSumRec's five add stages.
                    while values.numel() > 1:
                        values = values[::2] + values[1::2]
                    value, latency = values[0].to(torch.bfloat16), 7
                elif name == "vredmin.row.bf16":
                    value, latency = values.min().to(torch.bfloat16), 2
                else:
                    value, latency = values.max().to(torch.bfloat16), 2
                result = value.expand(16).contiguous()
                self._queue(op, age + latency, age, result, bank=int(insn.vd))
                self._queue(op, age + latency, age, result, bank=int(insn.vd) + 1)
            elif name == "vunpack.fp8.bf16":
                raw = self._read(op, int(insn.vs2), age)
                values = unpack_row(raw, op.scale)
                self._queue(op, 3 + 2 * age, 2 * age, values[:16])
                self._queue(op, 4 + 2 * age, 2 * age + 1, values[16:])
            else:
                source = int(insn.vs2 if name == "vpack.bf16.fp8" else insn.vs1)
                # Column reductions keep reading for 128 cycles, wrapping the pair.
                a = self._read(op, source + ((age // 32) & 1), age % 32).view(torch.bfloat16)
                if name in _COL_REDUCE:
                    if age < 64:
                        if op.reduction is None:
                            op.reduction = a.float().clone()
                        elif name == "vredsum.bf16":
                            op.reduction += a.float()
                        elif name == "vredmin.bf16":
                            op.reduction = torch.minimum(op.reduction, a.float())
                        else:
                            op.reduction = torch.maximum(op.reduction, a.float())
                elif name == "vpack.bf16.fp8":
                    packed = pack_row(a, op.scale)
                    if age % 2 == 0:
                        op.packed_low = packed
                    else:
                        self._queue(op, age + 2, age // 2, torch.cat((op.packed_low, packed)))
                else:
                    b = None
                    if name in _TWO_INPUT:
                        b = self._read(op, int(insn.vs2) + age // 32, age % 32).view(torch.bfloat16)
                    self._queue(op, age + 2, age, self._elementwise(name, a, b))
            if age == op.read_last:
                self.arch_state.conflict_checker.release_mreg(op.owner, reads=True, writes=False)
        if name in _COL_REDUCE and 66 <= age <= 129:
            result = _truncated_bf16(op.reduction) if name == "vredsum.bf16" else op.reduction.to(torch.bfloat16)
            self._queue(op, age, age - 66, result)
        for bank, row, data in op.pending.pop(age, []):
            self.arch_state.conflict_checker.access_mreg(self.cycle, bank, row, True, op.owner)
            self.arch_state.mrf[bank][row * 32:(row + 1) * 32] = data
        op.uop.execute_delay = max(0, op.write_last - age)
        if age == op.write_last:
            self.arch_state.conflict_checker.release_mreg(op.owner)
            self._pending_completions.append(op.uop)
            self._complete_count += 1

    def tick(self, idu_output: StageData[Uop | None]) -> None:
        self.cycle += 1
        self.arch_state.conflict_checker.begin_cycle(self.cycle)
        self.flush_completions()
        self._complete_count = 0
        self._read_cache = {}
        uop = idu_output.claim()
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
