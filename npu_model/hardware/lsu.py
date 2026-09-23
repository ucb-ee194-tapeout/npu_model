"""Cycle-level LSU paths from ScalarCore.scala and LSU.scala.

Cycles name the combinational work before the clock edge. An instruction issued
at T writes scalar stores at T+1, scalar load results at T+3, and vector rows at
T+3 through T+34. ScalarCore's command and response registers are included.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .exu import ExecutionUnit
from ..logging.logger import Logger, LaneType
from .arch_state import ArchState
from ..software.instruction import Uop
from ..isa import EXU
from .stage_data import StageData
from .config import HardwareConfig
from .bank_conflict import BankConflictError

_SCALAR_LOADS = {"lb", "lh", "lw", "lbu", "lhu", "seld"}
_SCALAR_STORES = {"sb", "sh", "sw"}
# Inclusive execute-stage cycles, including the scalar command/response flops.
LSU_OP_LATENCIES = {
    **dict.fromkeys(_SCALAR_LOADS, 4),
    **dict.fromkeys(_SCALAR_STORES, 2),
    "vload": 35,
    "vstore": 35,
}


@dataclass
class _Operation:
    uop: Uop
    issued: int
    address: int
    store_value: int = 0
    load_word: int = 0
    rows: dict[int, torch.Tensor] = field(default_factory=dict)


class LoadStoreUnit(ExecutionUnit):
    """Independent scalar, VLOAD, and VSTORE paths with streamed SRAM writes."""

    def __init__(self, name: str, logger: Logger, arch_state: ArchState,
                 lane_id: int = 0, config: HardwareConfig | None = None) -> None:
        super().__init__(name, logger, arch_state, lane_id, config)
        self.reset()

    def can_handle(self, uop: Uop) -> bool:
        return uop.insn.exu == EXU.LSU

    def reset(self) -> None:
        self.cycle = 0
        self._scalar: list[_Operation] = []
        self._vload: _Operation | None = None
        self._vstore: _Operation | None = None
        self._pending_completions: list[Uop] = []
        self._complete_count = 0
        self._total_instructions = 0
        self._busy_cycles = 0
        self.vmem_port_banks: frozenset[int] = frozenset()

    def abort(self) -> None:
        """Release LSU reservations after an explicitly ignored runtime fault."""
        for op in (self._vload, self._vstore):
            if op is not None:
                self.arch_state.conflict_checker.release_mreg(f"{self.name}:{op.uop.id}")
        self._scalar = []
        self._vload = None
        self._vstore = None
        self._pending_completions = []
        self.vmem_port_banks = frozenset()

    @property
    def in_flight(self) -> Uop | None:
        """Compatibility view for timeline consumers; execution has three paths."""
        ops = self._scalar + [op for op in (self._vload, self._vstore) if op]
        return min(ops, key=lambda op: op.issued).uop if ops else None

    def _get_latency(self, uop: Uop) -> int:
        if uop.insn.mnemonic in {"vload", "vstore"}:
            return self.arch_state.cfg.mrf_depth + 3
        return LSU_OP_LATENCIES[uop.insn.mnemonic]

    def _finish(self, op: _Operation) -> None:
        self._complete_count += 1
        self._pending_completions.append(op.uop)

    def _accept(self, uop: Uop) -> None:
        insn = uop.insn
        mnemonic = insn.mnemonic
        vector = mnemonic in {"vload", "vstore"}
        imm = int(insn.imm) & 0xFFF
        imm = imm - 0x1000 if imm & 0x800 else imm
        # The RTL connects only the VMEM-local address bits to LSU.
        addr_mask = (1 << (self.arch_state.cfg.vmem_size - 1).bit_length()) - 1
        alu_address = self.arch_state.read_xrf(insn.rs1) + (imm << 5 if vector else imm)
        if vector:
            # VLS operands are word addresses. ScalarCore drops three word
            # offset bits; LSU expands the resulting 32-byte line address.
            line_bits = (self.arch_state.cfg.vmem_size // 32 - 1).bit_length()
            address = ((alu_address >> 3) & ((1 << line_bits) - 1)) * 32
        else:
            address = alu_address & addr_mask
        assert address < self.arch_state.cfg.vmem_size, "LSU address exceeds VMEM capacity"
        op = _Operation(uop, self.cycle, address)
        if mnemonic in _SCALAR_LOADS:
            if any(old.uop.insn.mnemonic in _SCALAR_LOADS and self.cycle - old.issued < 3
                   for old in self._scalar):
                raise RuntimeError("scalar load issued while prior load response pending")
            self._scalar.append(op)
        elif mnemonic in _SCALAR_STORES:
            op.store_value = int(self.arch_state.read_xrf(insn.rs2)) & 0xFFFFFFFF
            self._scalar.append(op)
        elif vector:
            tile_bytes = self.arch_state.cfg.mrf_depth * self.arch_state.cfg.mrf_width
            bank_bytes = self.config.vmem_bank_bytes
            assert address % tile_bytes == 0, "VLOAD/VSTORE base must be 1 KiB aligned"
            assert address + tile_bytes <= self.arch_state.cfg.vmem_size, "LSU vector range exceeds VMEM capacity"
            assert address // bank_bytes == (address + tile_bytes - 1) // bank_bytes, "LSU vector range crosses VMEM bank"
            path = "_vload" if mnemonic == "vload" else "_vstore"
            if getattr(self, path) is not None:
                raise RuntimeError(f"{mnemonic.upper()} issued while {mnemonic.upper()} path is busy")
            banks = frozenset({insn.vd})
            self.arch_state.conflict_checker.reserve_mreg(
                f"{self.name}:{uop.id}",
                reads=banks if mnemonic == "vstore" else frozenset(),
                writes=banks if mnemonic == "vload" else frozenset(),
                allow_write_during_read=mnemonic == "vload",
            )
            setattr(self, path, op)
        else:
            raise ValueError(f"Unknown LSU instruction {mnemonic}")
        uop.execute_delay = self._get_latency(uop)
        self._total_instructions += 1
        self.logger.log_stage_end(uop.id, "D", lane=LaneType.DIU.value, cycle=self.cycle)
        self.logger.log_stage_start(uop.id, "E", lane=self.lane_id, cycle=self.cycle)

    def _check_vmem_ports(self) -> None:
        """LSU ports are deterministic; same physical-bank accesses assert."""
        accesses: list[tuple[str, int]] = []
        for op in self._scalar:
            if self.cycle - op.issued == 1:
                accesses.append((op.uop.insn.mnemonic, op.address))
        rows = self.arch_state.cfg.mrf_depth
        if self._vload and 1 <= self.cycle - self._vload.issued <= rows:
            accesses.append(("vload", self._vload.address))
        if self._vstore and 3 <= self.cycle - self._vstore.issued <= rows + 2:
            accesses.append(("vstore", self._vstore.address))
        banks: dict[int, str] = {}
        for mnemonic, address in accesses:
            bank = address // self.config.vmem_bank_bytes
            if bank in banks:
                raise BankConflictError(f"VMEM bank conflict: {mnemonic} and {banks[bank]} access bank {bank}")
            banks[bank] = mnemonic
        self.vmem_port_banks = frozenset(banks)

    def _check_writeback(self) -> None:
        current = getattr(self.arch_state, "current_uop", None)
        if current is None:
            return
        insn = current.insn
        if insn.exu == EXU.SCALAR:
            if insn.mnemonic == "seli":
                raise RuntimeError("scalar load response collides with SELI scale-register write")
            if hasattr(insn, "rd") and int(insn.rd) != 0:
                raise RuntimeError("scalar load response collides with scalar register writeback")

    def _tick_scalar(self) -> None:
        remaining: list[_Operation] = []
        for op in self._scalar:
            age = self.cycle - op.issued
            insn = op.uop.insn
            mnemonic = insn.mnemonic
            if mnemonic in _SCALAR_STORES and age == 1:
                size = {"sb": 1, "sh": 2, "sw": 4}[mnemonic]
                # Hardware ignores bit 0 for halfwords and bits 1:0 for words.
                address = op.address & ~(size - 1)
                raw = (op.store_value & ((1 << (size * 8)) - 1)).to_bytes(size, "little")
                self.arch_state.write_vmem(address, 0, torch.tensor(list(raw), dtype=torch.uint8))
                self._finish(op)
                continue
            if mnemonic in _SCALAR_LOADS:
                if age == 1:
                    raw = self.arch_state.read_vmem(op.address & ~3, 0, 4)
                    op.load_word = int.from_bytes(bytes(raw.tolist()), "little")
                if age == 3:
                    self._check_writeback()
                    value = op.load_word
                    if mnemonic in {"lb", "lbu"}:
                        value = (value >> ((op.address & 3) * 8)) & 0xFF
                        if mnemonic == "lb" and value & 0x80:
                            value |= 0xFFFFFF00
                    elif mnemonic in {"lh", "lhu"}:
                        value = (value >> ((op.address & 2) * 8)) & 0xFFFF
                        if mnemonic == "lh" and value & 0x8000:
                            value |= 0xFFFF0000
                    if mnemonic == "seld":
                        # ScalarCore selects memWord[7:0], without byte shifting.
                        self.arch_state.write_erf(insn.rd, value & 0xFF)
                    else:
                        self.arch_state.write_xrf(insn.rd, value)
                    self._finish(op)
                    continue
            remaining.append(op)
        self._scalar = remaining

    def _tick_vector(self, op: _Operation | None, *, load: bool) -> None:
        if op is None:
            return
        age = self.cycle - op.issued
        rows, width = self.arch_state.cfg.mrf_depth, self.arch_state.cfg.mrf_width
        reg = op.uop.insn.vd
        if 1 <= age <= rows:
            row = age - 1
            if load:
                op.rows[row] = self.arch_state.read_vmem(op.address + row * width, 0, width).clone()
            else:
                self.arch_state.conflict_checker.access_mreg(
                    self.cycle, reg, row, False, f"{self.name}:{op.uop.id}"
                )
                op.rows[row] = self.arch_state.read_mrf_u8(reg)[row].clone()
        if 3 <= age <= rows + 2:
            row = age - 3
            data = op.rows.pop(row)
            if load:
                self.arch_state.conflict_checker.access_mreg(
                    self.cycle, reg, row, True, f"{self.name}:{op.uop.id}"
                )
                self.arch_state.mrf[reg][row * width:(row + 1) * width] = data
            else:
                self.arch_state.write_vmem(op.address + row * width, 0, data)
        if age == rows + 2:
            self.arch_state.conflict_checker.release_mreg(f"{self.name}:{op.uop.id}")
            self._finish(op)
            if load:
                self._vload = None
            else:
                self._vstore = None

    def tick(self, idu_output: StageData[Uop | None]) -> None:
        self.cycle += 1
        self.arch_state.conflict_checker.begin_cycle(self.cycle)
        self.flush_completions()
        self._complete_count = 0
        uop = idu_output.claim()
        # Validate against pre-edge busy flags, including the final row cycle.
        if uop is not None:
            assert uop.insn.exu == EXU.LSU, "Non-LSU instruction passed to LSU"
            self._accept(uop)
        if self.has_in_flight:
            self._busy_cycles += 1
        self._check_vmem_ports()
        self._tick_scalar()
        self._tick_vector(self._vload, load=True)
        self._tick_vector(self._vstore, load=False)

    def flush_completions(self) -> None:
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id, cycle=self.cycle)
            self.logger.log_retire(uop.id)
        self._pending_completions = []

    def is_busy(self) -> bool:
        return self.has_in_flight

    @property
    def has_in_flight(self) -> bool:
        return bool(self._scalar or self._vload or self._vstore)

    @property
    def complete_count(self) -> int:
        return self._complete_count

    @property
    def total_instructions(self) -> int:
        return self._total_instructions

    @property
    def busy_cycles(self) -> int:
        return self._busy_cycles
