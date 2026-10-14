"""Cycle-level LSU paths from ScalarCore.scala and LSU.scala.

Cycles name the combinational work before the clock edge. An instruction issued
at T writes scalar stores at T+1, scalar load results at T+3, and vector rows at
T+3 through T+34. ScalarCore's command and response registers are included.
Values come from Instruction.exec; this unit decides only when they move.
"""
from __future__ import annotations

from dataclasses import dataclass

from .exu import ExecutionUnit, StagedExecution
from ..logging.logger import Logger
from .arch_state import ArchState
from ..software.instruction import Uop
from ..isa import EXU
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
    staged: StagedExecution


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
        reads: list[tuple] = []
        if mnemonic in _SCALAR_LOADS:
            if any(old.uop.insn.mnemonic in _SCALAR_LOADS and self.cycle - old.issued < 3
                   for old in self._scalar):
                raise RuntimeError("scalar load issued while prior load response pending")
            writes = [("erf" if mnemonic == "seld" else "xrf", int(insn.rd))]
        elif mnemonic in _SCALAR_STORES:
            writes = [("vmem",)]
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
            reads = [] if mnemonic == "vload" else [("mrf", int(insn.vd))]
            writes = [("mrf", int(insn.vd))] if mnemonic == "vload" else [("vmem",)]
        else:
            raise ValueError(f"Unknown LSU instruction {mnemonic}")
        op = _Operation(uop, self.cycle, address,
                        StagedExecution(uop, self.arch_state, reads, writes, "LSU"))
        if vector:
            setattr(self, path, op)
        else:
            self._scalar.append(op)
        uop.execute_delay = self._get_latency(uop)
        self._total_instructions += 1
        self.logger.log_stage_start(uop.id, "E", lane=self.lane_id, cycle=self.cycle)

    def _vmem_accesses_at(self, cycle: int) -> list[tuple[str, int]]:
        """(mnemonic, byte address) of every LSU VMEM port request on ``cycle``.

        Determined by operations issued before ``cycle``: scalar ports fire at
        issue+1, VLOAD reads rows at issue+1..+32, VSTORE writes at issue+3..+34.
        """
        accesses: list[tuple[str, int]] = []
        for op in self._scalar:
            if cycle - op.issued == 1:
                accesses.append((op.uop.insn.mnemonic, op.address))
        rows = self.arch_state.cfg.mrf_depth
        if self._vload and 1 <= cycle - self._vload.issued <= rows:
            row = cycle - self._vload.issued - 1
            accesses.append(("vload", self._vload.address + row * self.arch_state.cfg.mrf_width))
        if self._vstore and 3 <= cycle - self._vstore.issued <= rows + 2:
            row = cycle - self._vstore.issued - 3
            accesses.append(("vstore", self._vstore.address + row * self.arch_state.cfg.mrf_width))
        return accesses

    def _check_vmem_ports(self) -> None:
        """LSU ports are deterministic; same physical-bank accesses assert."""
        banks: dict[int, str] = {}
        for mnemonic, address in self._vmem_accesses_at(self.cycle):
            bank = address // self.config.vmem_bank_bytes
            if bank in banks:
                raise BankConflictError(f"VMEM bank conflict: {mnemonic} and {banks[bank]} access bank {bank}")
            banks[bank] = mnemonic
        self.vmem_port_banks = frozenset(banks)

    def _announce_vmem_ports(self) -> None:
        """Publish next cycle's bank accesses so grant-based clients (DMA) see them."""
        banks = {address // self.config.vmem_bank_bytes: mnemonic
                 for mnemonic, address in self._vmem_accesses_at(self.cycle + 1)}
        self.arch_state.conflict_checker.announce_vmem_ports(self.cycle + 1, banks)

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

    def _vmem_write(self, op: _Operation, start: int, length: int):
        """The bytes exec stored, which must be the region this unit addresses."""
        (address, data), = op.staged.result(("vmem",)).items()
        if (address, len(data)) != (start, length):
            raise RuntimeError(f"{op.uop.insn.mnemonic} exec wrote vmem[{address}:{address + len(data)}], "
                               f"but the LSU addresses vmem[{start}:{start + length}]")
        return data

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
                self.arch_state.write_vmem(address, 0, self._vmem_write(op, address, size))
                self._finish(op)
                continue
            if mnemonic in _SCALAR_LOADS:
                if age == 1:
                    word = op.address & ~3
                    op.staged.sample(("vmem",), slice(word, word + 4), self.arch_state.read_vmem(word, 0, 4))
                if age == 3:
                    self._check_writeback()
                    if mnemonic == "seld":
                        self.arch_state.write_erf(insn.rd, op.staged.result(("erf", int(insn.rd))))
                    else:
                        self.arch_state.write_xrf(insn.rd, op.staged.result(("xrf", int(insn.rd))))
                    self._finish(op)
                    continue
            remaining.append(op)
        self._scalar = remaining

    def _tick_vector(self, op: _Operation | None, *, load: bool) -> None:
        if op is None:
            return
        age = self.cycle - op.issued
        rows, width = self.arch_state.cfg.mrf_depth, self.arch_state.cfg.mrf_width
        reg = int(op.uop.insn.vd)
        owner = f"{self.name}:{op.uop.id}"
        if 1 <= age <= rows:
            row = age - 1
            if load:
                start = op.address + row * width
                op.staged.sample(("vmem",), slice(start, start + width), self.arch_state.read_vmem(start, 0, width))
            else:
                self.arch_state.conflict_checker.access_mreg(self.cycle, reg, row, False, owner)
                op.staged.sample(("mrf", reg), row, self.arch_state.read_mrf_u8(reg)[row])
        if 3 <= age <= rows + 2:
            row = age - 3
            if load:
                self.arch_state.conflict_checker.access_mreg(self.cycle, reg, row, True, owner)
                self.arch_state.read_mrf_u8(reg)[row] = op.staged.result(("mrf", reg))[row]
            else:
                data = self._vmem_write(op, op.address, rows * width)
                self.arch_state.write_vmem(op.address + row * width, 0, data[row * width:(row + 1) * width])
        if age == rows + 2:
            self.arch_state.conflict_checker.release_mreg(owner)
            self._finish(op)
            if load:
                self._vload = None
            else:
                self._vstore = None

    def tick(self, uop: Uop | None) -> None:
        self.cycle += 1
        self.arch_state.conflict_checker.begin_cycle(self.cycle)
        self.flush_completions()
        self._complete_count = 0
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
        self._announce_vmem_ports()

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
