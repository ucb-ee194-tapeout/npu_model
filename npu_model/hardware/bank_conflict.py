"""
Bank conflict detection for tensor register file (MRF) and VMEM.

Tensor registers and VMEM are implemented as banked SRAMs.  Simultaneous
accesses to the same bank by multiple in-flight instructions constitute a
bank conflict and must be avoided by software.  The performance model raises
BankConflictError when such a conflict is detected.

Bank mappings used by this checker:
  - MRF : one bank per tensor register (register index == bank index).
  - VMEM: 32-byte banks aligned to the DMA / tensor-transfer granularity.
"""

from typing import TYPE_CHECKING
from npu_model.isa import VRType
from ..isa_types import MatrixReg, WeightBuffer, Accumulator
from npu_model.isa_patterns import TensorBaseOffset, TensorComputeBinary, TensorComputeUnary, DirectImm, MXUAccumulatorPop, MXUWeightPush, MXUAccumulatorPopE1, MXUAccumulatorPush, MXUMatMul, ScalarComputeReg
from npu_model.configs.isa_definition import VMOV, VPACK_BF16_FP8, VUNPACK_FP8_BF16

if TYPE_CHECKING:
    from .arch_state import ArchState
    from ..isa import Instruction


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VMEM_BANK_BYTES: int = 32
"""Granularity of VMEM banks in bytes (matches DMA / vload / vstore alignment)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vmem_range_to_banks(base: int, length: int) -> frozenset[int]:
    """Convert a contiguous VMEM byte range to a set of 32-byte bank indices."""
    if length <= 0:
        return frozenset()
    first_bank = base // VMEM_BANK_BYTES
    last_bank = (base + length - 1) // VMEM_BANK_BYTES
    return frozenset(range(first_bank, last_bank + 1))


def _pair(reg: int) -> frozenset[int]:
    if isinstance(reg, MatrixReg):
        return frozenset({reg, MatrixReg(reg + 1)})
    return frozenset({reg, reg + 1})

def mrf_accesses(insn: Instruction) -> frozenset[int]:
    """
    Return the set of MRF register indices accessed by an instruction.

    Each register index is treated as its own SRAM bank.  Two concurrent
    instructions that share any index in their access sets constitute a
    bank conflict.
    """

    # vload, vstore, vli.* all only interact with vd
    if isinstance(insn, (TensorBaseOffset, DirectImm)):
        return frozenset({insn.vd})

    # all instructions that only use vd, vs1
    if isinstance(insn, TensorComputeUnary):
        if isinstance(insn, VMOV):
            return frozenset({insn.vs1, insn.vd})
        return _pair(insn.vs1) | _pair(insn.vd)
        
    # all instructions that use vd, vs1, vs2
    if isinstance(insn, TensorComputeBinary):
        return _pair(insn.vs1) | _pair(insn.vs2) | _pair(insn.vd)
    
    # vmatpush: reads one or two MRF registers into weight/acc buffer
    if isinstance(insn, MXUWeightPush):
        return frozenset({insn.vs1})
    
    if isinstance(insn, MXUAccumulatorPush):
        if insn.mnemonic.startswith("vmatpush.acc.bf16"):
            return _pair(insn.vs1)
        return frozenset({insn.vs1})

    if isinstance(insn, MXUAccumulatorPopE1):
        return frozenset({insn.vd})
    
    if isinstance(insn, MXUAccumulatorPop):
        return _pair(insn.vd)

    if isinstance(insn, MXUMatMul):
        return frozenset({insn.vs1})

    # Two-register read (bf16 pack): reads vs2 and vs2+1, writes vd -------
    if isinstance(insn, VPACK_BF16_FP8):
        return _pair(insn.vs2) | frozenset({insn.vd})

    # Two-register write (fp8 unpack): reads vs2, writes vd and vd+1 ------
    if isinstance(insn, VUNPACK_FP8_BF16):
        return frozenset({insn.vs2}) | _pair(insn.vd)

    if isinstance(insn, VRType):
        # Unknown VR type, raise an error
        raise ValueError(f"Unknown VR instruction passed to bank checker: {insn.mnemonic}")
    
    return frozenset()


def vmem_accesses(insn: Instruction, arch_state: ArchState) -> frozenset[int]:
    """
    Return the set of VMEM bank indices accessed by an instruction.

    Bank indices are computed from the byte address and length at dispatch
    time by reading the current scalar register file.
    """

    if isinstance(insn, TensorBaseOffset):
        addr = arch_state.read_xrf(insn.rs1) + (insn.imm << 5)
        length = arch_state.cfg.mrf_depth * arch_state.cfg.mrf_width
        return _vmem_range_to_banks(addr, length)

    if isinstance(insn, ScalarComputeReg):
        if insn.mnemonic.startswith("dma.load.ch"):
            vmem_addr = arch_state.read_xrf(insn.rd)
            length = arch_state.read_xrf(insn.rs2)
            return _vmem_range_to_banks(vmem_addr, length)
        if insn.mnemonic.startswith("dma.store.ch"):
            vmem_addr = arch_state.read_xrf(insn.rs1)
            length = arch_state.read_xrf(insn.rs2)
            return _vmem_range_to_banks(vmem_addr, length)

    return frozenset()


def weight_buffer_accesses(insn: Instruction) -> frozenset[int]:
    """Return the set of MXU IDs whose weight buffer is accessed."""
    if isinstance(insn, (MXUWeightPush, MXUMatMul)):
        return frozenset({WeightBuffer(int(insn.mnemonic[-1]))})
    return frozenset()


def acc_buffer_accesses(insn: Instruction) -> frozenset[int]:
    """Return the set of MXU IDs whose accumulation buffer is accessed."""
    if isinstance(insn, MXUMatMul):
        return frozenset({Accumulator(int(insn.mnemonic[-1]))})
    return frozenset()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class BankConflictError(RuntimeError):
    """
    Raised when two concurrently executing instructions access the same
    SRAM bank in the tensor register file or VMEM.
    """


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------


class BankConflictChecker:
    """
    Tracks which SRAM banks are currently in use by in-flight instructions
    and raises BankConflictError when a new instruction would access the
    same bank as an already-in-flight instruction.

    Usage (per execution unit):
        # When an instruction starts executing:
        self.arch_state.conflict_checker.acquire_mrf(banks, label)
        self.arch_state.conflict_checker.acquire_vmem(banks, label)

        # When the instruction completes:
        self.arch_state.conflict_checker.release_mrf(banks)
        self.arch_state.conflict_checker.release_vmem(banks)
    """

    def __init__(self) -> None:
        self._mrf_in_use: dict[int, str] = {}
        self._vmem_in_use: dict[int, str] = {}
        # Add tracking for MXU buffers
        self._weight_buf_in_use: dict[int, str] = {}
        self._acc_buf_in_use: dict[int, str] = {}
        self._mreg_reservations: dict[str, tuple[frozenset[int], frozenset[int]]] = {}
        self._mreg_ports: dict[tuple[bool, int], tuple[int, int, str]] = {}
        self._mreg_cycle: int | None = None
        self._mreg_releases: list[tuple[str, bool, bool]] = []

    def reset(self) -> None:
        self._mrf_in_use.clear()
        self._vmem_in_use.clear()
        self._weight_buf_in_use.clear()
        self._acc_buf_in_use.clear()
        self._mreg_reservations.clear()
        self._mreg_ports.clear()
        self._mreg_releases.clear()
        self._mreg_cycle = None

    def begin_cycle(self, cycle: int) -> None:
        """Apply last edge's releases before this cycle's issue checks.

        Deferral makes issue legality independent of the Python unit tick order.
        Every unit can call this with the same cycle; it advances only once.
        """
        if cycle == self._mreg_cycle:
            return
        for owner, release_reads, release_writes in self._mreg_releases:
            reads, writes = self._mreg_reservations.get(owner, (frozenset(), frozenset()))
            reads = frozenset() if release_reads else reads
            writes = frozenset() if release_writes else writes
            if reads or writes:
                self._mreg_reservations[owner] = (reads, writes)
            else:
                self._mreg_reservations.pop(owner, None)
        self._mreg_releases.clear()
        self._mreg_ports.clear()
        self._mreg_cycle = cycle

    def reserve_mreg(self, owner: str, reads: frozenset[int],
                     writes: frozenset[int], *, allow_write_during_read: bool = False) -> None:
        """ScalarCore's direction-aware logical-register issue assertions.

        VLOAD deliberately checks pending writes only. Physical SRAM port
        conflicts are checked separately on actual read/write request cycles.
        """
        if any(reg < 0 or reg >= 64 for reg in reads | writes):
            raise BankConflictError("MRF register outside m0..m63")
        for other, (busy_reads, busy_writes) in self._mreg_reservations.items():
            if other == owner:
                continue
            conflict = (reads & busy_writes) | (writes & busy_writes)
            if not allow_write_during_read:
                conflict |= writes & busy_reads
            if conflict:
                raise BankConflictError(
                    f"MRF bank conflict: '{owner}' accesses register(s) {sorted(conflict)} held by '{other}'"
                )
        self._mreg_reservations[owner] = (reads, writes)

    def release_mreg(self, owner: str, *, reads: bool = True, writes: bool = True) -> None:
        self._mreg_releases.append((owner, reads, writes))

    def access_mreg(self, cycle: int, reg: int, row: int, write: bool, owner: str) -> None:
        """Check the 32 physical 1R1W banks in MregFile.scala.

        mN and m(N+32) share a bank. One read and one write may coexist,
        except same-row read/write has undefined SyncReadMem data.
        """
        self.begin_cycle(cycle)
        bank = reg % 32
        physical_row = (reg // 32) * 32 + row
        key = (write, bank)
        if key in self._mreg_ports:
            other = self._mreg_ports[key][2]
            direction = "write" if write else "read"
            raise BankConflictError(
                f"MRF bank conflict: multiple {direction} ports targeting physical bank {bank}: {other}, {owner}"
            )
        opposite = self._mreg_ports.get((not write, bank))
        if opposite is not None and opposite[1] == physical_row:
            raise BankConflictError(
                f"Undefined MRF same-row read/write collision on physical bank {bank}, row {physical_row}"
            )
        self._mreg_ports[key] = (reg, physical_row, owner)

    # ------------------------------------------------------------------
    # MRF
    # ------------------------------------------------------------------

    def acquire_mrf(self, banks: frozenset[int], label: str) -> None:
        """
        Declare that the instruction identified by *label* is now using
        the given MRF banks.

        Raises BankConflictError if any of the requested banks is
        already held by a different in-flight instruction.
        """
        conflict = frozenset(self._mrf_in_use) & banks
        if conflict:
            holders = {self._mrf_in_use[b] for b in conflict}
            raise BankConflictError(
                f"MRF bank conflict: '{label}' accesses tensor register(s) "
                f"{sorted(conflict)} currently held by {holders}"
            )
        for bank in banks:
            self._mrf_in_use[bank] = label

    def release_mrf(self, banks: frozenset[int]) -> None:
        """Release the given MRF banks."""
        for bank in banks:
            self._mrf_in_use.pop(bank, None)

    # ------------------------------------------------------------------
    # Weight Buffer
    # ------------------------------------------------------------------
    def acquire_weight_buf(self, mxus: frozenset[int], label: str) -> None:
        conflict = frozenset(self._weight_buf_in_use) & mxus
        if conflict:
            holders = {self._weight_buf_in_use[b] for b in conflict}
            raise BankConflictError(
                f"Weight buffer conflict: '{label}' accesses MXU {sorted(conflict)} "
                f"currently held by {holders}"
            )
        for mxu in mxus:
            self._weight_buf_in_use[mxu] = label

    def release_weight_buf(self, mxus: frozenset[int]) -> None:
        for mxu in mxus:
            self._weight_buf_in_use.pop(mxu, None)

    # ------------------------------------------------------------------
    # Accumulation Buffer
    # ------------------------------------------------------------------
    def acquire_acc_buf(self, mxus: frozenset[int], label: str) -> None:
        conflict = frozenset(self._acc_buf_in_use) & mxus
        if conflict:
            holders = {self._acc_buf_in_use[b] for b in conflict}
            raise BankConflictError(
                f"Accumulation buffer conflict: '{label}' accesses MXU {sorted(conflict)} "
                f"currently held by {holders}"
            )
        for mxu in mxus:
            self._acc_buf_in_use[mxu] = label

    def release_acc_buf(self, mxus: frozenset[int]) -> None:
        for mxu in mxus:
            self._acc_buf_in_use.pop(mxu, None)

    # ------------------------------------------------------------------
    # VMEM
    # ------------------------------------------------------------------

    def acquire_vmem(self, banks: frozenset[int], label: str) -> None:
        """
        Declare that the instruction identified by *label* is now using
        the given VMEM banks.

        Raises BankConflictError if any of the requested banks is
        already held by a different in-flight instruction.
        """
        conflict = frozenset(self._vmem_in_use) & banks
        if conflict:
            holders = {self._vmem_in_use[b] for b in conflict}
            raise BankConflictError(
                f"VMEM bank conflict: '{label}' accesses VMEM banks "
                f"{sorted(conflict)} currently held by {holders}"
            )
        for bank in banks:
            self._vmem_in_use[bank] = label

    def release_vmem(self, banks: frozenset[int]) -> None:
        """Release the given VMEM banks."""
        for bank in banks:
            self._vmem_in_use.pop(bank, None)
