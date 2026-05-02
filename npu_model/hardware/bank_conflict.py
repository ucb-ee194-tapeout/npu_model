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
from npu_model.software.instruction import Instruction
from ..isa_types import MatrixReg, WeightBuffer, Accumulator
from npu_model.isa_patterns import TensorBaseOffset, TensorComputeBinary, TensorComputeUnary, DirectImm, MXUAccumulatorPop, MXUWeightPush, MXUAccumulatorPopE1, MXUAccumulatorPush, MXUMatMul, ScalarComputeReg
from npu_model.configs.isa_definition import VMOV, VPACK_BF16_FP8, VUNPACK_FP8_BF16

if TYPE_CHECKING:
    from .arch_state import ArchState


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

    def reset(self) -> None:
        self._mrf_in_use.clear()
        self._vmem_in_use.clear()
        self._weight_buf_in_use.clear()
        self._acc_buf_in_use.clear()

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
