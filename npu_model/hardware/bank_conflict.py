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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    return frozenset({reg, reg + 1})


# ---------------------------------------------------------------------------
# Resource-access queries
# ---------------------------------------------------------------------------

#: Instructions whose MRF operand set is ``{vs1, vs2, vd}`` (both sources + dest).
_VR_TWO_SRC = frozenset(
    {
        "vadd.bf16",
        "vsub.bf16",
        "vmul.bf16",
        "vminimum.bf16",
        "vmaximum.bf16",
    }
)

#: Instructions whose MRF operand set is ``{vs1, vd}`` (one source + dest).
_VR_ONE_SRC = frozenset(
    {
        "vredsum.bf16",
        "vredmin.bf16",
        "vredmax.bf16",
        "vredsum.row.bf16",
        "vredmin.row.bf16",
        "vredmax.row.bf16",
        "vmov",
        "vrecip.bf16",
        "vexp.bf16",
        "vexp2.bf16",
        "vrelu.bf16",
        "vsquare.bf16",
        "vcube.bf16",
        "vsin.bf16",
        "vcos.bf16",
        "vtanh.bf16",
        "vlog2.bf16",
        "vsqrt.bf16",
        "vtrpose.xlu",
    }
)


def mrf_accesses(mnemonic: str, args: Any) -> frozenset[int]:
    """
    Return the set of MRF register indices accessed by an instruction.

    Each register index is treated as its own SRAM bank.  Two concurrent
    instructions that share any index in their access sets constitute a
    bank conflict.
    """
    from ..isa import VectorArgs, MatrixArgs

    if isinstance(args, VectorArgs):
        vd, vs1, vs2 = args.vd, args.vs1, args.vs2

        # VLS ------------------------------------------------------------------
        if mnemonic in {"vload", "vstore"}:
            return frozenset({vd})

        # vmatpush: reads one or two MRF registers into weight/acc buffer ------
        if mnemonic in {
            "vmatpush.weight.mxu0",
            "vmatpush.weight.mxu1",
            "vmatpush.acc.fp8.mxu0",
            "vmatpush.acc.fp8.mxu1",
        }:
            return frozenset({vs1})

        if mnemonic in {
            "vmatpush.acc.bf16.mxu0",
            "vmatpush.acc.bf16.mxu1",
        }:
            # bf16 tile occupies two consecutive registers
            return frozenset({vs1, vs1 + 1})

        # vmatpop: writes acc buffer into one or two MRF registers -------------
        if mnemonic in {
            "vmatpop.fp8.acc.mxu0",
            "vmatpop.fp8.acc.mxu1",
        }:
            return frozenset({vd})

        if mnemonic in {
            "vmatpop.bf16.acc.mxu0",
            "vmatpop.bf16.acc.mxu1",
        }:
            return frozenset({vd, vd + 1})

        # Two-register read (bf16 pack): reads vs1 and vs1+1, writes vd -------
        if mnemonic == "vpack.bf16.fp8":
            return frozenset({vs1, vs1 + 1, vd})

        # Two-register write (fp8 unpack): reads vs1, writes vd and vd+1 ------
        if mnemonic == "vunpack.fp8.bf16":
            return frozenset({vs1, vd, vd + 1})

        # VI (immediate load): only writes vd, no MRF source reads -------------
        if mnemonic in {"vli.all", "vli.row", "vli.col", "vli.one"}:
            return frozenset({vd})

        # Two-source VR instructions -------------------------------------------
        if mnemonic in _VR_TWO_SRC:
            return _pair(vs1) | _pair(vs2) | _pair(vd)

        # One-source VR instructions (default) --------------------------------
        if mnemonic in _VR_ONE_SRC:
            if mnemonic == "vmov":
                return frozenset({vs1, vd})
            return _pair(vs1) | _pair(vd)

        # Fall-through: unknown VR – include all three fields conservatively
        return frozenset({vs1, vs2, vd}) if vs2 else frozenset({vs1, vd})

    if isinstance(args, MatrixArgs):
        # vmatmul.*: reads activation from MRF[vs1]; result goes to local acc
        return frozenset({args.vs1})

    return frozenset()


def vmem_accesses(mnemonic: str, args: Any, arch_state: ArchState) -> frozenset[int]:
    """
    Return the set of VMEM bank indices accessed by an instruction.

    Bank indices are computed from the byte address and length at dispatch
    time by reading the current scalar register file.
    """
    from ..isa import VectorArgs, DmaArgs

    if isinstance(args, VectorArgs):
        if mnemonic in {"vload", "vstore"}:
            addr = arch_state.read_xrf(args.rs1) + (args.imm12 << 5)
            length = arch_state.cfg.mrf_depth * arch_state.cfg.mrf_width
            return _vmem_range_to_banks(addr, length)

    if isinstance(args, DmaArgs):
        if mnemonic == "dma.load.ch<N>":
            vmem_addr = arch_state.read_xrf(args.rd)
            length = arch_state.read_xrf(args.rs2)
            return _vmem_range_to_banks(vmem_addr, length)
        if mnemonic == "dma.store.ch<N>":
            vmem_addr = arch_state.read_xrf(args.rs1)
            length = arch_state.read_xrf(args.rs2)
            return _vmem_range_to_banks(vmem_addr, length)

    return frozenset()


def weight_buffer_accesses(mnemonic: str) -> frozenset[int]:
    """Return the set of MXU IDs whose weight buffer is accessed."""
    if "weight.mxu0" in mnemonic or "matmul.mxu0" in mnemonic:
        return frozenset({0})
    if "weight.mxu1" in mnemonic or "matmul.mxu1" in mnemonic:
        return frozenset({1})
    return frozenset()


def acc_buffer_accesses(mnemonic: str) -> frozenset[int]:
    """Return the set of MXU IDs whose accumulation buffer is accessed."""
    if ".acc." in mnemonic and "mxu0" in mnemonic:
        return frozenset({0})
    if ".acc." in mnemonic and "mxu1" in mnemonic:
        return frozenset({1})
    if "matmul.mxu0" in mnemonic:
        return frozenset({0})
    if "matmul.mxu1" in mnemonic:
        return frozenset({1})
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
