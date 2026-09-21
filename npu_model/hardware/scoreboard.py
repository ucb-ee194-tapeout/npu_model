from npu_model.isa import Instruction
from npu_model.isa_patterns import (
    DMARegUnary,
    ExponentImm,
    ExponentOffsetLoad,
    JalrPattern,
    MXUAccumulatorPopE1,
    ScalarBaseOffsetStore,
    ScalarBranchImm,
    ScalarComputeImm,
    ScalarComputeReg,
    ScalarComputeShamt,
    ScalarImm,
    ScalarOffsetLoad,
    TensorBaseOffset,
    TensorComputeMixed,
)

from ..isa_types import EXU

_CSR_IMMEDIATE_MNEMONICS = frozenset({"csrrwi", "csrrsi", "csrrci"})
"""
`ScalarComputeImm.rs1` is a scalar register for every other instruction that
shares the pattern, but these three Zicsr immediate variants reuse that
field's bit position to encode a plain 5-bit immediate
(`self.rs1 & 0b11111` in `configs/isa_definition.py`) instead of indexing
`xrf`. Treating it as a register here would make the scheduler wait on a
register that was never actually read.
"""


def xrf_accesses(insn: Instruction) -> frozenset[int]:
    """Return the set of scalar (`x`) register indices read or written by `insn`."""
    if isinstance(insn, (ScalarOffsetLoad, JalrPattern, ScalarComputeShamt)):
        return frozenset({insn.rd, insn.rs1})

    if isinstance(insn, ScalarComputeImm):
        if insn.mnemonic in _CSR_IMMEDIATE_MNEMONICS:
            return frozenset({insn.rd})
        return frozenset({insn.rd, insn.rs1})

    if isinstance(insn, ScalarComputeReg):
        # Also matches _DMA_LOAD_CHN/_DMA_STORE_CHN, which read rd/rs1/rs2
        # instead of writing rd - the touched set is the same union either way.
        return frozenset({insn.rd, insn.rs1, insn.rs2})

    if isinstance(insn, (ScalarBaseOffsetStore, ScalarBranchImm)):
        return frozenset({insn.rs1, insn.rs2})

    if isinstance(insn, (ExponentOffsetLoad, TensorBaseOffset, DMARegUnary)):
        return frozenset({insn.rs1})

    if isinstance(insn, ScalarImm):
        return frozenset({insn.rd})

    return frozenset()


def ereg_accesses(insn: Instruction) -> frozenset[int]:
    """Return the set of scale (`e`) register indices read or written by `insn`."""
    if isinstance(insn, (ExponentOffsetLoad, ExponentImm)):
        return frozenset({insn.rd})

    if isinstance(insn, (TensorComputeMixed, MXUAccumulatorPopE1)):
        return frozenset({insn.es1})

    return frozenset()


class Scoreboard:
    """
    Tracks, per resource, the cycle at which it next becomes ready.

    Usage (from IDU's `schedule` mode, once per dispatched uop):
        ready = max(
            scoreboard.xrf_ready_cycle(regs_read | regs_written),
            scoreboard.mrf_ready_cycle(mrf_banks),
            scoreboard.ereg_ready_cycle(ereg_regs),
            scoreboard.exu_ready_cycle(uop.insn.exu),
        )
        ...
        scoreboard.mark_xrf_busy(regs_written, cycle + latency)
        scoreboard.mark_exu_busy(uop.insn.exu, cycle + latency)

    A resource that has never been marked busy is ready at cycle 0.
    """

    def __init__(self) -> None:
        self._xrf_ready: dict[int, int] = {}
        self._mrf_ready: dict[int, int] = {}
        self._ereg_ready: dict[int, int] = {}
        self._exu_ready: dict[EXU, int] = {}

    def reset(self) -> None:
        self._xrf_ready.clear()
        self._mrf_ready.clear()
        self._ereg_ready.clear()
        self._exu_ready.clear()


    @staticmethod
    def _ready_cycle(ready: dict[int, int], banks: frozenset[int]) -> int:
        """Cycle at which every bank in `banks` is free (0 if none are tracked)."""
        if not banks:
            return 0
        return max((ready.get(bank, 0) for bank in banks), default=0)

    @staticmethod
    def _mark_busy(ready: dict[int, int], banks: frozenset[int], until_cycle: int) -> None:
        for bank in banks:
            ready[bank] = max(ready.get(bank, 0), until_cycle)


    def xrf_ready_cycle(self, regs: frozenset[int]) -> int:
        """Cycle at which every register in `regs` holds its final value."""
        return self._ready_cycle(self._xrf_ready, regs)

    def mark_xrf_busy(self, regs: frozenset[int], until_cycle: int) -> None:
        """Mark `regs` as not holding their final value until `until_cycle`."""
        self._mark_busy(self._xrf_ready, regs, until_cycle)


    def mrf_ready_cycle(self, banks: frozenset[int]) -> int:
        return self._ready_cycle(self._mrf_ready, banks)

    def mark_mrf_busy(self, banks: frozenset[int], until_cycle: int) -> None:
        self._mark_busy(self._mrf_ready, banks, until_cycle)


    def ereg_ready_cycle(self, regs: frozenset[int]) -> int:
        return self._ready_cycle(self._ereg_ready, regs)

    def mark_ereg_busy(self, regs: frozenset[int], until_cycle: int) -> None:
        self._mark_busy(self._ereg_ready, regs, until_cycle)


    def exu_ready_cycle(self, exu: EXU) -> int:
        return self._exu_ready.get(exu, 0)

    def mark_exu_busy(self, exu: EXU, until_cycle: int) -> None:
        self._exu_ready[exu] = max(self._exu_ready.get(exu, 0), until_cycle)
