"""Beat-level DMA engine following ``diplomatic/memory/DMA.scala``.

Cycle T names the combinational work before clock edge T, as in ``lsu.py``.
A command S1 launches at T occupies a slot from T+1. The engine then:

- issues one 32-byte TileLink beat per cycle for the oldest undispatched
  slot, while fewer than ``maxInFlight`` beats are outstanding, the next
  source ID is free, and (for stores) VMEM data is queued;
- for stores, reads VMEM one line per cycle ahead of the request stream,
  subject to the per-bank grant, with the SyncReadMem and store-queue
  delays (data is requestable two cycles after the read is granted);
- accepts responses in any order, writing load data to VMEM when the bank
  grants it, holding channel D otherwise;
- retires a slot, clearing its channel's busy flag, once every beat has been
  issued and acknowledged. Slots can retire out of enqueue order.

VMEM grants follow ``VMEM.scala``: any LSU access to a bank in a cycle denies
the DMA, and a DMA write beats a DMA read to the same bank. The LSU announces
its per-cycle bank accesses through the shared ``BankConflictChecker``.

Off-chip memory sits behind ``MemoryBackend`` (``memory_backend.py``).
Values come from ``Instruction.exec``: the engine samples source memory as
each beat is read and commits each beat's bytes as the RTL would write them.

``dma.config`` is not an engine command in the RTL; ScalarCore updates
``dmaBaseReg`` at issue. It is handled at the launch tick without a slot.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import torch

from ..isa import EXU, RType
from ..logging.logger import Logger
from ..software.instruction import Uop
from .arch_state import ArchState
from .config import HardwareConfig
from .exu import ExecutionUnit, StagedExecution
from .memory_backend import BeatRequest, MemoryBackend, make_memory_backend


# ---------------------------------------------------------------------------
# Frozen-baseline transfer estimates (npu_spec/05_memory_model)
# ---------------------------------------------------------------------------
# These closed forms are the spec's whole-transfer estimates. They no longer
# drive the engine; ``memory_backend.default_fixed_latency`` derives the
# default link parameters from the same spec constants.


def dma_offchip_cycles(config: HardwareConfig, nbytes: int) -> int:
    bytes_per_beat = config.offchip_link_width_bits // 8
    command_bytes = 4 * config.dma_offchip_command_words
    return (
        math.ceil((nbytes + command_bytes) / bytes_per_beat)
        * config.offchip_link_core_cycles_per_beat
    )


def vmem_transfer_cycles(config: HardwareConfig, nbytes: int) -> int:
    bytes_per_beat = config.vmem_bus_width_bits // 8
    return (
        math.ceil(nbytes / bytes_per_beat) * config.vmem_bus_core_cycles_per_beat
    )


def dma_transfer_cycles(config: HardwareConfig, nbytes: int) -> int:
    return max(
        dma_offchip_cycles(config, nbytes),
        vmem_transfer_cycles(config, nbytes),
    )


# ---------------------------------------------------------------------------
# Engine state
# ---------------------------------------------------------------------------


@dataclass
class _Slot:
    """One ``commandQueue`` entry with its ``slotActive`` bookkeeping."""

    uop: Uop
    staged: StagedExecution
    channel: int
    is_store: bool
    vmem_addr: int
    """Byte address of the first VMEM line."""
    dram_addr: int
    """64-bit off-chip address of the first beat ({dma.base, x[rs]})."""
    beats: int
    launched: int
    outstanding: int = 0
    dispatched: bool = False


@dataclass
class _BeatMeta:
    """``sourceMeta``: what a returning source ID means."""

    slot: _Slot
    index: int
    is_load: bool


@dataclass
class _StoreBeat:
    """A VMEM line read for a store, in ``storeDataQueue`` from ``available``."""

    slot: _Slot
    index: int
    available: int


@dataclass
class DmaCycle:
    """What the engine did in one tick; compared against RTL harness traces."""

    cycle: int
    busy: list[bool] = field(default_factory=list)
    """Per-channel busy flags as registered before this cycle (RTL ``channelBusy``)."""
    a: list | None = None
    """[source, address, is_store] of the channel A fire, if any."""
    d: int | None = None
    """Source ID of the channel D fire, if any."""
    vmem_read: list | None = None
    """[bank, bank line, granted] of the store-path VMEM read request, if valid."""
    vmem_write: list | None = None
    """[bank, bank line, granted] of the load-path VMEM write request, if valid."""


class DmaExecutionUnit(ExecutionUnit):
    """Multi-channel DMA engine with beat-level TileLink and VMEM traffic.

    ``backend`` is the off-chip memory model; tests may replace it before
    the first tick. ``last_cycle`` records the engine's observable events.
    """

    _WRITES = {"dma.load": [("vmem",)], "dma.store": [("dram",)]}

    def __init__(
        self,
        name: str,
        logger: Logger,
        arch_state: ArchState,
        lane_id: int = 0,
        config: HardwareConfig | None = None,
    ) -> None:
        super().__init__(name, logger, arch_state, lane_id, config)
        self.backend: MemoryBackend = make_memory_backend(self.config)
        self.beat_bytes = self.config.dma_beat_bytes
        self.num_channels = self.config.dma_num_channels
        self.max_in_flight = self.config.dma_max_in_flight
        self.num_ids = 1 << max(1, math.ceil(math.log2(self.max_in_flight)))
        self.max_transfer_bytes = self.config.dma_max_transfer_bytes
        self.bank_bytes = self.config.vmem_bank_bytes
        self.reset()

    # ------------------------------------------------------------------
    # ExecutionUnit interface
    # ------------------------------------------------------------------

    def can_handle(self, uop: Uop) -> bool:
        return uop.insn.exu == EXU.DMA

    def reset(self) -> None:
        self.cycle = 0
        self.backend.reset()
        self._slots: list[_Slot] = []
        self._request_beat = 0
        self._next_source = 0
        self._ids_in_flight: set[int] = set()
        self._in_flight = 0
        self._vmem_read_beat = 0
        self._vmem_read_complete = False
        self._store_queue: deque[_StoreBeat] = deque()
        self._source_meta: dict[int, _BeatMeta] = {}
        self._pending_completions: list[Uop] = []
        self._complete_count = 0
        self._total_instructions = 0
        self._busy_cycles = 0
        self.last_cycle = DmaCycle(0)

    def abort(self) -> None:
        """Drop engine state after an explicitly ignored runtime fault."""
        for slot in self._slots:
            self.arch_state.clear_flag(slot.channel)
        pending = self._pending_completions
        self.reset()
        self._pending_completions = pending

    def flush_completions(self) -> None:
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id, cycle=self.cycle)
            self.logger.log_retire(uop.id)
        self._pending_completions = []

    def is_busy(self) -> bool:
        return bool(self._slots)

    @property
    def has_in_flight(self) -> bool:
        return bool(self._slots)

    @property
    def in_flight(self) -> list[Uop]:
        """Launched commands not yet retired, oldest first."""
        return [slot.uop for slot in self._slots]

    @property
    def complete_count(self) -> int:
        return self._complete_count

    @property
    def total_instructions(self) -> int:
        return self._total_instructions

    @property
    def busy_cycles(self) -> int:
        return self._busy_cycles

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def _make_slot(self, uop: Uop) -> _Slot:
        insn = uop.insn
        if not isinstance(insn, RType):
            raise ValueError("Invalid instruction format provided to DMA.")
        state = self.arch_state
        mnemonic = insn.mnemonic
        is_store = mnemonic.startswith("dma.store")
        if not is_store and not mnemonic.startswith("dma.load"):
            raise ValueError(f"Unknown DMA instruction {mnemonic}")
        nbytes = state.read_xrf(insn.rs2)
        if nbytes <= 0 or nbytes > self.max_transfer_bytes or nbytes % self.beat_bytes:
            raise RuntimeError(
                f"{mnemonic} transfer of {nbytes} bytes: sizes must be multiples of "
                f"{self.beat_bytes} up to {self.max_transfer_bytes}"
            )
        vmem_operand = state.read_xrf(insn.rs1 if is_store else insn.rd)
        dram_operand = state.read_xrf(insn.rd if is_store else insn.rs1)
        # AtlasCore passes vmemAddr(wordAddrBits-1, wordOffBits) as the line.
        vmem_addr = state.dma_vmem_byte_address(vmem_operand)
        if vmem_addr + nbytes > state.cfg.vmem_size:          # DmaEngine asserts
            raise RuntimeError(f"{mnemonic} VMEM range exceeds VMEM capacity")
        dram_addr = state.dma_dram_address(dram_operand)
        state._check_dram_window(dram_addr, dram_addr + nbytes, f"{mnemonic} transfer")
        writes = self._WRITES[mnemonic.rsplit(".", 1)[0]]
        staged = StagedExecution(uop, state, [], writes, self.name)
        return _Slot(uop, staged, int(insn.funct3), is_store, vmem_addr, dram_addr,
                     nbytes // self.beat_bytes, self.cycle)

    # ------------------------------------------------------------------
    # Data movement
    # ------------------------------------------------------------------

    def _load_bytes(self, slot: _Slot, index: int) -> torch.Tensor:
        """The bytes exec stored to VMEM for beat ``index`` of a load."""
        writes = slot.staged.result(("vmem",))
        if slot.vmem_addr not in writes or len(writes[slot.vmem_addr]) != slot.beats * self.beat_bytes:
            raise RuntimeError(f"{slot.uop.insn.mnemonic} exec wrote vmem regions {sorted(writes)}, "
                               f"but the engine addresses vmem[{slot.vmem_addr}:"
                               f"{slot.vmem_addr + slot.beats * self.beat_bytes}]")
        start = index * self.beat_bytes
        return writes[slot.vmem_addr][start:start + self.beat_bytes]

    def _store_bytes(self, slot: _Slot, index: int) -> torch.Tensor:
        """The bytes exec stored off-chip for beat ``index`` of a store."""
        writes = slot.staged.result(("dram",))
        if slot.dram_addr not in writes or len(writes[slot.dram_addr]) != slot.beats * self.beat_bytes:
            raise RuntimeError(f"{slot.uop.insn.mnemonic} exec wrote dram regions {sorted(writes)}, "
                               f"but the engine addresses dram[{slot.dram_addr:#x}:"
                               f"{slot.dram_addr + slot.beats * self.beat_bytes:#x}]")
        start = index * self.beat_bytes
        return writes[slot.dram_addr][start:start + self.beat_bytes]

    def _bank(self, line_address: int) -> tuple[int, int]:
        return line_address // self.bank_bytes, (line_address % self.bank_bytes) // self.beat_bytes

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, uop: Uop | None) -> None:
        self.cycle += 1
        cycle = self.cycle
        state = self.arch_state
        checker = state.conflict_checker
        checker.begin_cycle(cycle)
        self.flush_completions()
        self._complete_count = 0

        # dma.config writes ScalarCore's base register at issue; no slot.
        if uop is not None and uop.insn.mnemonic.startswith("dma.config"):
            assert uop.insn.exu == EXU.DMA, "Invalid arguments passed to DMA Engine"
            self.logger.log_stage_start(uop.id, "E", lane=self.lane_id, cycle=cycle)
            uop.execute_delay = 1
            uop.insn.exec(state)
            self._pending_completions.append(uop)
            self._total_instructions += 1
            uop = None

        # Registered state this cycle's logic observes.
        slots = list(self._slots)
        dispatched_reg = [slot.dispatched for slot in slots]
        in_flight_reg = self._in_flight
        ids_reg = frozenset(self._ids_in_flight)
        head = next((slot for slot in slots if not slot.dispatched), None)
        record = DmaCycle(cycle, busy=[any(slot.channel == ch for slot in slots)
                                       for ch in range(self.num_channels)])
        if slots:
            self._busy_cycles += 1

        # Channel D: load data needs the VMEM write grant; store acks retire freely.
        d_fire = False
        d_slot: _Slot | None = None
        d_source: int | None = None
        write_bank: int | None = None
        response = self.backend.response(cycle)
        if response is not None:
            meta = self._source_meta[response.source]
            if meta.is_load:
                line = meta.slot.vmem_addr + meta.index * self.beat_bytes
                bank, bank_line = self._bank(line)
                granted = checker.vmem_bank_owner(cycle, bank) is None
                record.vmem_write = [bank, bank_line, granted]
                if granted:
                    write_bank = bank
                    state.write_vmem(line, 0, self._load_bytes(meta.slot, meta.index))
                    d_fire = True
            else:
                d_fire = True
            if d_fire:
                self.backend.pop_response(cycle)
                d_slot, d_source = meta.slot, response.source
                record.d = response.source

        # Store path: read VMEM lines ahead of the request stream.
        read_granted = False
        read_last = False
        if head is not None and head.is_store and not self._vmem_read_complete:
            line = head.vmem_addr + self._vmem_read_beat * self.beat_bytes
            bank, bank_line = self._bank(line)
            granted = checker.vmem_bank_owner(cycle, bank) is None and bank != write_bank
            record.vmem_read = [bank, bank_line, granted]
            if granted:
                head.staged.sample(("vmem",), slice(line, line + self.beat_bytes),
                                   state.read_vmem(line, 0, self.beat_bytes))
                # SyncReadMem data next cycle, then one cycle through storeDataQueue.
                self._store_queue.append(_StoreBeat(head, self._vmem_read_beat, cycle + 2))
                read_granted = True
                read_last = self._vmem_read_beat == head.beats - 1

        # Channel A: one beat of the oldest undispatched slot.
        a_fire = False
        a_last = False
        if head is not None:
            data_ready = (not head.is_store) or (
                bool(self._store_queue) and self._store_queue[0].available <= cycle)
            if (in_flight_reg < self.max_in_flight and data_ready
                    and self._next_source not in ids_reg and self.backend.can_accept(cycle)):
                index = self._request_beat
                address = head.dram_addr + index * self.beat_bytes
                if head.is_store:
                    entry = self._store_queue.popleft()
                    assert entry.slot is head and entry.index == index
                    state.dram[address:address + self.beat_bytes] = self._store_bytes(head, index)
                else:
                    head.staged.sample(("dram",), slice(address, address + self.beat_bytes),
                                       state.dram[address:address + self.beat_bytes])
                self.backend.issue(BeatRequest(self._next_source, address, head.is_store, cycle))
                self._source_meta[self._next_source] = _BeatMeta(head, index, not head.is_store)
                record.a = [self._next_source, address, head.is_store]
                a_fire = True
                a_last = index == head.beats - 1

        # Slot retirement uses the registered dispatched flag and this cycle's deltas.
        retired: list[_Slot] = []
        for slot, dispatched in zip(slots, dispatched_reg):
            inc = a_fire and slot is head
            dec = d_fire and slot is d_slot
            if inc and not dec:
                will_be_zero = False
            elif dec and not inc:
                will_be_zero = slot.outstanding == 1
            else:
                will_be_zero = slot.outstanding == 0
            slot.outstanding += int(inc) - int(dec)
            if dispatched and will_be_zero:
                retired.append(slot)

        # Register updates.
        if d_fire:
            assert d_source is not None
            self._in_flight -= 1
            self._ids_in_flight.discard(d_source)
            del self._source_meta[d_source]
        if a_fire:
            assert head is not None
            self._in_flight += 1
            self._ids_in_flight.add(self._next_source)
            self._next_source = (self._next_source + 1) % self.num_ids
            self._request_beat += 1
            if a_last:
                self._request_beat = 0
                self._vmem_read_beat = 0
                self._vmem_read_complete = False
                head.dispatched = True
        if read_granted and not a_last:
            self._vmem_read_beat += 1
            if read_last:
                self._vmem_read_complete = True
        for slot in retired:
            self._slots.remove(slot)
            state.clear_flag(slot.channel)
            self._pending_completions.append(slot.uop)
            self._complete_count += 1

        # Enqueue this cycle's launch; it is active from the next cycle.
        if uop is not None:
            assert uop.insn.exu == EXU.DMA, "Invalid arguments passed to DMA Engine"
            if len(self._slots) >= self.num_channels:
                raise RuntimeError(
                    f"DMA {self.name} command queue full when S1 launched uop {uop.id} "
                    f"{uop.insn} on cycle {cycle}"
                )
            slot = self._make_slot(uop)
            if any(other.channel == slot.channel for other in self._slots):
                raise RuntimeError(f"DMA channel {slot.channel} is busy; issue is illegal")
            uop.execute_delay = slot.beats
            self._slots.append(slot)
            self._total_instructions += 1
            self.logger.log_stage_start(uop.id, "E", lane=self.lane_id, cycle=cycle)

        self.last_cycle = record
