import math

import torch

from ..isa import EXU, RType, is_scalar_itype
from ..logging.logger import Logger
from ..software.instruction import Uop
from .arch_state import ArchState
from .config import HardwareConfig
from .exu import ExecutionUnit, StagedExecution
from .bank_conflict import vmem_accesses


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


class DmaExecutionUnit(ExecutionUnit):
    """
    Execution unit for DMA (Direct Memory Access) operations.
    Handles data transfers between DRAM and VMEM.

    Supports up to 8 in-flight DMA instructions at once, queuing them in
    order and executing them head-of-line. Transfer latency is computed from
    the byte count stored in XRF[rs2] divided by the configured VMEM
    bandwidth (vmem_bytes_per_cycle), with a minimum of 1 cycle.
    Config ops (dma.config.ch<N>) are treated as fixed 1-cycle control ops.

    Completion logging is deferred by one cycle so that the trace
    reflects the cycle in which results become visible, and the corresponding
    channel flag is cleared on completion to unblock any waiting dma.wait.ch<N>
    instructions held in S1.

    Values come from Instruction.exec, committed when a transfer completes.
    Its register operands are those S1 read at launch.
    """

    # What each command's exec may change.
    _WRITES = {"dma.load": [("vmem",)], "dma.store": [("dram",)], "dma.config": [("base",)]}

    def _bytes_for_dma_uop(self, uop: Uop) -> int:
        """
        Determine transfer size in bytes for DMA ops.

        Current programs conventionally place the byte count in XRF[rs2] for
        dma.load/store (R-type).
        """
        if isinstance(uop.insn, RType):
            return int(self.arch_state.read_xrf(uop.insn.rs2))
        return 0

    def __init__(
        self,
        name: str,
        logger: Logger,
        arch_state: ArchState,
        lane_id: int = 0,
        config: HardwareConfig | None = None,
    ) -> None:
        super().__init__(
            name,
            logger,
            arch_state,
            lane_id,
            config,
        )
        self.reset()

    def can_handle(self, uop: Uop) -> bool:
        return True

    def reset(self) -> None:
        self.in_flight: list[Uop] = []
        self._in_flight_vmem_banks: list[frozenset[int]] = []
        self._staged: dict[int, StagedExecution] = {}
        self._complete_count = 0
        self._pending_completions: list[Uop] = []
        self._total_instructions = 0
        self._busy_cycles = 0

    def tick(self, uop: Uop | None) -> None:
        self.cycle += 1
        # Log deferred completions from last cycle
        for done in self._pending_completions:
            if not (is_scalar_itype(done.insn) or isinstance(done.insn, RType)):
                raise ValueError("Invalid Instruction format provided to DMA.")

            self.logger.log_stage_end(done.id, "E", lane=self.lane_id, cycle=self.cycle)
            self.logger.log_retire(done.id)
            # clear the flag
            self.arch_state.clear_flag(done.insn.funct3)
            print(f"DMA {self.name} cleared flag {done.insn.funct3}")

            if len(self.in_flight) != 0:
                # Log: start execute
                self.logger.log_stage_start(
                    self.in_flight[0].id,
                    "E",
                    lane=self.lane_id,
                    cycle=self.cycle,
                )

        self._pending_completions = []

        self._complete_count = 0

        # Accept new instruction into the 8-entry in-order queue.
        if uop is not None:
            if len(self.in_flight) >= 8:
                raise RuntimeError(
                    f"DMA {self.name} queue full when S1 launched uop {uop.id} "
                    f"{uop.insn} on cycle {self.cycle}"
                )
            assert uop.insn.exu == EXU.DMA, "Invalid arguments passed to DMA Engine"
            # Check and acquire VMEM banks before accepting.
            mnemonic = uop.insn.mnemonic
            label = f"{self.name}:{mnemonic}"
            banks = vmem_accesses(uop.insn, self.arch_state)
            self.arch_state.conflict_checker.acquire_vmem(banks, label)
            self._in_flight_vmem_banks.append(banks)
            # tag instruction with execution delay
            if mnemonic.startswith("dma.config"):
                # Config is a control op; keep it fixed-latency.
                uop.execute_delay = 1
            else:
                nbytes = self._bytes_for_dma_uop(uop)
                uop.execute_delay = max(
                    1,
                    dma_transfer_cycles(self.config, nbytes),
                )
            writes = self._WRITES.get(mnemonic.rsplit(".", 1)[0], [])
            self._staged[uop.id] = StagedExecution(uop, self.arch_state, [], writes, self.name)
            self.in_flight.append(uop)
            self._total_instructions += 1

            if len(self.in_flight) == 1:
                # Log: start execute
                self.logger.log_stage_start(
                    uop.id,
                    "E",
                    lane=self.lane_id,
                    cycle=self.cycle,
                )

        # Track if EXU was busy
        if self.is_busy():
            self._busy_cycles += 1

        # Process in-flight instructions
        if len(self.in_flight) != 0:
            self.in_flight[0].execute_delay -= 1
            if self.in_flight[0].execute_delay <= 0:
                self._commit(self._staged.pop(self.in_flight[0].id))
                self._complete_count = 1
                # Release acquired VMEM banks before retiring the instruction.
                self.arch_state.conflict_checker.release_vmem(
                    self._in_flight_vmem_banks[0]
                )
                self._in_flight_vmem_banks = self._in_flight_vmem_banks[1:]
                # Defer completion logging to next tick
                self._pending_completions.append(self.in_flight[0])
                self.in_flight = self.in_flight[1:]

    def _commit(self, staged: StagedExecution) -> None:
        """Move the data as the transfer completes."""
        state = self.arch_state
        # dma.base is engine state, updated by configs ahead in this queue.
        staged.sample(("base",), ..., torch.tensor(state.base))
        for kind in ("vmem", "dram"):
            memory = getattr(state, kind)
            for start, data in list(staged.memory_reads[kind].items()):
                span = slice(start, start + len(data))
                staged.sample((kind,), span, memory[span])
        for location in staged.writes:
            value = staged.result(location)
            if location == ("base",):
                state.base = value
                continue
            memory = getattr(state, location[0])
            for start, data in value.items():
                memory[start:start + len(data)] = data

    def flush_completions(self) -> None:
        """Flush any pending completions (call at end of simulation)."""
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id)
            self.logger.log_retire(uop.id)
        self._pending_completions = []

    def is_busy(self) -> bool:
        """Check if the EXU is busy."""
        return len(self.in_flight) != 0 and self.in_flight[0].insn.mnemonic != "delay"

    @property
    def has_in_flight(self) -> bool:
        """Check if there are any in-flight instructions."""
        return len(self.in_flight) != 0

    @property
    def complete_count(self) -> int:
        """Instructions completed this cycle."""
        return self._complete_count

    @property
    def total_instructions(self) -> int:
        """Total instructions executed."""
        return self._total_instructions

    @property
    def busy_cycles(self) -> int:
        """Number of cycles the EXU was busy."""
        return self._busy_cycles
