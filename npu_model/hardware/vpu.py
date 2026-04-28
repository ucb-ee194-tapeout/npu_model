import math

from .exu import ExecutionUnit
from ..logging.logger import Logger, LaneType
from ..hardware.arch_state import ArchState
from ..software.instruction import Uop
from ..isa import EXU
from ..isa_patterns import TensorComputeUnary
from .stage_data import StageData
from .config import HardwareConfig
from .bank_conflict import mrf_accesses, vmem_accesses


VPU_OP_LATENCIES = {
    # Col sum/max/min (Latency: 130)
    "vredsum.bf16": 130,
    "vredmin.bf16": 130,
    "vredmax.bf16": 130,
    # row reductions
    "vredsum.row.bf16": 39,
    "vredmin.row.bf16": 34,
    "vredmax.row.bf16": 34,
    # Vli (Latency: 65)
    # Vector Load Immediate instructions
    "vli.all": 65,
    "vli.row": 65,
    "vli.col": 65,
    "vli.one": 65,
    # Everything else (Latency: 66)
    # Element-wise arithmetic, conversions, and Matrix Unit (MXU) interactions
    "vadd.bf16": 66,
    "vsub.bf16": 66,
    "vmul.bf16": 66,
    "vminimum.bf16": 66,
    "vmaximum.bf16": 66,
    "vmov": 66,
    "vrecip.bf16": 66,
    "vexp.bf16": 66,
    "vexp2.bf16": 66,
    "vpack.bf16.fp8": 66,
    "vunpack.fp8.bf16": 66,
    "vrelu.bf16": 66,
    "vsin.bf16": 66,
    "vcos.bf16": 66,
    "vtanh.bf16": 66,
    "vlog2.bf16": 66,
    "vsqrt.bf16": 66,
    "vsquare.bf16": 66,
    "vcube.bf16": 66,
    "vtrpose.xlu": 66,
}


class VectorExecutionUnit(ExecutionUnit):
    """Execution unit for vector operations."""

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
        self._in_flight_mrf_banks: list[frozenset[int]] = []
        self._in_flight_vmem_banks: list[frozenset[int]] = []
        self._complete_count = 0
        self._pending_completions: list[Uop] = []
        self._total_instructions = 0
        self._busy_cycles = 0

        # Tracks if we rejected the instruction in idu_output last cycle
        self._stalled = False

    def _execution_latency(self, uop: Uop) -> int:
        mnemonic = uop.insn.mnemonic
        return VPU_OP_LATENCIES.get(mnemonic, 1)

    def can_accept(self, uop: Uop) -> bool:
        """
        VPU specific check:
        - 0 in flight: Accept anything.
        - 1 in flight: Accept ONLY if BOTH the existing and new instruction are TensorComputeUnary.
        - 2 in flight: Accept nothing.
        """
        if len(self.in_flight) == 0:
            return True

        if len(self.in_flight) >= 2:
            return False

        existing_uop = self.in_flight[0]

        if isinstance(existing_uop.insn, TensorComputeUnary) and isinstance(
            uop.insn, TensorComputeUnary
        ):
            return True

        return False

    def tick(self, idu_output: StageData[Uop | None]) -> None:
        self.cycle += 1
        # Log deferred completions from last cycle
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id, cycle=self.cycle)
            self.logger.log_retire(uop.id)

        self._pending_completions = []
        self._complete_count = 0

        # check if we can queue another instruction (Max 2 simultaneous Uops)
        if len(self.in_flight) < 2:
            uop = idu_output.peek()

            # accept new instruction logic
            if uop is not None:
                if self.can_accept(uop):
                    assert (
                        uop.insn.exu == EXU.VECTOR
                    ), "Non-vector instruction passed to Vector Unit."
                    label = f"{self.name}:{uop.insn.mnemonic}"
                    mrf_banks = mrf_accesses(uop.insn)
                    vmem_banks = vmem_accesses(uop.insn, self.arch_state)

                    checker = self.arch_state.conflict_checker
                    checker.acquire_mrf(mrf_banks, label)
                    checker.acquire_vmem(vmem_banks, label)

                    self._in_flight_mrf_banks.append(mrf_banks)
                    self._in_flight_vmem_banks.append(vmem_banks)

                    # Tag instruction with execution delay
                    uop.execute_delay = self._execution_latency(uop)
                    self.in_flight.append(uop)
                    self._total_instructions += 1

                    # Log: end dispatch, start execute
                    self.logger.log_stage_end(
                        uop.id,
                        "D",
                        lane=LaneType.DIU.value,
                        cycle=self.cycle,
                    )
                    self.logger.log_stage_start(
                        uop.id,
                        "E",
                        lane=self.lane_id,
                        cycle=self.cycle,
                    )

                    idu_output.claim()
                    self._stalled = False  # Successfully claimed, lift stall
                else:
                    # We couldn't accept it, so we leave it in the queue and assert stall
                    self._stalled = True
            else:
                self._stalled = False
        else:
            # Cannot peek because we are completely full
            self._stalled = True

        # Track if EXU was busy (for metrics)
        if self.is_busy():
            self._busy_cycles += 1

        # Process in-flight instructions (in parallel)
        if self.in_flight:
            remaining_uops = []
            remaining_mrf = []
            remaining_vmem = []

            for i, uop in enumerate(self.in_flight):
                uop.execute_delay -= 1
                if uop.execute_delay <= 0:
                    # execute the instruction
                    uop.insn.exec(self.arch_state)
                    self._complete_count += 1

                    # Release acquired banks before retiring the instruction.
                    checker = self.arch_state.conflict_checker
                    checker.release_mrf(self._in_flight_mrf_banks[i])
                    checker.release_vmem(self._in_flight_vmem_banks[i])

                    # Defer completion logging to next tick
                    self._pending_completions.append(uop)
                else:
                    # Instruction needs more cycles
                    remaining_uops.append(uop)
                    remaining_mrf.append(self._in_flight_mrf_banks[i])
                    remaining_vmem.append(self._in_flight_vmem_banks[i])

            self.in_flight = remaining_uops
            self._in_flight_mrf_banks = remaining_mrf
            self._in_flight_vmem_banks = remaining_vmem

    def flush_completions(self) -> None:
        """Flush any pending completions (call at end of simulation)."""
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id)
            self.logger.log_retire(uop.id)
        self._pending_completions = []

    def is_busy(self) -> bool:
        """Check if the EXU is busy."""
        # pipeline is completely empty
        if len(self.in_flight) == 0:
            return False

        # pipeline is entirely full
        if len(self.in_flight) >= 2:
            return True

        # we explicitly stalled upstream because we couldn't take the peeked instruction
        if self._stalled:
            return True

        # we have 1 instruction in flight, and it is not a TensorComputeUnary
        # we cannot accept a second instruction, so we must assert busy to stall the DIU
        insn = self.in_flight[0].insn
        if insn.mnemonic != "delay" and not isinstance(insn, TensorComputeUnary):
            return True

        return False

    @property
    def has_in_flight(self) -> bool:
        """Check if there are any in-flight instructions."""
        return len(self.in_flight) > 0

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
