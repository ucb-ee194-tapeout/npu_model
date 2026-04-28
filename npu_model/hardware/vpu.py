import math

from .exu import ExecutionUnit
from ..logging.logger import Logger, LaneType
from ..hardware.arch_state import ArchState
from ..software.instruction import Uop
from ..isa import EXU
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

# Maps single-input operations to their logic family to prevent intra-family overlaps
VPU_OVERLAP_FAMILIES = {
    "vmov": "mov",
    "vrecip.bf16": "recip",
    "vexp.bf16": "exp",
    "vexp2.bf16": "exp",
    "vpack.bf16.fp8": "pack",
    "vunpack.fp8.bf16": "unpack",
    "vsquare.bf16": "pow",
    "vcube.bf16": "pow",
    "vrelu.bf16": "relu",
    "vsin.bf16": "trig",
    "vcos.bf16": "trig",
    "vtanh.bf16": "tanh",
    "vlog2.bf16": "log2",
    "vsqrt.bf16": "sqrt",
    "vredsum.bf16": "redsum",
    "vredmin.bf16": "redmin",
    "vredmax.bf16": "redmax",
    "vli.all": "vli",
    "vli.row": "vli",
    "vli.col": "vli",
    "vli.one": "vli",
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
        VPU Hardware Spec Dual-Issue rules.
        """
        if uop.insn.mnemonic == "delay":
            return True

        if len(self.in_flight) == 0:
            return True

        if len(self.in_flight) >= 2:
            return False

        existing_uop = self.in_flight[0]

        if existing_uop.insn.mnemonic == "delay":
            return True

        existing_mnemonic = existing_uop.insn.mnemonic
        new_mnemonic = uop.insn.mnemonic

        # Both the running op and the new op must belong to the overlapping set
        if (
            existing_mnemonic not in VPU_OVERLAP_FAMILIES
            or new_mnemonic not in VPU_OVERLAP_FAMILIES
        ):
            return False

        # They must not share the same logic family
        if (
            VPU_OVERLAP_FAMILIES[existing_mnemonic]
            == VPU_OVERLAP_FAMILIES[new_mnemonic]
        ):
            return False

        return True

    def tick(self, idu_output: StageData[Uop | None]) -> None:
        self.cycle += 1
        # Log deferred completions from last cycle
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id, cycle=self.cycle)
            self.logger.log_retire(uop.id)

        self._pending_completions = []
        self._complete_count = 0

        # Check if we can queue another instruction
        if len(self.in_flight) < 2:
            uop = idu_output.peek()

            if uop is not None:
                # Issue only if dual-issue rules pass
                if self.can_accept(uop):
                    assert (
                        uop.insn.exu == EXU.VECTOR
                    ), "Non-vector instruction passed to Vector Unit."
                    label = f"{self.name}:{uop.insn.mnemonic}"

                    # Normal MREG rules (bank conflicts) are implicitly verified here!
                    # If this errors out, it crashes precisely like your MREG spec requires.
                    mrf_banks = mrf_accesses(uop.insn)
                    vmem_banks = vmem_accesses(uop.insn, self.arch_state)

                    checker = self.arch_state.conflict_checker
                    checker.acquire_mrf(mrf_banks, label)
                    checker.acquire_vmem(vmem_banks, label)

                    self._in_flight_mrf_banks.append(mrf_banks)
                    self._in_flight_vmem_banks.append(vmem_banks)

                    uop.execute_delay = self._execution_latency(uop)
                    self.in_flight.append(uop)
                    self._total_instructions += 1

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
                    self._stalled = False
                else:
                    self._stalled = True
            else:
                self._stalled = False
        else:
            self._stalled = True

        if self.is_busy():
            self._busy_cycles += 1

        # Process in-flight instructions
        if self.in_flight:
            remaining_uops = []
            remaining_mrf = []
            remaining_vmem = []

            for i, uop in enumerate(self.in_flight):
                uop.execute_delay -= 1
                if uop.execute_delay <= 0:
                    uop.insn.exec(self.arch_state)
                    self._complete_count += 1

                    checker = self.arch_state.conflict_checker
                    checker.release_mrf(self._in_flight_mrf_banks[i])
                    checker.release_vmem(self._in_flight_vmem_banks[i])

                    self._pending_completions.append(uop)
                else:
                    remaining_uops.append(uop)
                    remaining_mrf.append(self._in_flight_mrf_banks[i])
                    remaining_vmem.append(self._in_flight_vmem_banks[i])

            self.in_flight = remaining_uops
            self._in_flight_mrf_banks = remaining_mrf
            self._in_flight_vmem_banks = remaining_vmem

    def flush_completions(self) -> None:
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id)
            self.logger.log_retire(uop.id)
        self._pending_completions = []

    def is_busy(self) -> bool:
        """Check if the EXU is fully busy."""
        if len(self.in_flight) == 0:
            return False

        if len(self.in_flight) >= 2:
            return True

        if self._stalled:
            return True

        existing_uop = self.in_flight[0]
        if existing_uop.insn.mnemonic == "delay":
            return False

        # If the existing instruction does not support overlap at all,
        # the VPU is fully busy and cannot accept ANY second instruction.
        if existing_uop.insn.mnemonic not in VPU_OVERLAP_FAMILIES:
            return True

        return False

    @property
    def has_in_flight(self) -> bool:
        return len(self.in_flight) > 0

    @property
    def complete_count(self) -> int:
        return self._complete_count

    @property
    def total_instructions(self) -> int:
        return self._total_instructions

    @property
    def busy_cycles(self) -> int:
        return self._busy_cycles
