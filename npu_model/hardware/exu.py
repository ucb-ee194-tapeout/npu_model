from abc import abstractmethod
from typing import Iterable
import copy

import torch

from .hardware import Module
from ..logging.logger import Logger
from ..hardware.arch_state import ArchState
from ..software.instruction import Uop
from ..isa import EXU
from ..hardware.config import HardwareConfig

Location = tuple
"""Storage a unit samples or commits: a tile, ("mrf", reg), ("acc", mxu, slot)
or ("wb", mxu, slot); a register, ("xrf", reg), ("erf", reg) or ("base",); or
a whole byte memory, ("vmem",) or ("dram",)."""

_MEMORIES = ("vmem", "dram")
# Control state only S1 and the scalar unit change; staged exec must not.
_FIXED = ("pc", "npc", "redirect_requested", "halted", "halt_reason", "in_delay_slot",
          "flags", "csrf", "_csr_values", "_csr_written")


def _tiles(state: ArchState) -> dict[Location, torch.Tensor]:
    """Every tensor an instruction can read or write, as row-indexed views."""
    tiles: dict[Location, torch.Tensor] = {
        ("mrf", reg): state.read_mrf_u8(reg) for reg in range(len(state.mrf))
    }
    for mxu in state.acc:
        tiles.update({("acc", mxu, slot): data for slot, data in enumerate(state.acc[mxu])})
        tiles.update({("wb", mxu, slot): state.read_wb_u8(mxu, slot) for slot in range(len(state.wb[mxu]))})
    return tiles


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """Integer encodings, so NaN payloads and signed zeros compare exactly."""
    return tensor.view(torch.int16) if tensor.dtype == torch.bfloat16 else tensor


def _name(location: Location) -> str:
    kind, *index = location
    if kind in ("mrf", "xrf", "erf"):
        return f"{kind[0]}{index[0]}"
    return kind + "".join(f"[{part}]" for part in index)


class _Memory:
    """A byte memory as staged exec sees it.

    Reads return sampled regions, sampling live memory on first use; writes
    are captured for the unit to commit. Only regions exec touches are copied.
    """

    def __init__(self, live: torch.Tensor, reads: dict[int, torch.Tensor]) -> None:
        self.live = live
        self.reads = reads
        self.writes: dict[int, torch.Tensor] = {}

    def __getitem__(self, key: slice) -> torch.Tensor:
        for regions in (self.writes, self.reads):
            for start, data in regions.items():
                if start <= key.start and key.stop <= start + len(data):
                    return data[key.start - start:key.stop - start].clone()
        self.reads[key.start] = self.live[key].clone()
        return self.reads[key.start].clone()

    def __setitem__(self, key: slice, value: torch.Tensor) -> None:
        self.writes[key.start] = value.flatten().clone()


class StagedExecution:
    """Instruction.exec evaluated on operands as an execution unit samples them.

    exec runs against private copies, so nothing becomes architecturally
    visible until the unit commits ``result`` on its RTL write cycles. Scalar
    registers are the values S1 read at launch. Sampling data that changed
    since it was copied (VLOAD may write during a read window, a weight push
    may lead a matmul) re-runs exec before the next commit. That is exact
    because the RTL writes each result only after reading the inputs it
    depends on.
    """

    def __init__(self, uop: Uop, state: ArchState, reads: Iterable[Location],
                 writes: Iterable[Location], unit: str) -> None:
        self.uop = uop
        self.state = state
        self.unit = unit
        self.writes = frozenset(writes)
        tiles = _tiles(state)
        self.operands: dict[Location, torch.Tensor] = {
            ("xrf",): torch.tensor(state.xrf, dtype=torch.int64),
            ("erf",): torch.tensor(state.erf, dtype=torch.int64),
            ("base",): torch.tensor(state.base, dtype=torch.int64),
            **{location: tiles[location].clone() for location in reads},
        }
        # Memory operands are the regions exec reads, keyed by start address.
        self.memory_reads: dict[str, dict[int, torch.Tensor]] = {kind: {} for kind in _MEMORIES}
        self._execute()

    def _execute(self) -> None:
        state, operands = self.state, self.operands
        sources = {location: operands.get(location, data) for location, data in _tiles(state).items()}
        copies = {location: data.clone() for location, data in sources.items()}
        xrf, erf = operands[("xrf",)].tolist(), operands[("erf",)].tolist()
        base = int(operands[("base",)])
        memories = {kind: _Memory(getattr(state, kind), self.memory_reads[kind]) for kind in _MEMORIES}
        view = copy.copy(state)
        # Instance-level overrides (instrumentation) would reach live state.
        for name in [name for name, value in vars(view).items() if callable(value)]:
            delattr(view, name)
        view.logger = None  # The unit logs architectural values as it commits them.
        view.mrf = [copies[("mrf", reg)].view(-1) for reg in range(len(view.mrf))]
        view.acc = {mxu: [copies[("acc", mxu, slot)] for slot in range(len(slots))]
                    for mxu, slots in view.acc.items()}
        view.wb = {mxu: [copies[("wb", mxu, slot)].view(-1) for slot in range(len(slots))]
                   for mxu, slots in view.wb.items()}
        view.xrf, view.erf, view.base = list(xrf), list(erf), base
        view.vmem, view.dram = memories["vmem"], memories["dram"]
        for name in _FIXED:
            setattr(view, name, copy.copy(getattr(state, name)))
        self.uop.insn.exec(view)
        values: dict[Location, object] = {
            **copies,
            **{("xrf", reg): value for reg, value in enumerate(view.xrf)},
            **{("erf", reg): value for reg, value in enumerate(view.erf)},
            ("base",): view.base,
            **{(kind,): memory.writes for kind, memory in memories.items()},
        }
        written = [location for location, before in sources.items()
                   if not torch.equal(_bits(before), _bits(copies[location]))]
        written += [("xrf", reg) for reg, value in enumerate(xrf) if view.xrf[reg] != value]
        written += [("erf", reg) for reg, value in enumerate(erf) if view.erf[reg] != value]
        written += [("base",)] * (view.base != base)
        written += [(kind,) for kind, memory in memories.items() if memory.writes]
        written += [(name,) for name in _FIXED if getattr(view, name) != getattr(state, name)]
        for location in written:
            if location not in self.writes:
                raise RuntimeError(f"{self.uop.insn.mnemonic} exec wrote {_name(location)}, "
                                   f"which {self.unit} does not write")
        self._results = {location: values[location] for location in self.writes}
        self._stale = False

    def _region(self, kind: str, index: slice) -> tuple[int, torch.Tensor]:
        for start, data in self.memory_reads[kind].items():
            if start <= index.start and index.stop <= start + len(data):
                return start, data
        raise RuntimeError(f"{self.unit} reads {kind}[{index.start}:{index.stop}], "
                           f"which {self.uop.insn.mnemonic} exec does not read")

    def sample(self, location: Location, index, value: torch.Tensor) -> None:
        """Record ``value`` as what the unit read from ``location[index]``.

        A memory index is an absolute slice inside a region exec read.
        """
        if location[0] in _MEMORIES:
            start, operand = self._region(location[0], index)
            index = slice(index.start - start, index.stop - start)
        else:
            operand = self.operands[location]
        operand, value = _bits(operand), _bits(value)
        if not torch.equal(operand[index], value):
            operand[index] = value
            self._stale = True

    def result(self, location: Location):
        """A committed tile, a register value, or a memory's {start: bytes}."""
        if self._stale:
            self._execute()
        return self._results[location]


class ExecutionUnit(Module):
    """
    Abstract base class for execution units.

    Defines the interface that all execution units must implement.
    Subclass SimpleExecutionUnit for a ready-to-use implementation
    with latency modeling and trace logging.
    """

    def __init__(
        self,
        # name for logging purposes
        name: str,
        # handle to the logger
        logger: Logger,
        # handle to the architectural state
        arch_state: ArchState,
        # lane id for logging purposes
        lane_id: int = 0,
        # hardware configuration
        config: HardwareConfig | None = None,
    ) -> None:
        if config == None:
            raise ValueError("A HardwareConfig must be specified.")

        self.name = name
        self.logger = logger
        self.arch_state = arch_state
        self.lane_id = lane_id
        self.config = config
        self.cycle: int = 0

    @abstractmethod
    def can_handle(self, uop: Uop) -> bool:
        """Check if this execution unit can handle the given instruction."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the execution unit state."""
        pass

    @abstractmethod
    def tick(self, uop: Uop | None) -> None:
        """Execute one cycle; ``uop`` is the command S1 launched to this unit, if any."""
        pass

    @abstractmethod
    def flush_completions(self) -> None:
        """Flush any pending completions (call at end of simulation)."""
        pass

    @property
    @abstractmethod
    def has_in_flight(self) -> bool:
        """Check if there are any in-flight instructions."""
        pass

    @property
    @abstractmethod
    def complete_count(self) -> int:
        """Number of instructions completed this cycle."""
        pass

    @property
    @abstractmethod
    def total_instructions(self) -> int:
        """Total instructions executed."""
        pass

    @property
    @abstractmethod
    def busy_cycles(self) -> int:
        """Number of cycles the EXU was busy."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class ScalarExecutionUnit(ExecutionUnit):
    """
    Execution unit for scalar operations.
    Always executes 1 scalar instruction per cycle.
    Execute delay is always 1 cycle.
    """

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
        # List of memory instructions that should go to the LSU instead
        mem_ops = {
            "lb",
            "lh",
            "lw",
            "lbu",
            "lhu",
            "sb",
            "sh",
            "sw",
            "seld",
            "vload",
            "vstore",
        }
        return uop.insn.mnemonic not in mem_ops

    def reset(self) -> None:
        self.cycle = 0
        self._complete_count = 0
        # variables
        self._pending_completion_uop: Uop | None = None
        # logging variables
        self._total_instructions = 0
        self._busy_cycles = 0

    def tick(self, uop: Uop | None) -> None:
        self.cycle += 1
        # Log deferred completions from last cycle
        if self._pending_completion_uop is not None:
            self.logger.log_stage_end(
                self._pending_completion_uop.id,
                "E",
                lane=self.lane_id,
                cycle=self.cycle,
            )
            self.logger.log_retire(self._pending_completion_uop.id)
            self._pending_completion_uop = None

        # reset cycle states
        self._complete_count = 0

        # Accept new instruction
        if uop is not None:
            assert uop.insn.exu == EXU.SCALAR, "Attempted to pass non-scalar args to Scalar Excution Unit."
            # tag instruction with execution delay
            uop.execute_delay = 1
            self._pending_completion_uop = uop
            self._total_instructions += 1
            # Log: start execute
            self.logger.log_stage_start(
                uop.id,
                "E",
                lane=self.lane_id,
                cycle=self.cycle,
            )

            self._busy_cycles += uop.insn.mnemonic != "delay"
            self._complete_count = 1
            # execute the instruction and modify the arch state
            uop.insn.exec(self.arch_state)

    def flush_completions(self) -> None:
        """Flush any pending completions (call at end of simulation)."""
        if self._pending_completion_uop is not None:
            self.logger.log_stage_end(
                self._pending_completion_uop.id,
                "E",
                lane=self.lane_id,
                cycle=self.cycle,
            )
            self.logger.log_retire(self._pending_completion_uop.id)
            self._pending_completion_uop = None

    @property
    def total_instructions(self) -> int:
        """Total instructions executed."""
        return self._total_instructions

    @property
    def busy_cycles(self) -> int:
        """Number of cycles the EXU was busy."""
        return self._busy_cycles

    @property
    def complete_count(self) -> int:
        """Number of instructions completed this cycle."""
        return self._complete_count

    @property
    def has_in_flight(self) -> bool:
        return False