"""Speed of light: the fewest cycles a program's work could take on Atlas.

The program runs once on the cycle model (``npu_model.simulation``), which
measures its cycles and records what S1 launched. Each resource that limits
throughput is then charged for that work by the simulator's own definitions,
so the bound follows the model as it changes:

- DRAM: the program's DMA transfers, replayed on a fresh DMA engine and memory
  backend (``dma.py``, ``memory_backend.py``), each launched as soon as its
  channel is free, so the backend in use sets the time;
- each MXU: every command holds sequencer ports for ``mxu.PORT_CYCLES``
  (``mxu.PORT_CHOICES``), and pushes on one path write one at a time;
- the VPU: ``vpu.ISSUE_SLOTS`` issue slots, each instruction holding one, or
  both, for ``vpu.issue_cycles``; instructions that share lane logic never
  overlap;
- the XLU (``xlu.TRANSPOSE_CYCLES`` per transpose) and the LSU's VLOAD and
  VSTORE paths (``lsu.LSU_OP_LATENCIES``);
- S1, which launches one instruction per cycle.

No schedule of the same work finishes before its busiest resource does, so the
largest charge is the speed of light. Dependences are ignored, and of the
latencies only the DMA engine's count, so a short program, or one made of a
single dependence chain, stays well above it.

Usage:
    uv run python -m npu_speed_of_light.roofline PROGRAM [options]

PROGRAM is a program class from ``npu_model.configs.programs`` or a ``.S``
file. The memory backend defaults to the measured Mess curve; ``--memory
fixed`` selects the fixed-latency link instead.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from tabulate import tabulate

# Hardware must be imported before programs to avoid a circular import.
import npu_model.hardware  # noqa: F401
import npu_model.configs.hardware as hardware_configs
import npu_model.configs.programs as program_configs
from npu_model.hardware import Core, DmaExecutionUnit, mxu, vpu, xlu
from npu_model.hardware.config import HardwareConfig
from npu_model.hardware.lsu import LSU_OP_LATENCIES
from npu_model.hardware.memory_backend import CurveMemoryBackend, FixedLatencyBackend, make_memory_backend
from npu_model.isa import Instruction
from npu_model.isa_types import EXU
from npu_model.logging import Logger, LoggerConfig
from npu_model.simulation import Simulation
from npu_model.software.instruction import Uop
from npu_model.software.program import InstantiableProgram, Program
from npu_model.util.converter import load_asm

_MXUS = {EXU.MATRIX_SYSTOLIC: mxu.MatrixExecutionUnitSystolic, EXU.MATRIX_INNER: mxu.MatrixExecutionUnitInner}


@dataclass
class Transfer:
    """A DMA transfer as S1 launched it, with the scalar operands it read."""

    insn: Instruction
    xrf: dict[int, int]
    base: int

    @property
    def nbytes(self) -> int:
        return self.xrf[int(self.insn.rs2)]

    @property
    def is_store(self) -> bool:
        return self.insn.mnemonic == "dma.store"


@dataclass
class Execution:
    """What one run of the cycle model launched, and how long it took."""

    cycles: int
    launches: Counter[tuple[EXU, str]]
    transfers: list[Transfer]


@dataclass
class Resource:
    name: str
    work: str
    cycles: int


@dataclass
class SpeedOfLight:
    measured: int
    """Cycles the program took on the cycle model."""
    resources: list[Resource]
    flops: int
    """MXU work, two FLOPs per multiply-accumulate."""
    peak_flops_per_cycle: float
    dram_bytes: int
    dram_cycles: int

    @property
    def bound(self) -> Resource:
        return max(self.resources, key=lambda resource: resource.cycles)


def execute(program: Program, config: HardwareConfig, max_cycles: int) -> Execution:
    """Run ``program`` on the cycle model, recording every launch."""
    sim = Simulation(config, LoggerConfig(filename=os.devnull), program, verbose=False)
    core = sim.core
    launches: Counter[tuple[EXU, str]] = Counter()
    transfers: list[Transfer] = []
    tick = core.tick

    def observed_tick() -> None:
        state = core.arch_state
        # S1 reads scalar registers before this edge's writebacks land.
        xrf, base = list(state.xrf), state.base
        tick()
        if (uop := core.issued_uop) is None:
            return
        insn = uop.insn
        launches[insn.exu, insn.mnemonic] += 1
        if insn.mnemonic in ("dma.load", "dma.store"):
            registers = (int(insn.rd), int(insn.rs1), int(insn.rs2))
            transfers.append(Transfer(insn, {reg: xrf[reg] for reg in registers}, base))

    core.tick = observed_tick  # type: ignore[method-assign]
    try:
        sim.run(max_cycles=max_cycles)
        if not core.is_finished():
            raise RuntimeError(f"program did not finish within {max_cycles} cycles")
        return Execution(sim.cycle_count, launches, transfers)
    finally:
        sim.close()


def dram_cycles(config: HardwareConfig, transfers: list[Transfer]) -> int:
    """Cycles a fresh DMA engine needs to finish ``transfers`` by itself, each
    launched as soon as its channel is free."""
    if not transfers:
        return 0
    logger = Logger(LoggerConfig(filename=os.devnull))
    core = Core(config, logger)
    try:
        engine = next(unit for unit in core.exus if isinstance(unit, DmaExecutionUnit))
        state = core.arch_state
        for transfer in transfers:
            channel = int(transfer.insn.funct3)
            while state.check_flag(channel):
                engine.tick(None)
            for reg, value in transfer.xrf.items():
                if reg:
                    state.xrf[reg] = value
            state.base = transfer.base
            # S1 marks the channel busy as it launches the transfer.
            state.set_flag(channel)
            engine.tick(Uop(transfer.insn))
        while engine.has_in_flight:
            engine.tick(None)
        return engine.cycle
    finally:
        core.close()
        logger.close()


def mxu_cycles(launches: Counter[str]) -> int:
    """Sequencer port and push-path time for one MXU's commands."""
    kinds: Counter[str] = Counter()
    for mnemonic, count in launches.items():
        kinds[mxu.command_kind(mnemonic)] += count
    ports = sorted({port for choices in mxu.PORT_CHOICES.values() for choice in choices for port in choice})
    cycles = 0
    # Each command holds, of any set of ports, at least as many as the fewest
    # its choices take from that set.
    for size in range(1, len(ports) + 1):
        for group in itertools.combinations(ports, size):
            load = sum(count * min(len(set(choice) & set(group)) for choice in mxu.PORT_CHOICES[kind])
                       for kind, count in kinds.items())
            cycles = max(cycles, -(-load * mxu.PORT_CYCLES // size))
    paths: Counter[str] = Counter()
    for kind, count in kinds.items():
        if (path := mxu.push_path(kind)) is not None:
            paths[path] += count
    return max(cycles, mxu.PORT_CYCLES * max(paths.values(), default=0))


def vpu_cycles(launches: Counter[str]) -> int:
    """Issue-slot time for the VPU's instructions."""
    both = sum(vpu.issue_cycles(name) * count for name, count in launches.items() if vpu.takes_both_slots(name))
    single = {name: vpu.issue_cycles(name) * count for name, count in launches.items()
              if not vpu.takes_both_slots(name)}
    shared = 0
    for name in single:
        group = [other for other in single if vpu.shares_logic(name, other)]
        # Instructions that pairwise share lane logic run one at a time.
        if all(vpu.shares_logic(left, right) for left, right in itertools.combinations(group, 2)):
            shared = max(shared, sum(single[other] for other in group))
        else:
            shared = max(shared, single[name])
    return both + max(-(-sum(single.values()) // vpu.ISSUE_SLOTS), shared)


def speed_of_light(program: Program, config: HardwareConfig, max_cycles: int = 1_000_000) -> SpeedOfLight:
    """Measure ``program`` on the cycle model and charge each resource for its work."""
    run = execute(program, config, max_cycles)
    by_unit: dict[EXU, Counter[str]] = {}
    for (unit, mnemonic), count in run.launches.items():
        by_unit.setdefault(unit, Counter())[mnemonic] += count

    loaded = sum(transfer.nbytes for transfer in run.transfers if not transfer.is_store)
    stored = sum(transfer.nbytes for transfer in run.transfers if transfer.is_store)
    dram = dram_cycles(config, run.transfers)
    resources = [Resource("DRAM", f"{loaded:,} B loaded, {stored:,} B stored", dram)]

    arch = config.arch_state_config
    flops_per_matmul = 2 * arch.mrf_depth * arch.mrf_width * (arch.wb_width // arch.mrf_width)
    matmuls = 0
    units = 0
    for exu, unit in _MXUS.items():
        if exu.value not in config.execution_units.values():
            continue
        units += 1
        launches = by_unit.get(exu, Counter())
        kinds: Counter[str] = Counter()
        for mnemonic, count in launches.items():
            kinds[mxu.command_kind(mnemonic)] += count
        pushes = sum(count for kind, count in kinds.items() if mxu.push_path(kind) is not None)
        pops = sum(count for kind, count in kinds.items() if kind.startswith("pop"))
        matmuls += kinds["compute"]
        work = f"{kinds['compute']:,} matmuls, {pushes:,} pushes, {pops:,} pops"
        resources.append(Resource(unit.mxu.upper(), work, mxu_cycles(launches)))

    vector = by_unit.get(EXU.VECTOR, Counter())
    resources.append(Resource("VPU", f"{vector.total():,} instructions", vpu_cycles(vector)))
    transposes = by_unit.get(EXU.XLU, Counter()).total()
    resources.append(Resource("XLU", f"{transposes:,} transposes", xlu.TRANSPOSE_CYCLES * transposes))
    memory = by_unit.get(EXU.LSU, Counter())
    for path in ("vload", "vstore"):
        resources.append(Resource(f"{path.upper()} path", f"{memory[path]:,} {path}s",
                                  LSU_OP_LATENCIES[path] * memory[path]))
    launched = run.launches.total()
    resources.append(Resource("S1", f"{launched:,} instructions", launched))

    return SpeedOfLight(run.cycles, resources, matmuls * flops_per_matmul,
                        units * flops_per_matmul / mxu.PORT_CYCLES, loaded + stored, dram)


def configure_memory(config: HardwareConfig, memory: str, *, curve: str | None = None,
                     latency: int | None = None, cycles_per_beat: int | None = None) -> None:
    """Select the DMA memory backend; options left unset keep the config's or
    the backend's defaults."""
    params = dict(config.dma_memory_params) if config.dma_memory_backend == memory else {}
    if memory == "curve":
        params.setdefault("curves", HardwareConfig.dma_memory_params["curves"])
        if curve is not None:
            params["curves"] = curve
    else:
        if latency is not None:
            params["latency"] = latency
        if cycles_per_beat is not None:
            params["cycles_per_beat"] = cycles_per_beat
    config.dma_memory_backend = memory
    config.dma_memory_params = params


def describe_memory(config: HardwareConfig) -> str:
    backend = make_memory_backend(config)
    beat = config.dma_beat_bytes
    if isinstance(backend, CurveMemoryBackend):
        load, store = (1 / backend.curve_peak(read_pct)[0] for read_pct in (100.0, 0.0))
        name = Path(backend.path).name if backend.path else "curve"
        return (f"curve {name}: {backend.lead_off_latency:.0f}-cycle lead-off, {load:.1f} cycles per "
                f"{beat} B beat loading, {store:.1f} storing")
    if isinstance(backend, FixedLatencyBackend):
        every = "cycle" if backend.cycles_per_beat == 1 else f"{backend.cycles_per_beat} cycles"
        return f"fixed: {backend.latency}-cycle latency, one {beat} B beat every {every}"
    return type(backend).__name__


def load_program(name: str) -> Program:
    """A program class from ``npu_model.configs.programs``, or a ``.S`` file."""
    if name.endswith(".S"):
        return InstantiableProgram(load_asm(Path(name)))
    program = getattr(program_configs, name, None)
    if program is None:
        raise ValueError(f"unknown program {name!r}; give a .S file or one of: "
                         + ", ".join(sorted(getattr(program_configs, "__all__", []))))
    return program()


def format_report(label: str, memory: str, report: SpeedOfLight) -> str:
    rows = [[resource.name, resource.work, f"{resource.cycles:,}", f"{resource.cycles / report.measured:.1%}"]
            for resource in report.resources]
    bound = report.bound
    lines = [
        f"Program : {label}",
        f"Memory  : {memory}",
        "",
        tabulate(rows, headers=["Resource", "Work", "Cycles", "Of measured"], colalign=("left", "left", "right", "right")),
        "",
        f"Speed of light: {bound.cycles:,} cycles, bound by {bound.name}",
        f"Measured      : {report.measured:,} cycles, running at {bound.cycles / report.measured:.1%} of it",
    ]
    if report.flops and report.dram_bytes:
        intensity = report.flops / report.dram_bytes
        bandwidth = report.dram_bytes / report.dram_cycles
        ridge = report.peak_flops_per_cycle / bandwidth
        side = "memory-bound" if intensity < ridge else "compute-bound"
        lines.append(f"Roofline      : {intensity:,.1f} FLOP/B against a ridge point of {ridge:,.0f} FLOP/B "
                     f"({report.peak_flops_per_cycle:,.0f} FLOP/cycle peak, {bandwidth:.2f} B/cycle from DRAM), "
                     f"so {side}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("program", help="program class name or .S file")
    parser.add_argument("--hardware", default="DefaultHardwareConfig", help="HardwareConfig class name")
    parser.add_argument("--memory", choices=("curve", "fixed"), default="curve",
                        help="DMA memory backend (default: curve)")
    parser.add_argument("--curve", help="curve file, a path or a name in npu_model/configs/memory_curves "
                                        "(curve only; default: the hardware config's)")
    parser.add_argument("--latency", type=int,
                        help="cycles from a beat's request to its response (fixed only; default: the spec's link)")
    parser.add_argument("--cycles-per-beat", type=int,
                        help="minimum cycles between beats (fixed only; default: the spec's link)")
    parser.add_argument("--max-cycles", type=int, default=1_000_000)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    if args.memory == "curve" and (args.latency is not None or args.cycles_per_beat is not None):
        parser.error("--latency and --cycles-per-beat apply to --memory fixed")
    if args.memory == "fixed" and args.curve is not None:
        parser.error("--curve applies to --memory curve")

    hardware = getattr(hardware_configs, args.hardware, None)
    if hardware is None:
        parser.error(f"unknown hardware config {args.hardware!r}")
    try:
        program = load_program(args.program)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    config = hardware()
    configure_memory(config, args.memory, curve=args.curve, latency=args.latency,
                     cycles_per_beat=args.cycles_per_beat)
    memory = describe_memory(config)
    report = speed_of_light(program, config, args.max_cycles)

    if args.json:
        print(json.dumps({"program": args.program, "memory": memory, **asdict(report),
                          "speed_of_light": report.bound.cycles, "bound": report.bound.name}, indent=2))
    else:
        print(format_report(args.program, memory, report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
