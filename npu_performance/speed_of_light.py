"""Speed of light: the fewest cycles a program's work could take on Atlas.

The program runs once on the cycle model (``npu_model.simulation``), which
measures its cycles and records what S1 launched. Each resource that limits
throughput is then charged for that work by the simulator's own definitions,
so the bound follows the model as it changes:

- DRAM: the fixed backend's beat spacing or the curve backend's fastest peak
  across all read/write mixes, when pacing is enabled;
- each MXU: every command holds sequencer ports for ``mxu.PORT_CYCLES``
  (``mxu.PORT_CHOICES``), and pushes on one path write one at a time;
- the VPU: ``vpu.ISSUE_SLOTS`` issue slots, each instruction holding one, or
  both, for ``vpu.issue_cycles``; instructions that share lane logic never
  overlap;
- the XLU (``xlu.TRANSPOSE_CYCLES`` per transpose) and the LSU's VLOAD and
  VSTORE paths (``lsu.LSU_OP_LATENCIES``);
- S1, which launches one instruction per cycle.

No schedule of the same work finishes before its busiest resource does, so the
largest charge is the speed of light. Dependences and latencies are ignored,
so a short program, or one made of a single dependence chain, stays well above
it.

Usage:
    uv run python -m npu_performance.speed_of_light PROGRAM [options]

PROGRAM is a program class from ``npu_model.configs.programs`` or a ``.S``
file. The memory backend defaults to the measured Mess curve; ``--memory
fixed`` selects the fixed-latency link instead.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
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
from npu_model.hardware import DmaExecutionUnit, mxu, vpu, xlu
from npu_model.hardware.config import HardwareConfig
from npu_model.hardware.lsu import LSU_OP_LATENCIES
from npu_model.hardware.memory_backend import CurveMemoryBackend, FixedLatencyBackend, make_memory_backend
from npu_model.hardware.mxu import MatrixExecutionUnitInner, MatrixExecutionUnitSystolic
from npu_model.hardware.vpu import VectorExecutionUnit
from npu_model.isa_types import EXU
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation
from npu_model.software.program import InstantiableProgram, Program
from npu_model.util.converter import load_asm

MXUS = {EXU.MATRIX_SYSTOLIC: MatrixExecutionUnitSystolic, EXU.MATRIX_INNER: MatrixExecutionUnitInner}


@dataclass
class Execution:
    """What one run of the cycle model launched, and how long it took."""

    cycles: int
    launches: Counter[tuple[EXU, str]]
    load_beats: int
    store_beats: int
    dram_cycles: float
    """Schedule-independent DRAM spacing time."""


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
    """Run ``program`` on the cycle model, recording every launch and DMA beat."""
    sim = Simulation(config, LoggerConfig(filename=os.devnull), program, verbose=False)
    core = sim.core
    engine = next(unit for unit in core.exus if isinstance(unit, DmaExecutionUnit))
    launches: Counter[tuple[EXU, str]] = Counter()
    beats: Counter[str] = Counter()
    tick = core.tick

    def observed_tick() -> None:
        tick()
        uop = core.issued_uop
        if uop is not None:
            launches[uop.insn.exu, uop.insn.mnemonic] += 1
        # The engine records this cycle's beat, if any, as [source, address, is_store].
        beat = engine.last_cycle.a
        if beat is not None:
            is_store = beat[2]
            beats["store" if is_store else "load"] += 1

    core.tick = observed_tick  # type: ignore[method-assign]
    try:
        sim.run(max_cycles=max_cycles)
        if not core.is_finished():
            raise RuntimeError(f"program did not finish within {max_cycles} cycles")
        busy = [unit.name for unit in core.exus if unit.has_in_flight]
        if busy:
            raise RuntimeError("program halted with outstanding work in: " + ", ".join(busy))
        backend = engine.backend
        dram_cycles = backend.port_cycles
        if isinstance(backend, CurveMemoryBackend) and backend.enforce_peak:
            dram_cycles = max(0, beats.total() - 1) / backend.peak_bandwidth
        return Execution(sim.cycle_count, launches, beats["load"], beats["store"], dram_cycles)
    finally:
        sim.close()


def configured_mxus(config: HardwareConfig) -> list[EXU]:
    """The MXUs the hardware config instantiates."""
    return [exu for exu in MXUS if exu.value in config.execution_units.values()]


def matmul_flops(config: HardwareConfig) -> int:
    """FLOPs in one matmul: a rows x inner FP8 tile times an inner x columns
    weight tile, two per multiply-accumulate."""
    arch = config.arch_state_config
    rows, inner = arch.mrf_depth, arch.mrf_width
    columns = arch.wb_width // inner
    return 2 * rows * inner * columns


def command_kinds(launches: Counter[str]) -> Counter[str]:
    """MXU launches counted by ``mxu.PORT_CHOICES`` kind; both MXUs classify alike."""
    kinds: Counter[str] = Counter()
    for mnemonic, count in launches.items():
        kinds[MatrixExecutionUnitSystolic._kind(mnemonic)] += count
    return kinds


def mxu_cycles(launches: Counter[str]) -> int:
    """Sequencer port and push-path time for one MXU's commands."""
    kinds = command_kinds(launches)
    ports: set[int] = set()
    for choices in mxu.PORT_CHOICES.values():
        for choice in choices:
            ports.update(choice)

    # For each set of ports, add up how many of them every command must hold
    # (the fewest any of its choices takes) and spread that over the set.
    cycles = 0
    for size in range(1, len(ports) + 1):
        for group in itertools.combinations(sorted(ports), size):
            held = 0
            for kind, count in kinds.items():
                fewest = min(len(set(choice) & set(group)) for choice in mxu.PORT_CHOICES[kind])
                held += count * fewest
            cycles = max(cycles, math.ceil(held * mxu.PORT_CYCLES / size))

    # Pushes on one path write one at a time.
    pushes: Counter[str] = Counter()
    for kind, count in kinds.items():
        path = MatrixExecutionUnitSystolic._push_path(kind)
        if path is not None:
            pushes[path] += count
    for count in pushes.values():
        cycles = max(cycles, count * mxu.PORT_CYCLES)
    return cycles


def vpu_cycles(launches: Counter[str]) -> int:
    """Issue-slot time for the VPU's instructions."""
    both_slots = 0
    one_slot: dict[str, int] = {}
    for name, count in launches.items():
        cycles = vpu.issue_cycles(name) * count
        if VectorExecutionUnit._double(name):
            both_slots += cycles
        else:
            one_slot[name] = cycles

    # One-slot instructions pair up across the slots, except that those sharing
    # lane logic run one at a time. The logic groups do not overlap, so whatever
    # shares logic with one instruction also shares it with the rest.
    paired = math.ceil(sum(one_slot.values()) / vpu.ISSUE_SLOTS)
    serial = 0
    for name in one_slot:
        sharing = [other for other in one_slot if VectorExecutionUnit._share_logic(name, other)]
        serial = max(serial, sum(one_slot[other] for other in sharing))
    return both_slots + max(paired, serial)


def speed_of_light(program: Program, config: HardwareConfig, max_cycles: int = 1_000_000) -> SpeedOfLight:
    """Measure ``program`` on the cycle model and charge each resource for its work."""
    run = execute(program, config, max_cycles)
    by_unit: dict[EXU, Counter[str]] = {}
    for (unit, mnemonic), count in run.launches.items():
        by_unit.setdefault(unit, Counter())[mnemonic] += count

    loaded = run.load_beats * config.dma_beat_bytes
    stored = run.store_beats * config.dma_beat_bytes
    dram = math.ceil(run.dram_cycles)
    resources = [Resource("DRAM", f"{loaded:,} B loaded, {stored:,} B stored", dram)]

    flops_per_matmul = matmul_flops(config)
    matmuls = 0
    units = configured_mxus(config)
    for exu in units:
        unit = MXUS[exu]
        launches = by_unit.get(exu, Counter())
        kinds = command_kinds(launches)
        pushes = 0
        pops = 0
        for kind, count in kinds.items():
            if MatrixExecutionUnitSystolic._push_path(kind) is not None:
                pushes += count
            elif kind.startswith("pop"):
                pops += count
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

    return SpeedOfLight(
        measured=run.cycles,
        resources=resources,
        flops=matmuls * flops_per_matmul,
        peak_flops_per_cycle=len(units) * flops_per_matmul / mxu.PORT_CYCLES,
        dram_bytes=loaded + stored,
        dram_cycles=dram,
    )


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
        load_rate, _ = backend.curve_peak(100.0)
        store_rate, _ = backend.curve_peak(0.0)
        name = Path(backend.path).name if backend.path else "curve"
        return (f"curve {name}: {backend.lead_off_latency:.0f}-cycle lead-off, {1 / load_rate:.1f} cycles per "
                f"{beat} B beat loading, {1 / store_rate:.1f} storing")
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
    measured = report.measured or 1
    rows = []
    for resource in report.resources:
        share = resource.cycles / measured
        rows.append([resource.name, resource.work, f"{resource.cycles:,}", f"{share:.1%}"])
    table = tabulate(rows, headers=["Resource", "Work", "Cycles", "Of measured"],
                     colalign=("left", "left", "right", "right"))
    bound = report.bound
    lines = [
        f"Program : {label}",
        f"Memory  : {memory}",
        "",
        table,
        "",
        f"Speed of light: {bound.cycles:,} cycles, bound by {bound.name}",
        f"Measured      : {report.measured:,} cycles, running at {bound.cycles / measured:.1%} of it",
    ]
    if report.flops and report.dram_cycles:
        intensity = report.flops / report.dram_bytes
        bandwidth = report.dram_bytes / report.dram_cycles
        ridge = report.peak_flops_per_cycle / bandwidth
        side = "memory-bound" if intensity < ridge else "compute-bound"
        lines.append(f"Roofline      : {intensity:,.1f} FLOP/B against a ridge point of {ridge:,.0f} FLOP/B "
                     f"({report.peak_flops_per_cycle:,.0f} FLOP/cycle peak, {bandwidth:.2f} B/cycle from DRAM), "
                     f"so {side}")
    return "\n".join(lines)


def add_hardware_arguments(parser: argparse.ArgumentParser) -> None:
    """Options that select the hardware config, its DMA memory backend and the
    simulation's cycle limit."""
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


def hardware_from_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> HardwareConfig:
    """The hardware config that ``add_hardware_arguments``'s options select."""
    if args.memory == "curve" and (args.latency is not None or args.cycles_per_beat is not None):
        parser.error("--latency and --cycles-per-beat apply to --memory fixed")
    if args.memory == "fixed" and args.curve is not None:
        parser.error("--curve applies to --memory curve")

    hardware = getattr(hardware_configs, args.hardware, None)
    if hardware is None:
        parser.error(f"unknown hardware config {args.hardware!r}")
    config = hardware()
    configure_memory(config, args.memory, curve=args.curve, latency=args.latency,
                     cycles_per_beat=args.cycles_per_beat)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("program", help="program class name or .S file")
    add_hardware_arguments(parser)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    config = hardware_from_arguments(parser, args)
    try:
        program = load_program(args.program)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    memory = describe_memory(config)
    try:
        report = speed_of_light(program, config, args.max_cycles)
    except RuntimeError as error:
        parser.error(str(error))

    if args.json:
        summary = {"program": args.program, "memory": memory, **asdict(report),
                   "speed_of_light": report.bound.cycles, "bound": report.bound.name}
        print(json.dumps(summary, indent=2))
    else:
        print(format_report(args.program, memory, report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
