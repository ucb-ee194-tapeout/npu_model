"""Roofline: Atlas's throughput ceilings, and where programs run beneath them.

The ceilings follow from the hardware config alone, through the units' own
definitions, so they do not depend on any program:

- compute: each MXU finishes a matmul every ``mxu.PORT_CYCLES`` cycles, two
  FLOPs per FP8 multiply-accumulate. The VPU sustains the element rate of its
  fastest mix of BF16 arithmetic instructions across ``vpu.ISSUE_SLOTS``,
  charged as the speed of light charges them, one operation per element;
- bandwidth: DRAM delivers the memory backend's fastest beat rate across all
  read/write mixes (the curve backend's peak when pacing is enabled, or one
  beat every ``cycles_per_beat`` on the fixed link). VMEM delivers a register
  every ``lsu.LSU_OP_LATENCIES`` cycles on each of the VLOAD and VSTORE paths.

A program is a point: its arithmetic intensity, the work it did per byte
moved at a level of the hierarchy, and the work per cycle it achieved on the
cycle model. At each level it can go no faster than the compute ceiling or
intensity x bandwidth, whichever is lower; the intensity where the two meet
is that level's ridge point. Intensity is also reported against compulsory
traffic, the program's preloaded inputs and golden output, which no schedule
avoids.

MXU FLOPs and VPU operations have separate rooflines, since they are
different work on different engines. Moves and format conversions (``vmov``,
``vli``, ``vpack``, ``vunpack``) count no operations.

Usage:
    uv run python -m npu_performance.roofline [PROGRAM ...] [options]

PROGRAM is a program class from ``npu_model.configs.programs`` or a ``.S``
file. With no program, only the machine's ceilings and ridge points are
printed. ``--plot`` draws the rooflines with every program on them.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy
import torch
from tabulate import tabulate

from npu_model.hardware import mxu, vpu
from npu_model.hardware.config import HardwareConfig
from npu_model.hardware.lsu import LSU_OP_LATENCIES
from npu_model.hardware.memory_backend import CurveMemoryBackend, FixedLatencyBackend, make_memory_backend
from npu_model.hardware.mxu import MatrixExecutionUnitSystolic
from npu_model.isa_types import EXU
from npu_model.software.program import Program
from npu_performance.speed_of_light import (
    MXUS,
    add_hardware_arguments,
    configured_mxus,
    describe_memory,
    execute,
    hardware_from_arguments,
    load_program,
    matmul_flops,
    vpu_cycles,
)

ENGINES = ("MXU", "VPU")
WORK_UNITS = {"MXU": "FLOP", "VPU": "op"}

VPU_DATA_MOVEMENT = {"vmov", "vpack.bf16.fp8", "vunpack.fp8.bf16", "vli.all", "vli.row", "vli.col", "vli.one"}
"""VPU instructions that move or convert data, which count no operations."""
VPU_ARITHMETIC = sorted(set(vpu.VPU_OP_LATENCIES) - VPU_DATA_MOVEMENT)

COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#000000")
"""Okabe-Ito colors, one per program in order; markers change after the last."""
MARKERS = ("o", "s", "^", "D")
INK = "#333333"
MUTED = "#777777"


@dataclass
class Ceilings:
    """Peak rates of the configured machine, per cycle."""

    mxus: int
    mxu_flops: float
    """FLOPs per cycle with every MXU computing."""
    vpu_ops: float
    """BF16 element operations per cycle."""
    dram: float
    """Bytes per cycle between DRAM and VMEM."""
    vmem: float
    """Bytes per cycle between VMEM and registers, VLOAD and VSTORE together."""

    def peak(self, engine: str) -> float:
        return self.mxu_flops if engine == "MXU" else self.vpu_ops


@dataclass
class Point:
    """One program's work, traffic and time on the cycle model."""

    program: str
    cycles: int
    mxu_flops: int
    vpu_ops: int
    dram_bytes: int
    """Bytes the DMA engine moved between DRAM and VMEM."""
    vmem_bytes: int
    """Bytes VLOAD and VSTORE moved between VMEM and registers."""
    compulsory_bytes: int
    """Preloaded inputs plus the golden output; 0 when the program declares neither."""

    def work(self, engine: str) -> int:
        return self.mxu_flops if engine == "MXU" else self.vpu_ops


def register_bytes(config: HardwareConfig) -> int:
    arch = config.arch_state_config
    return arch.mrf_depth * arch.mrf_width


def vpu_elements(config: HardwareConfig) -> int:
    """BF16 elements in a register pair, the tile a VPU instruction works on."""
    pair_bytes = 2 * register_bytes(config)
    return pair_bytes // torch.bfloat16.itemsize


def dram_bandwidth(config: HardwareConfig) -> float:
    """Bytes per cycle the memory backend delivers at its fastest read/write mix."""
    backend = make_memory_backend(config)
    # Without pacing, only the DMA engine's one beat a cycle limits it.
    beats_per_cycle = 1.0
    if isinstance(backend, CurveMemoryBackend) and backend.enforce_peak:
        beats_per_cycle = backend.peak_bandwidth
    elif isinstance(backend, FixedLatencyBackend):
        beats_per_cycle = 1 / backend.cycles_per_beat
    return beats_per_cycle * config.dma_beat_bytes


def vpu_peak(config: HardwareConfig) -> float:
    """Element operations per cycle of the fastest mix of arithmetic
    instructions, one per issue slot, as the speed of light charges them."""
    elements = vpu_elements(config)
    best = 0.0
    for mix in itertools.combinations_with_replacement(VPU_ARITHMETIC, vpu.ISSUE_SLOTS):
        rate = len(mix) * elements / vpu_cycles(Counter(mix))
        best = max(best, rate)
    return best


def ceilings(config: HardwareConfig) -> Ceilings:
    mxus = len(configured_mxus(config))
    register = register_bytes(config)
    return Ceilings(
        mxus=mxus,
        mxu_flops=mxus * matmul_flops(config) / mxu.PORT_CYCLES,
        vpu_ops=vpu_peak(config),
        dram=dram_bandwidth(config),
        vmem=register / LSU_OP_LATENCIES["vload"] + register / LSU_OP_LATENCIES["vstore"],
    )


def compulsory_bytes(program: Program) -> int:
    """Bytes of the program's preloaded inputs and its golden output."""
    total = 0
    for _, data in program.memory_regions:
        total += data.numel() * data.element_size()
    golden = getattr(program, "golden_result", None)
    if golden is not None:
        _, data = golden
        total += data.numel() * data.element_size()
    return total


def measure(label: str, program: Program, config: HardwareConfig, max_cycles: int = 1_000_000) -> Point:
    """Run ``program`` on the cycle model and count its work and traffic."""
    run = execute(program, config, max_cycles)
    matmuls = 0
    vector_instructions = 0
    register_moves = 0
    for (unit, mnemonic), count in run.launches.items():
        if unit in MXUS and MatrixExecutionUnitSystolic._kind(mnemonic) == "compute":
            matmuls += count
        elif unit == EXU.VECTOR and mnemonic in VPU_ARITHMETIC:
            vector_instructions += count
        elif unit == EXU.LSU and mnemonic in ("vload", "vstore"):
            register_moves += count
    return Point(
        program=label,
        cycles=run.cycles,
        mxu_flops=matmuls * matmul_flops(config),
        vpu_ops=vector_instructions * vpu_elements(config),
        dram_bytes=(run.load_beats + run.store_beats) * config.dma_beat_bytes,
        vmem_bytes=register_moves * register_bytes(config),
        compulsory_bytes=compulsory_bytes(program),
    )


def intensity(work: int, traffic: int) -> float | None:
    """Work per byte, or None when nothing moved."""
    if traffic == 0:
        return None
    return work / traffic


def roof(machine: Ceilings, point: Point, engine: str) -> tuple[float, str]:
    """The most work per cycle ``engine`` could do with the point's traffic,
    and the ceiling that sets it."""
    work = point.work(engine)
    limits = [(machine.peak(engine), engine)]
    if point.dram_bytes:
        limits.append((work / point.dram_bytes * machine.dram, "DRAM"))
    if point.vmem_bytes:
        limits.append((work / point.vmem_bytes * machine.vmem, "VMEM"))
    return min(limits)


def rate_unit(unit: str, clock_ghz: float | None) -> str:
    if clock_ghz is None:
        return f"{unit}/cycle"
    return f"G{unit}/s"


def format_number(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 100:
        return f"{value:,.0f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3g}"


def format_ceilings(memory: str, machine: Ceilings, clock_ghz: float | None) -> str:
    scale = clock_ghz or 1.0
    bandwidth = rate_unit("B", clock_ghz)
    rows = []
    for engine in ENGINES:
        unit = WORK_UNITS[engine]
        name = f"MXU x{machine.mxus}" if engine == "MXU" else engine
        peak = machine.peak(engine)
        rows.append([name, f"{format_number(peak * scale)} {rate_unit(unit, clock_ghz)}",
                     f"{format_number(peak / machine.dram)} {unit}/B",
                     f"{format_number(peak / machine.vmem)} {unit}/B"])
    table = tabulate(rows, headers=["Engine", "Peak", "Ridge, DRAM", "Ridge, VMEM"],
                     colalign=("left", "right", "right", "right"), disable_numparse=True)
    lines = [
        f"Memory   : {memory}",
        f"Bandwidth: DRAM {format_number(machine.dram * scale)} {bandwidth}, "
        f"VMEM {format_number(machine.vmem * scale)} {bandwidth} (VLOAD and VSTORE)",
        "",
        table,
    ]
    return "\n".join(lines)


def format_points(machine: Ceilings, points: list[Point], clock_ghz: float | None) -> str:
    scale = clock_ghz or 1.0
    rows = []
    for point in points:
        for engine in ENGINES:
            work = point.work(engine)
            if work == 0:
                continue
            achieved = work / point.cycles
            best, bound = roof(machine, point, engine)
            rows.append([
                point.program,
                engine,
                f"{work:,} {WORK_UNITS[engine]}",
                format_number(intensity(work, point.dram_bytes)),
                format_number(intensity(work, point.vmem_bytes)),
                format_number(intensity(work, point.compulsory_bytes)),
                format_number(achieved * scale),
                format_number(best * scale),
                f"{achieved / best:.1%}",
                bound,
            ])
    if not rows:
        return "No program did MXU or VPU work."
    rate = "G per s" if clock_ghz else "per cycle"
    headers = ["Program", "Engine", "Work", "DRAM AI", "VMEM AI", "Compulsory AI",
               f"Achieved {rate}", f"Roof {rate}", "Of roof", "Bound"]
    return tabulate(rows, headers=headers, disable_numparse=True,
                    colalign=("left", "left", "right", "right", "right", "right", "right", "right", "right", "left"))


def program_style(index: int) -> tuple[str, str]:
    """Color and marker of the index-th program."""
    color = COLORS[index % len(COLORS)]
    marker = MARKERS[index // len(COLORS) % len(MARKERS)]
    return color, marker


def plot(path: Path, machine: Ceilings, points: list[Point], clock_ghz: float | None = None) -> None:
    """Draw each engine's roofline with every program that used the engine."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    scale = clock_ghz or 1.0
    figure, panels = plt.subplots(1, len(ENGINES), figsize=(13, 5.5))
    for panel, engine in zip(panels, ENGINES):
        unit = WORK_UNITS[engine]
        peak = machine.peak(engine) * scale
        intensities = [machine.peak(engine) / machine.dram, machine.peak(engine) / machine.vmem]
        for point in points:
            work = point.work(engine)
            for traffic in (point.dram_bytes, point.vmem_bytes):
                if work and traffic:
                    intensities.append(work / traffic)
        x = numpy.geomspace(min(intensities) / 10, max(intensities) * 10, 512)
        panel.plot(x, numpy.minimum(peak, x * machine.dram * scale), color=INK, linewidth=1.5)
        panel.plot(x, numpy.minimum(peak, x * machine.vmem * scale), color=INK, linewidth=1.5, linestyle="--")
        name = f"MXU x{machine.mxus}" if engine == "MXU" else engine
        panel.annotate(f"{name}: {format_number(peak)} {rate_unit(unit, clock_ghz)}", (x[-1], peak),
                       xytext=(0, 4), textcoords="offset points", ha="right", va="bottom", color=INK)

        for index, point in enumerate(points):
            work = point.work(engine)
            if work == 0:
                continue
            achieved = work / point.cycles * scale
            color, marker = program_style(index)
            if point.dram_bytes:
                panel.plot(work / point.dram_bytes, achieved, marker=marker, color=color, markersize=8)
            # A larger ring, so it circles the DRAM point when the intensities match.
            if point.vmem_bytes:
                panel.plot(work / point.vmem_bytes, achieved, marker=marker, markersize=13,
                           markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.5)

        panel.set_xscale("log")
        panel.set_yscale("log")
        panel.set_title(f"{engine} ({'FP8' if engine == 'MXU' else 'BF16'})", color=INK)
        panel.set_xlabel(f"Arithmetic intensity ({unit}/B)", color=INK)
        panel.set_ylabel(f"Performance ({rate_unit(unit, clock_ghz)})", color=INK)
        panel.grid(True, which="major", color="#dddddd", linewidth=0.6)
        panel.spines[["top", "right"]].set_visible(False)

    bandwidth = rate_unit("B", clock_ghz)
    handles = [
        Line2D([], [], color=INK, label=f"DRAM roof, {format_number(machine.dram * scale)} {bandwidth}"),
        Line2D([], [], color=INK, linestyle="--", label=f"VMEM roof, {format_number(machine.vmem * scale)} {bandwidth}"),
        Line2D([], [], color=MUTED, marker="o", linestyle="", label="DRAM intensity"),
        Line2D([], [], color=MUTED, marker="o", markerfacecolor="none", linestyle="", label="VMEM intensity"),
    ]
    for index, point in enumerate(points):
        color, marker = program_style(index)
        handles.append(Line2D([], [], color=color, marker=marker, linestyle="", label=point.program))
    columns = 4
    rows = math.ceil(len(handles) / columns)
    figure.legend(handles=handles, loc="lower center", ncol=columns, frameon=False)
    figure.tight_layout(rect=(0, 0.06 * rows, 1, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("programs", nargs="*", metavar="program", help="program class names or .S files")
    add_hardware_arguments(parser)
    parser.add_argument("--clock-ghz", type=float, help="report rates per second at this clock instead of per cycle")
    parser.add_argument("--plot", type=Path, help="write the rooflines to this image file")
    parser.add_argument("--json", action="store_true", help="print the ceilings and points as JSON, per cycle")
    args = parser.parse_args(argv)
    config = hardware_from_arguments(parser, args)
    programs = []
    for name in args.programs:
        try:
            programs.append((name, load_program(name)))
        except (OSError, ValueError) as error:
            parser.error(str(error))

    machine = ceilings(config)
    memory = describe_memory(config)
    points = []
    for name, program in programs:
        try:
            points.append(measure(name, program, config, args.max_cycles))
        except RuntimeError as error:
            parser.error(f"{name}: {error}")

    if args.json:
        summary = {"memory": memory, "ceilings": asdict(machine), "programs": []}
        for point in points:
            entry = asdict(point)
            for engine in ENGINES:
                if point.work(engine):
                    best, bound = roof(machine, point, engine)
                    entry[engine.lower()] = {"achieved": point.work(engine) / point.cycles,
                                             "roof": best, "bound": bound}
            summary["programs"].append(entry)
        print(json.dumps(summary, indent=2))
    else:
        print(format_ceilings(memory, machine, args.clock_ghz))
        if points:
            print()
            print(format_points(machine, points, args.clock_ghz))
    if args.plot is not None:
        plot(args.plot, machine, points, args.clock_ghz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
