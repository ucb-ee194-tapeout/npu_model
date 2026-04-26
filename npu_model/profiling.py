"""Execution unit utilization profiling utilities.

Two main features:
  - print_utilization_report: per-EXU busy/idle/utilization table (idea 1)
  - print_timeline: ASCII activity timeline + overlap stats (idea 2)
  - print_stats: both at once
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from npu_model.simulation import Simulation

BAR_WIDTH = 72


def print_utilization_report(sim: "Simulation") -> None:
    """Print a per-EXU utilization table."""
    stats = sim.get_stats()
    total = stats.cycles

    print(f"Utilization  ({total} total cycles, {stats.total_instructions} instructions, IPC {stats.ipc:.3f})")
    print(f"  {'Unit':<14} {'Busy':>7} {'Idle':>7} {'Util':>7} {'Insns':>7}")
    print(f"  {'-'*14} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for name, s in stats.exu_stats.items():
        idle = total - s.busy_cycles
        print(f"  {name:<14} {s.busy_cycles:>7} {idle:>7} {s.utilization:>6.1%} {s.instructions:>7}")
    print()


def print_timeline(sim: "Simulation") -> None:
    """Print an ASCII activity timeline and overlap statistics."""
    timeline = getattr(sim, "timeline", None)
    if not timeline:
        print("No timeline data. Run with record_timeline=True.")
        return

    exu_names = list(timeline[0].keys())
    total_cycles = len(timeline)
    cycles_per_char = total_cycles / BAR_WIDTH

    def _bar(name: str) -> str:
        chars = []
        for col in range(BAR_WIDTH):
            lo = col * total_cycles // BAR_WIDTH
            hi = max((col + 1) * total_cycles // BAR_WIDTH, lo + 1)
            active = any(timeline[c][name] for c in range(lo, min(hi, total_cycles)))
            chars.append("█" if active else "·")
        return "".join(chars)

    print(f"Timeline  (1 char ≈ {cycles_per_char:.1f} cycles  █=active  ·=idle)")
    print()
    for name in exu_names:
        busy_count = sum(1 for c in timeline if c[name])
        pct = busy_count / total_cycles if total_cycles else 0.0
        print(f"  {name:<12} │{_bar(name)}│ {pct:>5.1%}")
    print()

    # Concurrent unit counts per cycle
    active_counts = [sum(1 for n in exu_names if timeline[c][n]) for c in range(total_cycles)]
    overlap_2 = sum(1 for n in active_counts if n >= 2)
    overlap_3 = sum(1 for n in active_counts if n >= 3)
    peak = max(active_counts, default=0)

    print(f"  Peak concurrent units : {peak}")
    print(f"  ≥2 units simultaneous : {overlap_2:>6} / {total_cycles} cycles  ({overlap_2/total_cycles:.1%})")
    if overlap_3:
        print(f"  ≥3 units simultaneous : {overlap_3:>6} / {total_cycles} cycles  ({overlap_3/total_cycles:.1%})")

    # MXU vs VPU overlap specifically
    mxu_names = [n for n in exu_names if "mxu" in n.lower()]
    vpu_names = [n for n in exu_names if "vpu" in n.lower()]
    if mxu_names and vpu_names:
        mxu_active = [any(timeline[c][n] for n in mxu_names) for c in range(total_cycles)]
        vpu_active = [any(timeline[c][n] for n in vpu_names) for c in range(total_cycles)]
        both = sum(1 for m, v in zip(mxu_active, vpu_active) if m and v)
        print(f"  MXU+VPU overlap       : {both:>6} / {total_cycles} cycles  ({both/total_cycles:.1%})")
    print()


def print_stats(sim: "Simulation") -> None:
    """Print utilization report followed by timeline."""
    print_utilization_report(sim)
    print_timeline(sim)
