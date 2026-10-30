"""Shared performance-counter schema and event tagging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .logger import Logger


@dataclass(frozen=True)
class CounterSpec:
    name: str
    unit: str
    description: str


class PerformanceCounters:
    """Accumulate RTL-named PMU events and tag instruction-owned events."""

    def __init__(self, specs: list[CounterSpec], logger: Logger | None = None) -> None:
        self.specs = {spec.name: spec for spec in specs}
        if len(self.specs) != len(specs):
            raise ValueError("Performance counter names must be unique")
        self.values = {spec.name: 0 for spec in specs}
        self.logger = logger
        self.enabled = True

    def begin_cycle(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def add(
        self,
        name: str,
        amount: int = 1,
        *,
        instruction_id: int | None = None,
        lane: int = 0,
        cycle: int | None = None,
    ) -> None:
        if name not in self.specs:
            raise KeyError(f"Unregistered performance counter: {name}")
        if amount < 0:
            raise ValueError("Counter increments must be nonnegative")
        if not amount or not self.enabled:
            return
        self.values[name] += amount
        if self.logger is not None:
            self.logger.log_counter_event(
                instruction_id,
                name,
                amount,
                self.specs[name].unit,
                lane=lane,
                cycle=cycle,
            )

    def reset(self) -> None:
        for name in self.values:
            self.values[name] = 0

    def snapshot(self) -> dict[str, dict[str, int | str]]:
        return {
            name: {
                "value": self.values[name],
                "unit": spec.unit,
                "description": spec.description,
            }
            for name, spec in self.specs.items()
        }


def atlas_counter_specs(num_mreg_banks: int, num_vmem_banks: int) -> list[CounterSpec]:
    """The PMU event schema declared by AtlasCore.scala."""
    rows = [
        ("core.run_cycles", "cycles", "Cycles while Atlas execution is enabled"),
        ("dma.active_cycles", "cycles", "Cycles with at least one DMA channel active"),
        ("dma.commands", "commands", "DMA commands issued by the scalar core"),
        ("dma.beats_issued", "beats", "VMEM beats issued by DMA"),
        ("dma.beats_completed", "beats", "VMEM beats completed by DMA"),
        ("dma.vmem_stall_cycles", "cycles", "Cycles DMA is stalled by VMEM"),
        ("dma.saturated_cycles", "cycles", "Cycles DMA is at its configured in-flight limit"),
        ("dma.limit_blocked_cycles", "cycles", "Cycles an otherwise ready DMA request is blocked by the in-flight limit"),
        ("dma.outstanding_cycles", "cycles", "Cycles with at least one DMA beat outstanding"),
        ("dma.response_wait_beat_cycles", "beat-cycles", "Outstanding DMA beats summed across cycles while awaiting responses"),
        ("dma.fence_wait_cycles", "cycles", "Cycles the scalar core waits for a DMA fence"),
        ("vpu.active_cycles", "cycles", "Cycles the vector processing unit is active"),
        ("vpu.commands", "commands", "VPU commands issued by the scalar core"),
        ("mxu0.compute_cycles", "cycles", "Cycles MXU 0 performs computation"),
        ("mxu0.data_cycles", "cycles", "Cycles MXU 0 is active in its data path"),
        ("mxu0.commands", "commands", "MXU 0 commands issued by the scalar core"),
        ("mxu1.compute_cycles", "cycles", "Cycles MXU 1 performs computation"),
        ("mxu1.data_cycles", "cycles", "Cycles MXU 1 is active in its data path"),
        ("mxu1.commands", "commands", "MXU 1 commands issued by the scalar core"),
        ("xlu.commands", "commands", "XLU transpose commands issued by the scalar core"),
        ("xlu.active_cycles", "cycles", "Cycles the XLU is processing a transpose"),
        ("xlu.mreg_write_rows", "rows", "Transposed rows written by the XLU to MREG"),
        ("lsu.scalar_loads", "operations", "Scalar loads issued to VMEM"),
        ("lsu.scalar_stores", "operations", "Scalar stores issued to VMEM"),
        ("lsu.scalar_load_completions", "operations", "Scalar load results returned by the LSU"),
        ("lsu.vload_commands", "commands", "VLOAD commands issued to the LSU"),
        ("lsu.vstore_commands", "commands", "VSTORE commands issued to the LSU"),
        ("lsu.vload_lines_issued", "lines", "VMEM lines requested by VLOAD"),
        ("lsu.vload_rows_written", "rows", "MREG rows written by VLOAD"),
        ("lsu.vstore_rows_read", "rows", "MREG rows requested by VSTORE"),
        ("lsu.vstore_lines_written", "lines", "VMEM lines written by VSTORE"),
        ("lsu.scalar_busy_cycles", "cycles", "Cycles a scalar load is in flight"),
        ("lsu.vload_active_cycles", "cycles", "Cycles the VLOAD path is active"),
        ("lsu.vstore_active_cycles", "cycles", "Cycles the VSTORE path is active"),
    ]
    for bank in range(num_mreg_banks):
        rows.extend([
            (f"mreg.bank_{bank:02d}.read_cycles", "cycles", f"Cycles MREG bank {bank} accepts a read"),
            (f"mreg.bank_{bank:02d}.write_cycles", "cycles", f"Cycles MREG bank {bank} accepts a write"),
            (f"mreg.bank_{bank:02d}.read_contended_cycles", "cycles", f"Cycles MREG bank {bank} has contended read requests"),
            (f"mreg.bank_{bank:02d}.write_contended_cycles", "cycles", f"Cycles MREG bank {bank} has contended write requests"),
        ])
    for bank in range(num_vmem_banks):
        rows.extend([
            (f"vmem.bank_{bank:02d}.read_cycles", "cycles", f"Cycles VMEM bank {bank} accepts a read"),
            (f"vmem.bank_{bank:02d}.write_cycles", "cycles", f"Cycles VMEM bank {bank} accepts a write"),
            (f"vmem.bank_{bank:02d}.contended_cycles", "cycles", f"Cycles VMEM bank {bank} has contending requests"),
        ])
    return [CounterSpec(*row) for row in rows]
