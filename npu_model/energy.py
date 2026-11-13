"""Technology-parameterized energy estimation from simulator activity."""
from dataclasses import dataclass, field
from collections import Counter
from typing import Any, Callable
from typing import Mapping


@dataclass
class EnergyParameters:
    """Energy costs supplied by characterization or PDK analysis.

    Dynamic models are keyed by op kind and return joules for one occurrence.
    """

    clock_period_seconds: float = 1.0
    dynamic_joules_per_activity: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class EnergyEstimator:
    """Evaluate recorded energy ops and compute average dynamic power."""

    parameters: EnergyParameters

    def estimate_ops(self, ops, cycles: int) -> dict:
        """Evaluate a stream of EnergyOps with named analytical models.

        Models are registered by operation kind and accept the op's parameter
        mapping, returning joules. Missing models are retained as unmodeled
        counts so instrumentation can be developed before characterization.
        """
        if cycles < 0:
            raise ValueError("cycles must be non-negative")
        evaluated = Counter()
        energy_by_kind: dict[str, float] = {}
        for op in ops:
            evaluated[op.kind] += 1
            model = self.parameters.dynamic_joules_per_activity.get(op.kind)
            if callable(model):
                energy_by_kind[op.kind] = energy_by_kind.get(op.kind, 0.0) + float(model(op.parameters))
        energy = sum(energy_by_kind.values())
        elapsed = cycles * self.parameters.clock_period_seconds
        return {"cycles": cycles, "elapsed_seconds": elapsed,
                "average_power_w": energy / elapsed if elapsed else 0.0,
                "op_counts": dict(evaluated),
                "energy_by_kind_joules": energy_by_kind,
                "unmodeled_ops": sorted(kind for kind in evaluated if kind not in energy_by_kind),
                "total_dynamic_joules": energy}

@dataclass(frozen=True)
class EnergyOp:
    """One modeled hardware operation emitted by a dynamic instruction."""

    kind: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


class EnergyOpLog:
    """Per-Uop operation log; op payloads deliberately contain no timestamps."""

    def __init__(self) -> None:
        self._ops: list[EnergyOp] = []

    def record(self, kind: str, **parameters: Any) -> EnergyOp:
        op = EnergyOp(kind, dict(parameters))
        self._ops.append(op)
        return op

    @property
    def ops(self) -> tuple[EnergyOp, ...]:
        return tuple(self._ops)

    def __iter__(self):
        return iter(self._ops)


CSR_DBG1 = 0xC11
"""Characterization programs write it at the start and end of their window."""


@dataclass
class MeasurementWindow:
    """What issued between a characterization program's two CSR_DBG1 writes
    (``npu_model/configs/programs/energy``)."""

    uops: list = field(default_factory=list)
    """Uops issued strictly between the two writes, in issue order."""
    inspected: list = field(default_factory=list)
    """``inspect(uop, arch_state)`` for each of ``uops``, at its issue."""
    idle: list[bool] = field(default_factory=list)
    """For each CSR_DBG1 write: every unit was idle when it issued."""
    halt_reason: str | None = None


def measure_window(program: Any, config: Any, max_cycles: int,
                   inspect: Callable[[Any, Any], Any] | None = None) -> MeasurementWindow:
    """Run ``program`` on ``config`` and record its measurement window."""
    import contextlib
    import io
    import tempfile
    from pathlib import Path

    from npu_model.logging import LoggerConfig
    from npu_model.simulation import Simulation

    window = MeasurementWindow()
    quiet = contextlib.redirect_stdout(io.StringIO())
    with tempfile.TemporaryDirectory() as scratch, quiet:
        sim = Simulation(hardware_config=config, program=program,
                         logger_config=LoggerConfig(filename=str(Path(scratch) / "trace.json")))
        core = sim.core
        tick = core.tick

        def traced_tick() -> None:
            tick()
            uop = core.issued_uop
            if uop is None:
                return
            if uop.insn.mnemonic == "csrrw" and int(uop.insn.imm) == CSR_DBG1:
                window.idle.append(not any(exu.has_in_flight for exu in core.exus))
            elif len(window.idle) == 1:
                window.uops.append(uop)
                window.inspected.append(inspect(uop, core.arch_state) if inspect else None)

        core.tick = traced_tick
        try:
            sim.run(max_cycles=max_cycles)
            window.halt_reason = core.arch_state.halt_reason
        finally:
            sim.close()
    return window
