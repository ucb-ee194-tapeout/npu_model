"""Technology-parameterized energy estimation from simulator activity."""
from dataclasses import dataclass, field
from typing import Mapping


@dataclass
class EnergyParameters:
    """Energy costs supplied by characterization or PDK analysis.

    Static power is keyed by unit name. Dynamic costs are keyed by activity
    name and expressed in joules per activity unit (for example PE-cycle).
    """

    clock_period_seconds: float = 1.0
    static_power_w: Mapping[str, float] = field(default_factory=dict)
    dynamic_joules_per_activity: Mapping[str, float] = field(default_factory=dict)


@dataclass
class EnergyEstimator:
    """Convert cycle counts and activity totals into an energy breakdown."""

    parameters: EnergyParameters

    def estimate(self, cycles: int, activity: Mapping[str, Mapping[str, float]]) -> dict:
        """Estimate static and dynamic energy from per-unit activity totals.

        ``activity`` maps unit names to named accumulated quantities. A
        ``pe_cycles`` quantity can represent the sum of active PEs each cycle.
        """
        if cycles < 0:
            raise ValueError("cycles must be non-negative")
        period = self.parameters.clock_period_seconds
        if period <= 0:
            raise ValueError("clock_period_seconds must be positive")

        units = {}
        static_total = dynamic_total = 0.0
        names = set(activity) | set(self.parameters.static_power_w)
        for unit in sorted(names):
            static = self.parameters.static_power_w.get(unit, 0.0) * cycles * period
            details = {}
            dynamic = 0.0
            for metric, amount in activity.get(unit, {}).items():
                cost = self.parameters.dynamic_joules_per_activity.get(metric, 0.0)
                energy = amount * cost
                details[metric] = {"activity": amount, "energy_joules": energy}
                dynamic += energy
            units[unit] = {"static_joules": static, "dynamic_joules": dynamic,
                           "total_joules": static + dynamic, "activity": details}
            static_total += static
            dynamic_total += dynamic
        return {"clock_period_seconds": period, "cycles": cycles, "units": units,
                "static_joules": static_total, "dynamic_joules": dynamic_total,
                "total_joules": static_total + dynamic_total}
