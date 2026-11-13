"""Energy-characterization programs: one instruction, repeated on random operands.

``asm/energy_<mnemonic>.S`` is written by ``scripts/gen_energy_programs.py``,
which documents the layout. Each file runs ``@ENERGY_OP`` ``@ENERGY_REPS``
times between two CSR_DBG1 writes; RTL power (Cadence Joules) over that window,
with the EnergyOps the model attributes to it, gives energy per op.

The checked-in set is for ``DefaultHardwareConfig``; every file registers a
``Energy<Mnemonic>Program`` (``vmatmul.acc.mxu0`` -> ``EnergyVmatmulAccMxu0Program``).
Sets the generator writes for other configs (``--hardware``, ``--param``,
``--spacing``) load with ``load_programs`` and are not registered. Assembly is
parsed on instantiation, so importing the registry stays cheap.
``npu_model.energy.measure_window`` runs a program and returns its window.
"""
from __future__ import annotations

from pathlib import Path

from npu_model.software.program import Program
from npu_model.util.converter import load_asm

ASM = Path(__file__).resolve().parent / "asm"


def directives(source: Path) -> dict[str, str]:
    """``# @KEY value`` header lines of an assembly file."""
    found = {}
    for line in source.read_text().splitlines():
        if line.startswith("# @"):
            key, _, value = line[3:].partition(" ")
            found[key] = value.strip()
    return found


class _EnergyProgram(Program):
    source: Path
    target: str
    """Mnemonic under test."""
    reps: int
    """Executions of ``target`` inside the window."""
    kernel_max_cycles: int
    hardware: str
    """The hardware config class the program was generated for."""
    spacing: str
    """"isolated" (no repetition overlaps the previous) or "dense" (each issues
    at the earliest cycle the issue rules allow)."""

    def __init__(self) -> None:
        self.instructions = load_asm(self.source)


def _class_name(mnemonic: str) -> str:
    return "Energy" + "".join(part.capitalize() for part in mnemonic.split(".")) + "Program"


def load_programs(directory: Path, module: str = __name__) -> dict[str, type[_EnergyProgram]]:
    """Mnemonic -> program class for every ``energy_*.S`` in ``directory``."""
    programs = {}
    for source in sorted(Path(directory).glob("energy_*.S")):
        found = directives(source)
        target = found["ENERGY_OP"]
        rows, row_bytes = (int(v) for v in found["TILE"].split())
        programs[target] = type(_class_name(target), (_EnergyProgram,), {
            "__module__": module,
            "__doc__": f"{found['ENERGY_REPS']} x {target} on random operands ({source.name}).",
            "source": source,
            "target": target,
            "reps": int(found["ENERGY_REPS"]),
            "kernel_max_cycles": int(found["TIMEOUT"]),
            "tile": (rows, row_bytes),
            "hardware": found["HARDWARE"],
            "spacing": found["SPACING"],
        })
    return programs


PROGRAMS: dict[str, type[_EnergyProgram]] = load_programs(ASM)
"""Mnemonic -> program class of the checked-in (``DefaultHardwareConfig``) set."""
for _program in PROGRAMS.values():
    globals()[_program.__name__] = _program
