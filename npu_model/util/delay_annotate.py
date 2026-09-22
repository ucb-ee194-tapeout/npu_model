"""
Delay annotation: convert a kernel with no explicit timing into an equivalent
kernel with DELAY instructions inserted wherever the scoreboard (see
`hardware/scoreboard.py`) found a hazard, so it reproduces the same
hazard-free schedule when later run without live scoreboard tracking.
"""

import tempfile
from pathlib import Path

from ..configs.isa_definition import DELAY
from ..hardware.config import HardwareConfig
from ..logging import LoggerConfig
from ..simulation import Simulation
from ..software.program import InstantiableProgram, Program


def annotate_delays(
    program: Program, hardware_config: HardwareConfig, max_cycles: int = 10000
) -> InstantiableProgram:
    """
    Run `program` once in scoreboard mode and return an equivalent program
    with DELAY instructions inserted before every instruction the scoreboard
    had to stall.

    Assumes straight-line code: a static instruction executed more than once
    (e.g. inside a loop) only keeps the stall from its last dynamic instance.
    """
    index_by_insn = {id(insn): i for i, insn in enumerate(program.instructions)}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        trace_path = handle.name
    try:
        sim = Simulation(
            hardware_config=hardware_config,
            logger_config=LoggerConfig(filename=trace_path),
            program=program,
            verbose=False,
            schedule_mode=True,
        )
        sim.run(max_cycles=max_cycles)

        stalls_by_index: dict[int, int] = {}
        for uop, stall in sim.core.idu.schedule_stalls:
            index = index_by_insn[id(uop.insn)]
            stalls_by_index[index] = stalls_by_index.get(index, 0) + stall
        sim.close()
    finally:
        Path(trace_path).unlink(missing_ok=True)

    instructions = []
    for index, insn in enumerate(program.instructions):
        stall = stalls_by_index.get(index)
        if stall:
            instructions.append(DELAY(imm=stall))
        instructions.append(insn)
    return InstantiableProgram(instructions)
