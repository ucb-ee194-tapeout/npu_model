import io
import itertools
import tempfile
from contextlib import nullcontext, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Callable
import gc

import torch

from npu_model.isa import UJType
from npu_model.isa_types import Bundled, Imm20, Imm21
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation


_ACTIVE_SIMULATIONS: list[Simulation] = []


def cleanup_tracked_simulations() -> None:
    while _ACTIVE_SIMULATIONS:
        sim = _ACTIVE_SIMULATIONS.pop()
        sim.close()
    gc.collect()


def run_simulation(
    program,
    hardware_config,
    *,
    max_cycles: int,
    verbose: bool = False,
    ignore_runtime_errors: bool = False,
    before_run: Callable[[Simulation], None] | None = None,
    randomize_init: bool = False,
    init_seed: int = 42,
    record_timeline: bool = False,
) -> Simulation:
    simulation_hardware_config = hardware_config
    if randomize_init:
        simulation_hardware_config = replace(
            hardware_config,
            arch_state_config=replace(
                hardware_config.arch_state_config,
                randomize_init=True,
                init_seed=init_seed,
            ),
        )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as handle:
        trace_path = handle.name

    try:
        sim = Simulation(
            hardware_config=simulation_hardware_config,
            logger_config=LoggerConfig(filename=trace_path),
            program=program,
            verbose=verbose,
            ignore_runtime_errors=ignore_runtime_errors,
            record_timeline=record_timeline,
        )
        _ACTIVE_SIMULATIONS.append(sim)
        if before_run is not None:
            before_run(sim)

        output_context = nullcontext() if verbose else redirect_stdout(io.StringIO())
        with output_context:
            sim.run(max_cycles=max_cycles)
        return sim
    finally:
        try:
            Path(trace_path).unlink(missing_ok=True)
        except (PermissionError, OSError):
            pass


def read_dram_tensor(
    sim: Simulation, base_addr: int, expected: torch.Tensor
) -> torch.Tensor:
    """Read a program-relative DRAM region (``Program.dram_base`` + offset)."""
    size = expected.numel() * expected.element_size()
    data = sim.core.arch_state.read_dram(sim.program.dram_base + base_addr, size)
    return data.view(expected.dtype).reshape(expected.shape).clone()


def operands(cls) -> list[tuple[str, type]]:
    named = [n for p in cls.params for n in ((p.reg, p.imm) if isinstance(p, Bundled) else (p,))]
    # jal's constructor takes the encoded immediate, twice the assembly word offset.
    return [(n.repr, Imm21 if issubclass(cls, UJType) and n.inner is Imm20 else n.inner) for n in named]


def corners(kind: type) -> list[int]:
    """Zero, both ends of the operand range, and the largest even value (branch immediates, BF16 tile
    registers) and multiple of 4 (FP32 tile registers)."""
    top = kind.upper_bound
    return sorted({v for v in (0, kind.lower_bound, top - 1, top - 2, top - 4) if not kind.lint(v)})


def words(cls) -> list[int]:
    """The instruction's words for every combination of operand corners."""
    ops = operands(cls)
    return [cls(**{name: v for (name, _), v in zip(ops, values)}).to_bytecode() & 0xFFFFFFFF
            for values in itertools.product(*(corners(kind) for _, kind in ops))]
