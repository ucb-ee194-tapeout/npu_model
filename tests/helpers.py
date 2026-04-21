import io
import tempfile
from contextlib import nullcontext, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Callable
import gc

import torch

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
    size = expected.numel() * expected.element_size()
    data = sim.core.arch_state.read_dram(base_addr, size)
    return data.view(expected.dtype).reshape(expected.shape).clone()
