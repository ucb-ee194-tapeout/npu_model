import pytest
import torch

import npu_model.configs.programs as program_configs

from npu_model.configs.isa_definition import *  # noqa: F401, F403
from tests.helpers import read_dram_tensor, run_simulation


PROGRAM_NAMES = sorted(getattr(program_configs, "__all__", []))


def test_program_registry_is_not_empty() -> None:
    assert PROGRAM_NAMES, (
        "No programs were discovered in npu_model.configs.programs.__all__. "
        "This would cause the parametrized program execution test to collect zero cases, "
        "which can hide import or registration failures."
    )


@pytest.mark.parametrize("program_name", PROGRAM_NAMES)
def test_registered_program_executes(
    program_name: str,
    hardware_config_cls,
    max_cycles: int,
    sim_verbose: bool,
) -> None:
    program_cls = getattr(program_configs, program_name)
    program = program_cls()
    sim = run_simulation(
        program,
        hardware_config_cls(),
        max_cycles=max_cycles,
        verbose=sim_verbose,
    )

    if not getattr(program, "golden_result", None):
        return

    output_base, golden_tensor = program.golden_result
    actual = read_dram_tensor(sim, output_base, golden_tensor)
    rtol, atol = getattr(program, "kernel_tolerance", (1e-2, 1e-2))

    if not torch.allclose(actual.float(), golden_tensor.float(), rtol=rtol, atol=atol):
        diff = (actual.float() - golden_tensor.float()).abs().max()
        pytest.fail(
            f"{program_name} golden check failed: max diff = {diff:.6f}.\n"
            f"Result = {actual.float()}\nExpected = {golden_tensor}"
        )
