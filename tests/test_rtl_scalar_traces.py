"""Replay cycle traces produced by the actual Chisel ScalarCore in Verilator.

See tests/rtl/README.md for the harness boundary and regeneration command.
"""
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.hardware.core import Core
from npu_model.logging import Logger, LoggerConfig
from npu_model.util.converter import input_to_program


_FIXTURES = Path(__file__).parent / "rtl"
_CASES = json.loads((_FIXTURES / "scalar_cases.json").read_text())


@pytest.mark.parametrize("name", _CASES)
def test_scalar_core_matches_recorded_rtl_cycle_trace(name: str, tmp_path: Path) -> None:
    scenario = _CASES[name]
    expected = json.loads((_FIXTURES / "scalar_traces.json").read_text())[name]
    program = input_to_program(io.StringIO(scenario["source"]))
    assert program.assemble() == [int(word, 16) for word in scenario["words"]]
    config = DefaultHardwareConfig()
    config.arch_state_config = replace(config.arch_state_config, dram_size=4096)
    logger = Logger(LoggerConfig(filename=str(tmp_path / "trace.json")))
    core = Core(config, logger)
    core.load_program(program)
    state = core.arch_state
    state.write_vmem(0, 0, torch.tensor([0x78, 0x56, 0x34, 0x12], dtype=torch.uint8))
    try:
        for row in expected:
            if row["cycle"] in scenario.get("dma_busy_cycles", []):
                state.set_flag(0)
            else:
                state.clear_flag(0)
            s1 = core.idu.uop or core.ifu.output.peek()
            assert state.pc == row["fetch_pc"], (name, row)
            if s1 is not None:
                assert s1.pc == row["s1_pc"], (name, row)
            if "csr_data" in row:
                assert s1 is not None
                assert state.read_xrf(s1.insn.rs1) == row["csr_data"], (name, row)
            if row["illegal"]:
                with pytest.raises(RuntimeError, match="Illegal control-flow instruction"):
                    core.tick()
                assert state.halted
                assert state.pc == row["next_pc"]
            else:
                core.tick()
                for field in ("cycle", "fetch_pc", "s1_fire", "halted", "next_pc"):
                    assert core.last_cycle[field] == row[field], (name, field, row, core.last_cycle)
        assert state.halted
    finally:
        core.close()
        logger.close()
