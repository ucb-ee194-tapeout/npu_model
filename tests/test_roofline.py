"""The roofline's ceilings follow the units, and programs run beneath them."""
import json
from pathlib import Path

import pytest

import npu_model.configs.programs as program_configs
from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_performance.roofline import ENGINES, ceilings, main, measure, roof
from tests.test_speed_of_light import MEMORIES, config


def test_ceilings_follow_the_unit_definitions() -> None:
    machine = ceilings(DefaultHardwareConfig())
    # Two MXUs, each finishing a 32x32x32 FP8 matmul every 32 cycles.
    assert machine.mxu_flops == 2 * 65_536 / 32
    # Two different one-slot instructions, each holding a slot for 65 cycles.
    assert machine.vpu_ops == 2 * 1024 / 65
    # A 1 KiB register every 35 cycles on each of the VLOAD and VSTORE paths.
    assert machine.vmem == 2 * 1024 / 35
    # The curve's fastest peak, storing: a 32 B beat every 21.67 cycles.
    assert machine.dram == pytest.approx(32 / 21.67, rel=1e-3)
    assert ceilings(config("fast-fixed")).dram == 32


def test_programs_run_beneath_their_roof() -> None:
    # MXU work, VPU work, and both.
    for name in ("SmolVLAMatmulProgram", "SmolVLASoftmaxProgram", "GemmaMlpProgram"):
        for memory in MEMORIES:
            hardware = config(memory)
            machine = ceilings(hardware)
            point = measure(name, getattr(program_configs, name)(), hardware)
            for engine in ENGINES:
                if point.work(engine):
                    best, _ = roof(machine, point, engine)
                    assert 0 < point.work(engine) / point.cycles <= best, (name, memory, engine)


def test_command_line_reports_and_plots(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image = tmp_path / "roofline.png"
    assert main(["SmolVLAMatmulProgram", "--json", "--plot", str(image)]) == 0
    report = json.loads(capsys.readouterr().out)
    (point,) = report["programs"]
    # One 32x32x32 matmul over 2 KiB of FP8 operands and a 2 KiB BF16 result.
    assert (point["mxu_flops"], point["dram_bytes"], point["compulsory_bytes"]) == (65_536, 4096, 4096)
    assert point["mxu"]["bound"] == "DRAM"
    assert "vpu" not in point
    assert image.stat().st_size > 0
    assert main([]) == 0
    assert "Ridge, DRAM" in capsys.readouterr().out
