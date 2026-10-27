"""The speed of light is a lower bound that follows the cycle model."""
import io
import json
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

import pytest

import npu_model.configs.programs as program_configs
from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.hardware import BankConflictError, Core
from npu_model.hardware import mxu, vpu, xlu
from npu_model.hardware.lsu import LSU_OP_LATENCIES
from npu_model.isa_types import EXU
from npu_model.logging.logger import Logger
from npu_model.software.instruction import Uop
from npu_model.software.program import InstantiableProgram
from npu_model.util.converter import stream_to_instrs
from npu_performance.speed_of_light import configure_memory, main, mxu_cycles, speed_of_light, vpu_cycles


MEMORIES = {
    "curve": ("curve", {}),
    # A fast fixed link, which leaves programs compute-bound.
    "fast-fixed": ("fixed", {"latency": 43, "cycles_per_beat": 1}),
}


def assemble(source: str):
    return stream_to_instrs(io.StringIO(source))


def config(memory: str) -> DefaultHardwareConfig:
    backend, options = MEMORIES[memory]
    hardware = DefaultHardwareConfig()
    configure_memory(hardware, backend, **options)
    return hardware


def test_speed_of_light_never_exceeds_measured_cycles() -> None:
    # DRAM-bound on both links, and VPU-bound on the fast one.
    for name in ("GemmaMlpProgram", "SmolVLAGeluTanhProgram"):
        for memory in MEMORIES:
            report = speed_of_light(getattr(program_configs, name)(), config(memory))
            assert 0 < report.bound.cycles <= report.measured, (name, memory)


def legal_after(unit: EXU, first: str, second: str, gap: int) -> bool:
    """Whether ``unit`` accepts ``second`` ``gap`` cycles after ``first``."""
    core = Core(DefaultHardwareConfig(), Mock(spec=Logger))
    try:
        engine = core.exu_map[unit]
        engine.tick(Uop(assemble(first)[0]))
        for _ in range(gap - 1):
            engine.tick(None)
        try:
            engine.tick(Uop(assemble(second)[0]))
        except (RuntimeError, BankConflictError):
            return False
        return True
    finally:
        core.close()


def test_each_charge_is_when_the_unit_accepts_the_next_command() -> None:
    """Two independent copies of a command issue exactly their charge apart."""
    cases = [
        (EXU.MATRIX_SYSTOLIC, "vmatmul.mxu0 acc0, m0, w0", "vmatmul.mxu0 acc1, m2, w1",
         mxu_cycles(Counter({"vmatmul.mxu0": 1}))),
        (EXU.MATRIX_INNER, "vmatpush.weight.mxu1 w0, m0", "vmatpush.weight.mxu1 w1, m2",
         mxu_cycles(Counter({"vmatpush.weight.mxu1": 1}))),
        (EXU.VECTOR, "vadd.bf16 m4, m0, m2", "vadd.bf16 m10, m6, m8", vpu_cycles(Counter({"vadd.bf16": 1}))),
        (EXU.VECTOR, "vexp.bf16 m4, m0", "vexp.bf16 m10, m6", vpu_cycles(Counter({"vexp.bf16": 1}))),
        (EXU.XLU, "vtrpose.xlu m1, m0", "vtrpose.xlu m3, m2", xlu.TRANSPOSE_CYCLES),
        (EXU.LSU, "vload m0, 0(x0)", "vload m1, 256(x0)", LSU_OP_LATENCIES["vload"]),
    ]
    for unit, first, second, charge in cases:
        assert not legal_after(unit, first, second, charge - 1), first
        assert legal_after(unit, first, second, charge), first


def test_charges_combine_across_ports_and_slots() -> None:
    port = mxu.PORT_CYCLES
    # Weight pushes take read port 1 alongside computes on port 0...
    assert mxu_cycles(Counter({"vmatmul.mxu0": 4, "vmatpush.weight.mxu0": 4})) == 4 * port
    # ...but a BF16 push holds both read ports.
    assert mxu_cycles(Counter({"vmatmul.mxu1": 2, "vmatpush.acc.bf16.mxu1": 2})) == 4 * port
    exp, recip = vpu.issue_cycles("vexp.bf16"), vpu.issue_cycles("vrecip.bf16")
    # Different lane logic shares the two VPU slots; the same logic runs alone.
    assert vpu_cycles(Counter({"vexp.bf16": 2, "vrecip.bf16": 2})) == 2 * max(exp, recip)
    assert vpu_cycles(Counter({"vexp.bf16": 2, "vexp2.bf16": 2})) == 4 * exp


def test_dram_bound_is_the_memory_ports_beat_spacing() -> None:
    program = InstantiableProgram(assemble("""
        addi x30, x0, 1
        dma.config x30, ch0
        addi x2, x0, 1024
        addi x4, x0, 256
        addi x5, x0, 512
        addi x6, x0, 768
        dma.load x0, x0, x2, ch0
        dma.load x4, x2, x2, ch1
        dma.load x5, x2, x2, ch2
        dma.store x2, x6, x2, ch3
        dma.wait ch0
        dma.wait ch1
        dma.wait ch2
        dma.wait ch3
    """))
    reports = {memory: speed_of_light(program, config(memory)) for memory in MEMORIES}
    for report in reports.values():
        assert (report.bound.name, report.dram_bytes) == ("DRAM", 4096)
        assert report.bound.cycles <= report.measured
    # 128 beats, one a cycle: 127 cycles from the first to the last.
    assert reports["fast-fixed"].bound.cycles == 127


def test_command_line_selects_the_memory_backend(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["MatmulProgram", "--memory", "fixed", "--latency", "43", "--cycles-per-beat", "1", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["memory"].startswith("fixed: 43-cycle latency")
    assert report["speed_of_light"] == max(resource["cycles"] for resource in report["resources"])
    assert main(["MatmulProgram", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["memory"].startswith("curve ee290sim_vcs_probe.json")
    with pytest.raises(SystemExit):
        main(["MatmulProgram", "--memory", "fixed", "--curve", "ee290sim_vcs_probe.json"])


@pytest.mark.parametrize("enforce_peak", [True, False])
def test_curve_bound_is_independent_of_dma_transfer_order(enforce_peak: bool) -> None:
    hardware = DefaultHardwareConfig()
    hardware.dma_memory_params = {
        "curves": {
            "measuredChannels": 1, "accessBytes": 32, "frequencyGHz": 0.5,
            "curves": {"0": [[2000, 10]], "50": [[8000, 10]], "100": [[4000, 10]]},
        },
        "enforce_peak": enforce_peak,
    }
    work = [(False, 0), (True, 1), (False, 2), (False, 3), (False, 4)]
    reports = []
    for order in (work, sorted(work)):
        lines = ["addi x30, x0, 1", "dma.config x30, ch0", "addi x2, x0, 256",
                 "addi x3, x0, 1024", "li x4, 0x10000"]
        # Loads and stores use disjoint VMEM and off-chip addresses.
        for store, channel in order:
            lines.append(f"dma.store x3, x4, x2, ch{channel}" if store
                         else f"dma.load x0, x0, x2, ch{channel}")
        lines += [f"dma.wait ch{channel}" for channel in range(5)]
        reports.append(speed_of_light(InstantiableProgram(assemble("\n".join(lines))), hardware))
    for report in reports:
        assert report.dram_bytes == 1280
        # 40 beats, 39 intervals at the fastest (50% read) peak of 0.5 beats/cycle.
        assert report.dram_cycles == (78 if enforce_peak else 0)
        assert report.bound.cycles <= min(run.measured for run in reports)


def test_command_line_rejects_outstanding_work(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "unfinished.S"
    path.write_text("vtrpose.xlu m1, m0\necall\n")
    with pytest.raises(SystemExit) as failure:
        main([str(path)])
    assert failure.value.code == 2
    output = capsys.readouterr()
    assert "program halted with outstanding work in: XLU" in output.err
    assert output.out == ""


def test_command_line_accepts_an_empty_program(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "empty.S"
    path.write_text("")
    assert main([str(path)]) == 0
    output = capsys.readouterr()
    assert "0.0%" in output.out
    assert "Speed of light: 0 cycles" in output.out
    assert "Measured      : 0 cycles" in output.out
    assert output.err == ""
