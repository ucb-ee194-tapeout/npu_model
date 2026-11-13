"""Geometry rules (npu_model/geometry.py): pinned at the default, enforced by the
timed units and the issue-rule engine alike at every supported size, and
spot-checked against the parameterized RTL (sp26-atlas-acc 6b48d36)."""
from io import StringIO

import pytest

import npu_model.configs.hardware as hardware_configs
import npu_model.configs.programs as program_configs
from npu_model.configs.hardware import DefaultHardwareConfig, Tile16HardwareConfig, Tile64HardwareConfig
from npu_model.geometry import DEFAULT, MXU_MNEMONICS, VPU_OPS, Geometry, validate
from npu_model.hardware.config import apply_overrides, scaled_config
from npu_model.scheduling import op_from_instruction, profile
from npu_model.software.instruction import Uop
from npu_model.util.converter import stream_to_instrs
from tests.test_exec_units import GEOMETRIES, asm, make

# The 32 x 32 RTL's numbers (tests/rtl fixtures); the rules must reproduce them.
DEFAULT_VPU_LATENCIES = {
    **dict.fromkeys(["vadd.bf16", "vsub.bf16", "vmul.bf16", "vminimum.bf16", "vmaximum.bf16", "vmov",
                     "vrecip.bf16", "vexp.bf16", "vexp2.bf16", "vpack.bf16.fp8", "vrelu.bf16",
                     "vsin.bf16", "vcos.bf16", "vtanh.bf16", "vlog2.bf16", "vsqrt.bf16",
                     "vsquare.bf16", "vcube.bf16"], 66),
    **dict.fromkeys(["vredsum.bf16", "vredmin.bf16", "vredmax.bf16"], 130),
    "vredsum.row.bf16": 39, "vredmin.row.bf16": 34, "vredmax.row.bf16": 34,
    "vli.all": 65, "vli.row": 65, "vli.col": 33, "vli.one": 33, "vunpack.fp8.bf16": 67,
}


def test_default_geometry_reproduces_the_32_row_rtl_numbers():
    g = DEFAULT
    assert g.vpu_latencies() == DEFAULT_VPU_LATENCIES
    assert {name: g.vpu_read_last(name) for name in ("vadd.bf16", "vredsum.bf16", "vredsum.row.bf16",
                                                     "vunpack.fp8.bf16", "vli.all")} == {
        "vadd.bf16": 63, "vredsum.bf16": 127, "vredsum.row.bf16": 31, "vunpack.fp8.bf16": 31, "vli.all": -1}
    assert g.mxu_latencies() == {name: 95 if name.startswith("vmatmul") and name.endswith("0")
                                 else 35 if name.startswith("vmatmul") else 33 for name in MXU_MNEMONICS}
    assert (g.mxu_compute_first(0), g.mxu_compute_first(1), g.mxu0_wslot_drain) == (63, 3, 62)
    assert (g.mxu_port_cycles, g.mxu1_acc_push_gate, g.mxu_reservation_free) == (32, 32, 33)
    assert (g.transpose_cycles, g.xlu_write_first) == (66, 34)
    assert g.lsu_latencies() == {**dict.fromkeys(["lb", "lbu", "lh", "lhu", "lw", "seld"], 4),
                                 **dict.fromkeys(["sb", "sh", "sw"], 2), "vload": 35, "vstore": 35}
    assert Geometry.from_config(DefaultHardwareConfig()) == DEFAULT


def test_registries_load_every_module():
    """npu_model/util/importer.py skips modules that fail to import; a missing
    entry here is an import error somewhere."""
    assert sorted(hardware_configs.__all__) == [
        "DefaultHardwareConfig", "Tile16HardwareConfig", "Tile64HardwareConfig"]
    assert len(program_configs.__all__) == 177


@pytest.mark.parametrize("config", GEOMETRIES, ids=lambda config: config.name)
def test_supported_configs_validate(config):
    g = validate(config())
    assert (g.lanes_bf16, g.mxu_k, g.mxu_n) == (g.rows // 2, g.rows, g.rows)
    # WithAtlasTileDim leaves VMEM and the DMA alone.
    assert (g.vmem_line_bytes, g.dma_beat_bytes, g.vmem_size) == (32, 32, DEFAULT.vmem_size)


@pytest.mark.parametrize("change,message", [
    (dict(mrf_depth=16), "square"),
    (dict(mrf_depth=48, mrf_width=48, wb_width=48 * 48), "power of two"),
    (dict(mrf_depth=4, mrf_width=4, wb_width=16), "at least 8"),
    (dict(vmem_line_bytes=24, dma_beat_bytes=24), "VMEM line"),
    (dict(vmem_bank_bytes=512), "fit in a VMEM bank"),
    (dict(wb_width=512), "weight slot"),
    (dict(num_m_registers=66), "tensor registers"),
    (dict(dma_beat_bytes=64), "DMA beat"),
    (dict(dma_num_channels=9), "channel field"),
])
def test_inconsistent_configs_are_rejected(change, message):
    config = scaled_config(DefaultHardwareConfig, 32)
    apply_overrides(config, change)
    with pytest.raises(ValueError, match=message):
        validate(config)


def test_scaled_config_does_not_touch_the_shared_defaults():
    before = DefaultHardwareConfig.arch_state_config
    config = scaled_config(DefaultHardwareConfig, 64, vmem_size=2 * 1024 * 1024)
    config.execution_units.pop("DMA0")
    assert DefaultHardwareConfig.arch_state_config is before and "DMA0" in DefaultHardwareConfig.execution_units
    assert (config.arch_state_config.mrf_depth, config.arch_state_config.vmem_size, config.dma_beat_bytes) == (
        64, 2 * 1024 * 1024, 32)


UNIT_CASES = ([("VPU", name) for name in sorted(VPU_OPS)]
              + [(f"MXU{name[-1]}", name) for name in sorted(MXU_MNEMONICS)]
              + [("XLU", "vtrpose.xlu"), ("LSU", "vload m20, 32(x2)"), ("LSU", "vstore m0, 32(x2)")])


@pytest.mark.parametrize("config", GEOMETRIES, ids=lambda config: config.name)
@pytest.mark.parametrize("unit,name", UNIT_CASES)
def test_timed_units_move_the_rows_the_issue_rules_describe(unit, name, config):
    """Every MREG row and VMEM line the timed unit reads or writes, at the issue
    offset it does so, is exactly the rule engine's profile, and the unit
    finishes at its latency."""
    source = name if unit == "LSU" else asm(name)
    exu = make(unit, "pytorch", config)
    insn = stream_to_instrs(StringIO(source))[0]
    g = exu.geometry
    rows = profile(op_from_instruction(insn, exu.arch_state.xrf, geometry=g), g).rows
    expected = {(r.resource[1], r.row, r.at, r.write) for r in rows if r.resource[0] == "mrf"}
    expected_lines = {(r.row * g.vmem_line_bytes, r.at, r.write) for r in rows if r.resource[0] == "vmem"}

    seen, lines = [], []
    checker = exu.arch_state.conflict_checker
    original, record_vmem = checker.access_mreg, exu._record_vmem

    def record(cycle, reg, row, write, owner):
        original(cycle, reg, row, write, owner)
        seen.append((reg, row, cycle - 1, write))  # issued on the unit's first tick, cycle 1

    def record_line(uop, direction, port, address, mask_bytes=None):
        record_vmem(uop, direction, port, address, mask_bytes)
        lines.append((address, exu.cycle - 1, direction == "write"))

    checker.access_mreg = record
    exu._record_vmem = record_line
    exu.tick(Uop(insn))
    while exu.has_in_flight:
        exu.tick(None)
    assert set(seen) == expected and len(seen) == len(expected)
    assert set(lines) == expected_lines and len(lines) == len(expected_lines)
    mnemonic = insn.mnemonic
    latency = (g.vpu_latency(mnemonic) if unit == "VPU" else g.mxu_latency(mnemonic) if unit.startswith("MXU")
               else g.transpose_cycles if unit == "XLU" else g.lsu_latency(mnemonic))
    assert exu.cycle == latency


def test_rules_scale_as_the_rtl_does():
    """Spot values at T = 16 and 64, written out so a change to a rule is visible."""
    small, large = Geometry.from_config(Tile16HardwareConfig()), Geometry.from_config(Tile64HardwareConfig())
    assert [g.vpu_latency("vadd.bf16") for g in (small, large)] == [34, 130]
    assert [g.vpu_latency("vredsum.bf16") for g in (small, large)] == [66, 258]
    # AddSubSumVec: log2(N / 2) adder levels.
    assert [g.vpu_latency("vredsum.row.bf16") for g in (small, large)] == [22, 72]
    assert [g.mxu_latency("vmatmul.mxu0") for g in (small, large)] == [47, 191]
    assert [g.mxu_latency("vmatmul.mxu1") for g in (small, large)] == [19, 67]
    assert [g.mxu_latency("vmatpush.weight.mxu0") for g in (small, large)] == [17, 65]
    assert [g.mxu0_wslot_drain for g in (small, large)] == [30, 126]
    assert [g.transpose_cycles for g in (small, large)] == [34, 130]
    # 32-byte lines: 16 x 16 moves two rows per line read, 64 x 64 two lines per row.
    assert [g.vls_beats for g in (small, large)] == [16, 128]
    assert [g.vls_latency for g in (small, large)] == [19, 131]
