import io
from collections import Counter

from npu_model.energy import EnergyEstimator, EnergyOp, EnergyOpLog, EnergyParameters
import pytest


def test_energy_op_log_keeps_kind_and_parameters_without_timestamps():
    log = EnergyOpLog()
    recorded = log.record("dma.beat", direction="read", bytes=32)

    assert recorded == EnergyOp("dma.beat", {"direction": "read", "bytes": 32})
    assert log.ops == (recorded,)
    assert not hasattr(recorded, "cycle")


def test_estimator_applies_registered_model_and_reports_unknown_ops():
    estimator = EnergyEstimator(EnergyParameters(
        clock_period_seconds=2e-9,
        dynamic_joules_per_activity={
            "instruction_fetch": lambda params: 1e-12,
            "dma.beat": lambda params: params["bytes"] * 2e-15,
        },
    ))
    result = estimator.estimate_ops([
        EnergyOp("instruction_fetch"),
        EnergyOp("dma.beat", {"bytes": 32}),
        EnergyOp("vpu.add"),
    ], cycles=100)

    assert result["op_counts"] == {"instruction_fetch": 1, "dma.beat": 1, "vpu.add": 1}
    assert result["unmodeled_ops"] == ["vpu.add"]
    assert result["total_dynamic_joules"] == 1e-12 + 64e-15
    assert result["average_power_w"] == pytest.approx(result["total_dynamic_joules"] / 200e-9)


def _issued_ops(source: str) -> dict[str, list[list[EnergyOp]]]:
    """Mnemonic -> the ops of each issued occurrence, in issue order."""
    from npu_model.configs.hardware import DefaultHardwareConfig
    from npu_model.util.converter import input_to_program
    from tests.helpers import run_simulation

    sim = run_simulation(input_to_program(io.StringIO(source)), DefaultHardwareConfig(), max_cycles=1000)
    issued: dict[str, list[list[EnergyOp]]] = {}
    for uop in sim._energy_uops:
        issued.setdefault(uop.insn.mnemonic, []).append(list(uop.energy_ops))
    return issued


def _census(ops: list[EnergyOp], prefix: str = "") -> Counter:
    """Ops (except instruction_fetch) whose kind starts with ``prefix``, by kind and parameters."""
    return Counter((op.kind, tuple(sorted(op.parameters.items()))) for op in ops
                   if op.kind != "instruction_fetch" and op.kind.startswith(prefix))


def _op(kind: str, count: int = 1, **parameters) -> Counter:
    return Counter({(kind, tuple(sorted(parameters.items()))): count})


def test_lsu_records_its_command_and_every_vmem_and_mrf_access():
    ops = _issued_ops("""
        li     x8, 0x40004       # bank 1 (256 KiB banks)
        sw     x8, 0(x8)
        sb     x8, 1(x8)
        lw     x9, 0(x8)
        delay  3
        seld   e1, 0(x8)
        delay  3
        vload  m0, 0(x0)         # VMEM 0x0000, bank 0
        delay  34
        vstore m0, 8(x0)         # VMEM 0x0400, bank 0
        delay  34
    """)
    for load in ("lw", "seld"):
        assert _census(ops[load][0]) == (
            _op("lsu.scalar", mnemonic=load) + _op("vmem.read", port="scalar", bank=1, bytes=32))
    for store, enabled in (("sw", 4), ("sb", 1)):
        assert _census(ops[store][0]) == (
            _op("lsu.scalar", mnemonic=store)
            + _op("vmem.write", port="scalar", bank=1, bytes=32, mask_bytes=enabled))
    assert _census(ops["vload"][0]) == (
        _op("lsu.vector", mnemonic="vload", rows=32, beats=32)
        + _op("vmem.read", 32, port="vector", bank=0, bytes=32)
        + _op("mrf.write", 32, port="lsu", bank=0, bytes=32))
    assert _census(ops["vstore"][0]) == (
        _op("lsu.vector", mnemonic="vstore", rows=32, beats=32)
        + _op("vmem.write", 32, port="vector", bank=0, bytes=32, mask_bytes=32)
        + _op("mrf.read", 32, port="lsu", bank=0, bytes=32))


def test_every_mrf_client_records_each_physical_row_access():
    ops = _issued_ops("""
        vli.all   m2, 0x3f80
        delay     66
        vadd.bf16 m4, m2, m6
        delay     66
        vadd.bf16 m8, m2, m2      # both operands read the same row: one physical read
        delay     66
        vtrpose.xlu m10, m2
        delay     66
        vmatpush.weight.mxu0 w0, m2
        delay     33
        vmatpop.bf16.acc.mxu1 m12, acc0
        delay     33
        vli.all   m34, 0x3f80     # m34 and m35 live in physical banks 2 and 3
        delay     66
    """)

    def rows(kind, port, *banks):
        return sum((_op(kind, 32, port=port, bank=bank, bytes=32) for bank in banks), Counter())

    assert _census(ops["vli.all"][0], "mrf.") == rows("mrf.write", "vpu", 2, 3)
    assert _census(ops["vadd.bf16"][0], "mrf.") == (
        rows("mrf.read", "vpu", 2, 3, 6, 7) + rows("mrf.write", "vpu", 4, 5))
    assert _census(ops["vadd.bf16"][1], "mrf.") == (
        rows("mrf.read", "vpu", 2, 3) + rows("mrf.write", "vpu", 8, 9))
    assert _census(ops["vtrpose.xlu"][0], "mrf.") == (
        rows("mrf.read", "xlu", 2) + rows("mrf.write", "xlu", 10))
    assert _census(ops["vmatpush.weight.mxu0"][0], "mrf.") == rows("mrf.read", "mxu0", 2)
    assert _census(ops["vmatpop.bf16.acc.mxu1"][0], "mrf.") == rows("mrf.write", "mxu1", 12, 13)
    assert _census(ops["vli.all"][1], "mrf.") == rows("mrf.write", "vpu", 2, 3)


def test_dma_records_its_vmem_line_accesses_as_vmem_ops():
    ops = _issued_ops("""
        addi  x30, x0, 1
        dma.config x30, ch0
        li    x10, 0x10000        # VMEM word 0x10000 = byte 0x40000, bank 1
        addi  x11, x0, 0          # DRAM offset 0
        addi  x12, x0, 64         # two 32-byte beats
        dma.store x11, x10, x12, ch0
        dma.wait ch0
        dma.load  x10, x11, x12, ch1
        dma.wait ch1
    """)
    assert _census(ops["dma.store"][0]) == (
        _op("dma.beat", direction="write", bytes=32, index=0)
        + _op("dma.beat", direction="write", bytes=32, index=1)
        + _op("vmem.read", 2, port="dma", bank=1, bytes=32))
    assert _census(ops["dma.load"][0]) == (
        _op("dma.beat", direction="read", bytes=32, index=0)
        + _op("dma.beat", direction="read", bytes=32, index=1)
        + _op("vmem.write", 2, port="dma", bank=1, bytes=32, mask_bytes=32))
