"""Energy-characterization programs (npu_model/configs/programs/energy).

Each program runs one instruction ``reps`` times on random operands between two
CSR_DBG1 writes. Cadence Joules gives the RTL energy of that window, and the
model gives the EnergyOps issued in it, so energy per op is only as good as the
agreement between the two. These tests check the model side:

* every instruction except the two halts has a program, and the checked-in
  programs match ``scripts/gen_energy_programs.py``;
* all of the following hold for every program set the generator writes: the
  checked-in 32 x 32 set and a 16 x 16 set (``Tile16HardwareConfig``) in full,
  and a 64 x 64 set (``Tile64HardwareConfig``) and dense spacing at 16 x 16 and
  32 x 32, where tensor commands issue as early as ``npu_model.scheduling``
  allows, one program per unit (every program with ``--full-energy``);
* each program runs without a structural hazard (the model raises on the ones
  the RTL leaves to software) and halts with ecall;
* each window starts and ends with every unit idle, so no setup or straggler
  activity lands in it, and holds only the instruction and timing filler;
* the window issues the instruction exactly ``reps`` times, on random operand
  data that differs between repetitions;
* every repetition records the same EnergyOps, so window energy can be
  attributed to them.
"""
import atexit
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest
import torch

import npu_model.configs.hardware as hardware_configs
from npu_model.configs.isa_definition import _vmem_address, _vmem_line_address
from npu_model.configs.programs.energy import ASM, PROGRAMS, load_programs
from npu_model.energy import measure_window
from npu_model.isa import IsaSpec
from npu_model.isa_patterns import (
    CSRReg,
    DMAConfig,
    ExponentOffsetLoad,
    ScalarBaseOffsetStore,
    ScalarBranchImm,
    ScalarComputeImm,
    ScalarComputeReg,
    ScalarComputeShamt,
    ScalarOffsetLoad,
)
from npu_model.isa_types import EXU, Accumulator, MatrixReg, WeightBuffer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import gen_energy_programs as generator  # noqa: E402

TARGETS = sorted(PROGRAMS)
SETS = {
    # name: (hardware config, spacing)
    "Default": ("DefaultHardwareConfig", "isolated"),
    "Tile16": ("Tile16HardwareConfig", "isolated"),
    "Tile64": ("Tile64HardwareConfig", "isolated"),
    "Default-dense": ("DefaultHardwareConfig", "dense"),
    "Tile16-dense": ("Tile16HardwareConfig", "dense"),
}
TENSOR_SAMPLE = ("vload", "vstore", "vadd.bf16", "vexp.bf16", "vredsum.bf16", "vredsum.row.bf16",
                 "vli.all", "vtrpose.xlu", "vmatpush.weight.mxu0", "vmatpush.acc.bf16.mxu1",
                 "vmatmul.mxu0", "vmatmul.acc.mxu1", "vmatpop.fp8.acc.mxu0")
"""One program per tensor unit and schedule shape."""
TILE64_SAMPLE = ("add", "lw", "sw", "dma.load") + TENSOR_SAMPLE
"""Every unit at 64 x 64: the full set takes minutes (``--full-energy``)."""
FILLER = {"delay", "dma.wait"}
"""Window instructions besides the one under test and the delay-slot nops."""
CLOSING = ("csrrs", "sub")
"""The cycle stamp that ends every window, just before the second CSR_DBG1 write."""
UNATTRIBUTED = {"dma.config", "dma.wait"}
"""The DMA unit records no EnergyOp for these yet, so their window energy has no op to land on."""


def config_of(set_name: str):
    return getattr(hardware_configs, SETS[set_name][0])()


@cache
def program_set(set_name: str) -> dict:
    """Mnemonic -> program class of ``set_name`` (generated into a temporary directory)."""
    if set_name == "Default":
        return PROGRAMS
    hardware, spacing = SETS[set_name]
    directory = Path(tempfile.mkdtemp(prefix=f"energy-{set_name}-"))
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    for name, text in generator.generate(config_of(set_name), spacing).items():
        (directory / name).write_text(text)
    return load_programs(directory)


def spaced(set_name: str) -> set[str]:
    """Programs whose delays depend on the spacing."""
    hardware, spacing = SETS[set_name]
    layout = generator.Layout.of(config_of(set_name), spacing)
    return {spec.mnemonic for spec in generator.specs(layout) if spec.spaced}


def cases(full: bool) -> list[tuple[str, str]]:
    out = [("Default", t) for t in TARGETS] + [("Tile16", t) for t in TARGETS]
    out += [("Tile64", t) for t in TARGETS if full or t in TILE64_SAMPLE]
    for name in ("Default-dense", "Tile16-dense"):
        # Spacing changes only the tensor programs; the rest match the isolated sets.
        out += [(name, t) for t in TARGETS if t in spaced(name) and (full or t in TENSOR_SAMPLE)]
    return out


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "case" not in metafunc.fixturenames:
        return
    expect_ops = metafunc.function.__name__ == "test_every_repetition_records_the_same_energy_ops"
    params = [pytest.param(case, id=f"{case[0]}-{case[1]}", marks=pytest.mark.xfail(
                  strict=True, reason="unit records no EnergyOp yet") if expect_ops and case[1] in UNATTRIBUTED else ())
              for case in cases(metafunc.config.getoption("full_energy"))]
    metafunc.parametrize("case", params)


@dataclass
class Window:
    target: str
    reps: int
    uops: list
    """Uops issued strictly between the two CSR_DBG1 writes."""
    reads: list[list[bytes] | None]
    """Per uop, the operand data an instance of ``target`` reads at issue."""
    idle: list[bool]
    """For each CSR_DBG1 write: every unit was idle when it issued."""
    halt_reason: str | None

    @property
    def body(self) -> list:
        """The window without its closing cycle stamp."""
        return self.uops[:-len(CLOSING)]

    @property
    def repetitions(self) -> list:
        return [uop for uop in self.body if uop.insn.mnemonic == self.target]

    @property
    def sources(self) -> list[list[bytes]]:
        return [read for read in self.reads[:-len(CLOSING)] if read is not None]


def _bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).flatten().numpy().tobytes()


_SCALAR_DATA = ((ScalarComputeReg, ("rs1", "rs2")), (ScalarBranchImm, ("rs1", "rs2")),
                (ScalarComputeImm, ("rs1",)), (ScalarComputeShamt, ("rs1",)), (CSRReg, ("rs1",)),
                (ScalarBaseOffsetStore, ("rs2",)), (DMAConfig, ("rs1",)))


def _sources(insn, state) -> list[bytes]:
    """The data ``insn`` reads: not addresses, sizes, scales or immediates."""
    for pattern, fields in _SCALAR_DATA:
        if isinstance(insn, pattern):
            return [state.read_xrf(int(getattr(insn, f))).to_bytes(4, "little") for f in fields]
    m = insn.mnemonic
    if isinstance(insn, (ScalarOffsetLoad, ExponentOffsetLoad)):
        return [_bytes(state.read_vmem(_vmem_address(state, insn.rs1, insn.imm) & ~3, 0, 4))]
    if m == "vload":
        tile = state.cfg.mrf_depth * state.cfg.mrf_width
        return [_bytes(state.read_vmem(_vmem_line_address(state, insn.rs1, insn.imm), 0, tile))]
    if m == "vstore":
        return [_bytes(state.read_mrf_u8(int(insn.vd)))]
    if m in ("dma.load", "dma.store"):
        length = state.read_xrf(insn.rs2)
        if m == "dma.load":
            return [_bytes(state.read_dram(state.dma_dram_address(state.read_xrf(insn.rs1)), length))]
        return [_bytes(state.read_vmem(state.dma_vmem_byte_address(state.read_xrf(insn.rs1)), 0, length))]
    unit = "mxu0" if insn.exu == EXU.MATRIX_SYSTOLIC else "mxu1"
    fields = ["vs1", "vs2"] + (["vd"] if m.startswith("vmatmul.acc") else [])
    data = []
    for field in fields:
        reg = getattr(insn, field, None)
        if isinstance(reg, MatrixReg):
            data.append(_bytes(state.read_mrf_u8(int(reg))))
        elif isinstance(reg, Accumulator):
            data.append(_bytes(state.acc[unit][int(reg)]))
        elif isinstance(reg, WeightBuffer):
            data.append(_bytes(state.read_wb_u8(unit, int(reg))))
    return data


@cache
def run_window(case: tuple[str, str]) -> Window:
    """Run the program for ``case`` (set, mnemonic) once, recording its window as it issues."""
    set_name, target = case
    cls = program_set(set_name)[target]
    window = measure_window(cls(), config_of(set_name), cls.kernel_max_cycles,
                            inspect=lambda uop, state: _sources(uop.insn, state)
                            if uop.insn.mnemonic == target else None)
    return Window(target, cls.reps, window.uops, window.inspected, window.idle, window.halt_reason)


def test_every_instruction_has_an_energy_program() -> None:
    assert set(generator.EXCLUDED) <= set(IsaSpec.operations)
    assert set(PROGRAMS) == set(IsaSpec.operations) - set(generator.EXCLUDED)


def test_checked_in_programs_match_the_generator() -> None:
    expected = generator.generate()
    on_disk = {path.name: path.read_text() for path in ASM.glob("energy_*.S")}
    stale = sorted(name for name in set(expected) | set(on_disk) if expected.get(name) != on_disk.get(name))
    assert not stale, f"rerun scripts/gen_energy_programs.py: {stale}"


def test_program_sets_declare_their_geometry_and_spacing() -> None:
    for set_name, (hardware, spacing) in SETS.items():
        arch = config_of(set_name).arch_state_config
        programs = program_set(set_name)
        assert set(programs) == set(TARGETS)
        assert {(cls.hardware, cls.tile, cls.spacing) for cls in programs.values()} == {
            (hardware, (arch.mrf_depth, arch.mrf_width), spacing)}


def test_window_holds_exactly_the_repetitions_between_idle_markers(case: tuple[str, str]) -> None:
    target = case[1]
    window = run_window(case)
    assert window.halt_reason == "ecall"
    assert window.idle == [True, True], "a unit was busy when a window marker issued"
    assert tuple(uop.insn.mnemonic for uop in window.uops[-len(CLOSING):]) == CLOSING
    assert len(window.repetitions) == window.reps
    for uop in window.body:
        insn = uop.insn
        if insn.mnemonic != target:
            nop = insn.mnemonic == "addi" and int(insn.rd) == 0
            assert insn.mnemonic in FILLER or nop, f"unexpected {insn} in the window"


def test_operands_are_random_and_vary_between_repetitions(case: tuple[str, str]) -> None:
    target = case[1]
    window = run_window(case)
    if not any(window.sources[0]):
        # Immediate-only or operand-free: the encodings (immediates, destinations, channels) vary.
        encodings = {uop.insn.to_bytecode() for uop in window.repetitions}
        assert target == "fence" or len(encodings) > 1
        return
    words = set()
    for data in (data for rep in window.sources for data in rep):
        if len(data) == 4:
            words.add(data)
        else:
            assert len(set(data)) >= 16, f"{target} reads a degenerate {len(data)}-byte operand"
    assert not words or (len(words) > 1 and any(int.from_bytes(w, "little") for w in words))
    assert len({tuple(rep) for rep in window.sources}) > 1, "every repetition reads the same data"


def test_every_repetition_records_the_same_energy_ops(case: tuple[str, str]) -> None:
    target = case[1]
    window = run_window(case)
    signatures = {frozenset(Counter(op.kind for op in uop.energy_ops).items())
                  for uop in window.repetitions}
    assert len(signatures) == 1, f"{target} repetitions record different EnergyOps: {signatures}"
    kinds = {kind for kind, _ in next(iter(signatures))} - {"instruction_fetch"}
    if target == "delay":
        assert not kinds  # the idle reference: fetched, never executed
    else:
        assert kinds, f"{target} records no EnergyOp besides instruction_fetch"
