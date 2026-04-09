#!/usr/bin/env python3
"""
Tests for bank conflict detection in MRF (tensor register file) and VMEM.

Verifies that BankConflictError is raised when concurrent in-flight
instructions access the same SRAM bank, and that valid programs with
non-overlapping resource accesses run without errors.

Usage:
    uv run python scripts/test_bank_conflict.py
"""

import sys
import tempfile
from pathlib import Path
from typing import Any, List, Tuple

# Add project root for imports when run as script
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import torch
from npu_model.hardware.bank_conflict import BankConflictError
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation
from npu_model.configs.isa_definition import *  # noqa: F401, F403
from npu_model.configs.hardware import DefaultHardwareConfig
from npu_model.software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(program: Program, max_cycles: int = 200) -> None:
    """Run a program; re-raise any exception from the simulation."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        trace_path = f.name
    try:
        sim = Simulation(
            hardware_config=DefaultHardwareConfig(),
            logger_config=LoggerConfig(filename=trace_path),
            program=program,
        )
        # Suppress per-cycle DMA prints during tests
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            sim.run(max_cycles=max_cycles)
        finally:
            sys.stdout = old_stdout
    finally:
        try:
            Path(trace_path).unlink(missing_ok=True)
        except (PermissionError, OSError):
            pass


# ---------------------------------------------------------------------------
# Programs that SHOULD trigger BankConflictError
# ---------------------------------------------------------------------------

class _MrfConflictProgram(Program):
    """
    VPU executes ``vadd.bf16 m5, m0, m0`` (reads m0; 2-cycle latency).
    While VPU is still in-flight, MXU0 is dispatched ``vmatmul.mxu0``
    which also reads m0.  Both units hold m0 concurrently → MRF bank conflict.
    """

    instructions: List[Instruction[Any]] = [
        # VPU: reads m0 (execute_delay = 2 cycles)
        Instruction("vadd.bf16", VectorArgs(vd=5, vs1=0, vs2=0)),
        # MXU: dispatched 1 cycle after VPU accepts vadd → conflict on m0
        Instruction("vmatmul.mxu0", MatrixArgs(vd=0, vs1=0, vs2=0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _VmemConflictProgram(Program):
    """
    DMA channel 0 issues a ``dma.load`` that writes to VMEM[0..1023]
    (16-cycle transfer).  Before it finishes, VPU issues a ``vload`` that
    reads the same VMEM range → VMEM bank conflict.
    """

    instructions: List[Instruction[Any]] = [
        # x2 = 1024  (transfer length)
        Instruction("addi", ScalarArgs(rd=2, rs1=0, imm=1024)),
        # Set up DMA base address (DRAM base = 0)
        Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=0)),
        # Wait for channel 0 to be idle (completes immediately on first run)
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        # DMA load: VMEM[x0=0..1023], takes ceil(1024/64)=16 cycles
        Instruction("dma.load.ch<N>", DmaArgs(rd=0, rs1=0, rs2=2, channel=0)),
        # vload: reads VMEM[x0=0..1023] while DMA is still in-flight → conflict
        Instruction("vload", VectorArgs(vd=0, rs1=0, imm12=0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0, torch.zeros(1024, dtype=torch.uint8)),
    ]


# ---------------------------------------------------------------------------
# Programs that should run WITHOUT any bank conflict
# ---------------------------------------------------------------------------

class _NoMrfConflictProgram(Program):
    """
    VPU reads m0/m1 and writes m5; MXU0 reads m2.  No register overlap.
    A ``delay`` instruction ensures VPU has completed before MXU starts.
    """

    instructions: List[Instruction[Any]] = [
        # VPU: reads m0, m1; writes m5 (2-cycle latency)
        Instruction("vadd.bf16", VectorArgs(vd=5, vs1=0, vs2=1)),
        # Wait 3 cycles so VPU completes before MXU is dispatched
        Instruction("delay", ScalarArgs(imm=3)),
        # MXU: reads m2 – disjoint from {m0, m1, m5}
        Instruction("vmatmul.mxu0", MatrixArgs(vd=0, vs1=2, vs2=0)),
        # Allow MXU to finish
        Instruction("delay", ScalarArgs(imm=32)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = []


class _NoVmemConflictProgram(Program):
    """
    DMA writes to VMEM[0..1023].  VPU loads from VMEM[1024..2047].
    The two ranges are in disjoint banks → no VMEM bank conflict.
    """

    instructions: List[Instruction[Any]] = [
        # x2 = 1024 (transfer length for DMA)
        Instruction("addi", ScalarArgs(rd=2, rs1=0, imm=1024)),
        # x3 = 1024 (VMEM base for vload, = 32 * imm12 with imm12=32)
        Instruction("addi", ScalarArgs(rd=3, rs1=0, imm=1024)),
        # DMA base address = 0
        Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        # DMA load: VMEM[0..1023] (banks 0-31)
        Instruction("dma.load.ch<N>", DmaArgs(rd=0, rs1=0, rs2=2, channel=0)),
        # vload: addr = x3 + imm12*32 = 1024 + 0*32 = 1024 → VMEM[1024..2047]
        # (banks 32-63) – disjoint from DMA banks 0-31
        Instruction("vload", VectorArgs(vd=0, rs1=3, imm12=0)),
        # Wait for DMA before finishing
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
    ]
    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0, torch.zeros(1024, dtype=torch.uint8)),
        (1024, torch.zeros(1024, dtype=torch.uint8)),
    ]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def _assert_raises(exc_type, program: Program, label: str) -> None:
    """Assert that running *program* raises *exc_type*."""
    try:
        _run(program)
    except exc_type as e:
        print(f"OK   {label}: correctly raised {exc_type.__name__}: {e}")
        return
    except Exception as e:
        print(f"FAIL {label}: expected {exc_type.__name__} but got {type(e).__name__}: {e}",
              file=sys.stderr)
        raise
    raise AssertionError(
        f"{label}: expected {exc_type.__name__} but no exception was raised"
    )


def _assert_no_raise(program: Program, label: str) -> None:
    """Assert that running *program* does not raise."""
    try:
        _run(program)
    except Exception as e:
        print(f"FAIL {label}: unexpected exception {type(e).__name__}: {e}",
              file=sys.stderr)
        raise
    print(f"OK   {label}")


def main() -> int:
    failed: list[tuple[str, Exception]] = []

    cases_conflict = [
        (_MrfConflictProgram(), "MrfBankConflict"),
        (_VmemConflictProgram(), "VmemBankConflict"),
    ]

    cases_ok = [
        (_NoMrfConflictProgram(), "NoMrfConflict"),
        (_NoVmemConflictProgram(), "NoVmemConflict"),
    ]

    for program, label in cases_conflict:
        try:
            _assert_raises(BankConflictError, program, label)
        except Exception as e:
            failed.append((label, e))

    for program, label in cases_ok:
        try:
            _assert_no_raise(program, label)
        except Exception as e:
            failed.append((label, e))

    if failed:
        print(f"\n{len(failed)} test(s) failed.", file=sys.stderr)
        for name, err in failed:
            print(f"  {name}: {err}", file=sys.stderr)
        return 1

    print(f"\nAll {len(cases_conflict) + len(cases_ok)} bank-conflict tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
