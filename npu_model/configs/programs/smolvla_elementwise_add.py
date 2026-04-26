"""SmolVLA elementwise addition kernel.

Adds two 32x32 bf16 tiles elementwise. Appears 539 times across SmolVLA
(residual adds, bias adds, etc.). 16 shape variants in the full model;
this Program implements the 32x32 canonical form. Other shapes use the
same pattern with different transfer sizes.

Everything lives in this one file:
    - The MLIR op definition (as a string, compilable with iree.compiler)
    - The PyTorch reference implementation
    - The NPU ISA program
    - The golden result (computed from PyTorch, cross-checked with MLIR if IREE available)

Model context:
    Elementwise add is the most common op in SmolVLA (residuals in every
    transformer block + bias adds + positional adds).

MLIR → ISA mapping:
    arith.addf %a %b → vadd.bf16(a, b)   pair-op: (vd, vd+1) = (vs1, vs1+1) + (vs2, vs2+1)
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs


# ═══════════════════════════════════════════════════════════════════════════
# 1. MLIR definition — the exact op from SmolVLA's global-optimization IR.
# ═══════════════════════════════════════════════════════════════════════════

ELEMENTWISE_ADD_MLIR = """\
func.func @elementwise_add(
    %a: tensor<32x32xf32>, %b: tensor<32x32xf32>
) -> tensor<32x32xf32> {
  %empty = tensor.empty() : tensor<32x32xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%empty : tensor<32x32xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %sum = arith.addf %lhs, %rhs : f32
    linalg.yield %sum : f32
  } -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}
"""


# ═══════════════════════════════════════════════════════════════════════════
# 2. PyTorch reference — computes the golden output.
# ═══════════════════════════════════════════════════════════════════════════


def elementwise_add_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise add: y[i, j] = a[i, j] + b[i, j]. Cast back to input dtype."""
    return (a.float() + b.float()).to(a.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Golden data — deterministic inputs + expected output.
# ═══════════════════════════════════════════════════════════════════════════

torch.manual_seed(42)
INPUT_A = torch.randn(32, 32, dtype=torch.bfloat16)
INPUT_B = torch.randn(32, 32, dtype=torch.bfloat16)

EXPECTED = elementwise_add_reference(INPUT_A, INPUT_B)

# Optional cross-check via IREE (mirrors smolvla_silu.py).
#
# Gated behind NPU_MODEL_ENABLE_IREE_CROSSCHECK because iree-runtime's
# nanobind bindings leak MappedMemory / HalBufferView instances at
# interpreter shutdown (the C extension's type objects finalize before
# our module globals release their refs). CI leaves the env var unset
# so the test process exits without triggering nanobind's leak report.
import os

if os.environ.get("NPU_MODEL_ENABLE_IREE_CROSSCHECK", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}:
    try:
        import numpy as np
        import iree.compiler as compiler
        import iree.runtime as runtime

        _vmfb = compiler.compile_str(
            ELEMENTWISE_ADD_MLIR,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
        _config = runtime.Config("local-task")
        _ctx = runtime.SystemContext(config=_config)
        _ctx.add_vm_module(runtime.VmModule.copy_buffer(_ctx.instance, _vmfb))
        _iree_out = _ctx.modules.module["elementwise_add"](
            INPUT_A.float().numpy(), INPUT_B.float().numpy()
        )
        _iree_expected = torch.from_numpy(np.array(_iree_out)).to(torch.bfloat16)
        _diff = (EXPECTED.float() - _iree_expected.float()).abs().max().item()
        assert _diff < 1e-3, f"MLIR vs PyTorch mismatch: {_diff}"
        EXPECTED = _iree_expected
    except ImportError:
        pass  # IREE not available — use PyTorch reference (fine for CI)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Memory layout.
#
# The kernel treats a 32x32 bf16 tile as two [32, 16] halves packed
# consecutively in memory (bytes 0..1023 = cols 0-15, bytes 1024..2047 =
# cols 16-31). For a plain torch.randn(32, 32).contiguous(), this
# interpretation maps row-major bytes to the two halves as "first 16
# rows then next 16 rows" — the elementwise add is layout-invariant, so
# reading the output as [32, 32] row-major recovers the arithmetic sum.
# ═══════════════════════════════════════════════════════════════════════════

DRAM_A_BASE = 0x0000
DRAM_B_BASE = 0x1000
DRAM_OUTPUT_BASE = 0x2000
VMEM_A_BASE = 0x4000
VMEM_B_BASE = 0x5000
VMEM_OUTPUT_BASE = 0x6000
TILE_BYTES = 2048  # 32 * 32 * 2 (bf16)


# ═══════════════════════════════════════════════════════════════════════════
# 5. NPU ISA program.
#
# Two vloads per operand (imm12=0 / imm12=32 picks the first / second
# 32x16 half), two vadds, two vstores, one DMA store.
# ═══════════════════════════════════════════════════════════════════════════


class SmolVLAElementwiseAddProgram(Program):
    """y = a + b on two 32x32 bf16 tiles. cycles: ~270"""

    instructions: List[Instruction[Any]] = [
        # ── Scalar setup: VMEM / DRAM addresses + transfer size ──
        Instruction(
            mnemonic="lui", args=ScalarArgs(rd=1, imm=0x4)
        ),  # x1 = VMEM_A = 0x4000
        Instruction(
            mnemonic="lui", args=ScalarArgs(rd=2, imm=0x5)
        ),  # x2 = VMEM_B = 0x5000
        Instruction(
            mnemonic="lui", args=ScalarArgs(rd=3, imm=0x6)
        ),  # x3 = VMEM_OUT = 0x6000
        Instruction(
            mnemonic="addi", args=ScalarArgs(rd=4, rs1=0, imm=DRAM_A_BASE)
        ),  # x4 = 0
        Instruction(
            mnemonic="lui", args=ScalarArgs(rd=5, imm=0x1)
        ),  # x5 = DRAM_B = 0x1000
        Instruction(
            mnemonic="lui", args=ScalarArgs(rd=6, imm=0x2)
        ),  # x6 = DRAM_OUT = 0x2000
        # Transfer size = 2048 (= 0x800). Split via lui + addi because
        # 2048 exceeds the signed 12-bit addi immediate range.
        Instruction(
            mnemonic="lui", args=ScalarArgs(rd=7, imm=0x1)
        ),  # x7 = 0x1000 = 4096
        Instruction(
            mnemonic="addi", args=ScalarArgs(rd=7, rs1=7, imm=-2048)
        ),  # x7 = 2048
        # ── DMA: DRAM → VMEM (channels 0 and 1 in parallel) ──
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=1)),
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=1, rs1=4, rs2=7, channel=0)
        ),  # VMEM[x1] ← DRAM[x4]
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=2, rs1=5, rs2=7, channel=1)
        ),  # VMEM[x2] ← DRAM[x5]
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
        # ── Load 2 halves per operand (imm12 in units of 32 bytes) ──
        Instruction(
            mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)
        ),  # v0 = A[:32, :16]
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(
            mnemonic="vload", args=VectorArgs(vd=1, rs1=1, imm12=32)
        ),  # v1 = A[:32, 16:32] (= next 1024 B)
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(
            mnemonic="vload", args=VectorArgs(vd=2, rs1=2, imm12=0)
        ),  # v2 = B half 0
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(
            mnemonic="vload", args=VectorArgs(vd=3, rs1=2, imm12=32)
        ),  # v3 = B half 1
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        # ── Pair-op add: (v4, v5) = (v0, v1) + (v2, v3) ──
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=4, vs1=0, vs2=2)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=66)),
        # ── Store: MRF → VMEM → DRAM ──
        Instruction(mnemonic="vstore", args=VectorArgs(vd=4, rs1=3, imm12=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=5, rs1=3, imm12=32)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(
            mnemonic="dma.store.ch<N>", args=DmaArgs(rd=6, rs1=3, rs2=7, channel=0)
        ),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_A_BASE, INPUT_A),
        (DRAM_B_BASE, INPUT_B),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUTPUT_BASE,
        EXPECTED,
    )
