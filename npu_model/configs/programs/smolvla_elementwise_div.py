"""SmolVLA elementwise_div kernel.

Elementwise divide. 217 instances in SmolVLA; 6 shape variants.
This Program implements the 32x32 canonical form. The ISA uses
``vrecip`` + ``vmul`` (no dedicated divide op).
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs


# ═══════════════════════════════════════════════════════════════════════════
# 1. MLIR definition.
# ═══════════════════════════════════════════════════════════════════════════

ELEMENTWISE_DIV_MLIR = """\
func.func @elementwise_div(
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
  ^bb0(%n: f32, %d: f32, %_o: f32):
    %q = arith.divf %n, %d : f32
    linalg.yield %q : f32
  } -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}
"""

# ═══════════════════════════════════════════════════════════════════════════
# 2. PyTorch reference.
# ═══════════════════════════════════════════════════════════════════════════


def elementwise_div_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Kernel computes via vrecip + vmul in bf16; mirror the rounding.
    inv_b = (1.0 / b.float()).to(torch.bfloat16)
    return (a * inv_b).to(a.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Golden data.
# ═══════════════════════════════════════════════════════════════════════════

torch.manual_seed(45)
INPUT_A = torch.randn(32, 32, dtype=torch.bfloat16)
# Avoid divisor magnitudes close to zero so vrecip doesn't blow up.
INPUT_B = torch.randn(32, 32, dtype=torch.bfloat16)
INPUT_B = torch.where(INPUT_B.abs() < 0.25, torch.full_like(INPUT_B, 0.5), INPUT_B)
EXPECTED = elementwise_div_reference(INPUT_A, INPUT_B)

# Cross-check: compile the MLIR via IREE on CPU and compare to PyTorch.
# MLIR uses f32 divf; PyTorch mirrors the kernel's bf16 vrecip+vmul.
# Tolerance ~5e-2 = a couple bf16 ULPs, which is the expected gap
# between f32-exact and bf16-reciprocal.
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
            ELEMENTWISE_DIV_MLIR,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
        _cfg = runtime.Config("local-task")
        _ctx = runtime.SystemContext(config=_cfg)
        _ctx.add_vm_module(runtime.VmModule.copy_buffer(_ctx.instance, _vmfb))
        _iree_out = _ctx.modules.module["elementwise_div"](
            INPUT_A.float().numpy(), INPUT_B.float().numpy()
        )
        _iree_bf16 = torch.from_numpy(np.array(_iree_out)).to(torch.bfloat16)
        _diff = (EXPECTED.float() - _iree_bf16.float()).abs().max().item()
        assert _diff < 5e-2, f"MLIR vs PyTorch mismatch: {_diff}"
    except ImportError:
        pass

DRAM_A_H0 = 0x0000
DRAM_A_H1 = 0x0400
DRAM_B_H0 = 0x0800
# dram_in_3 (B_h1) at 0xC00; the kernel writes its first output half back
# at 0xC00 too (see manifest patch_points for elementwise_div). So
# DRAM_B_H1 and DRAM_OUT_H0 share the same address — the DMA store
# overwrites the B_h1 buffer in place once it's no longer needed.
DRAM_B_H1 = 0x0C00
DRAM_OUT_H0 = 0x0C00  # written after B_h1 is read into VMEM
DRAM_OUT_H1 = 0x1000
EXPECTED_STACKED = torch.cat((EXPECTED[:, :16], EXPECTED[:, 16:]), dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# 4. NPU ISA program.
# ═══════════════════════════════════════════════════════════════════════════


class SmolVLAElementwiseDivProgram(Program):
    """Auto-generated single-file Program for the ``elementwise_div`` kernel.

    ISA is lifted from the merlin kernel manifest (see
    ``benchmarks/SaturnNPU/kernel_library/manifest.json``). This Program
    mirrors the ``smolvla_silu.py`` template: self-contained, no cross-
    file helpers, torch-allclose golden check via ``pytest tests/test_programs.py``.
    """

    instructions: List[Instruction[Any]] = [
        Instruction("lui", ScalarArgs(rd=1, imm=2)),
        Instruction("addi", ScalarArgs(rd=2, rs1=1, imm=1024)),
        Instruction("addi", ScalarArgs(rd=3, rs1=2, imm=1024)),
        Instruction("addi", ScalarArgs(rd=4, rs1=3, imm=1024)),
        Instruction("lui", ScalarArgs(rd=5, imm=3)),
        Instruction("addi", ScalarArgs(rd=6, rs1=5, imm=1024)),
        Instruction("addi", ScalarArgs(rd=7)),
        Instruction("addi", ScalarArgs(rd=8, imm=1024)),
        Instruction("addi", ScalarArgs(rd=9, rs1=8, imm=1024)),
        Instruction("addi", ScalarArgs(rd=10, rs1=9, imm=1024)),
        Instruction("lui", ScalarArgs(rd=11, imm=1)),
        Instruction("addi", ScalarArgs(rd=11, rs1=11, imm=-1024)),
        Instruction("addi", ScalarArgs(rd=12, rs1=11, imm=1024)),
        Instruction("addi", ScalarArgs(rd=13, imm=1024)),
        Instruction("dma.config.ch<N>", DmaArgs(channel=1)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=7, rs2=13, channel=0)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=8, rs2=13, channel=1)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=3, rs1=9, rs2=13, channel=0)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=4, rs1=10, rs2=13, channel=1)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        Instruction("vload", VectorArgs(vd=0, rs1=1)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vload", VectorArgs(vd=1, rs1=2)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vload", VectorArgs(vd=2, rs1=3)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vload", VectorArgs(vd=3, rs1=4)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vrecip.bf16", VectorArgs(vd=4, vs1=2)),  # (v4, v5) = 1 / (v2, v3)
        Instruction("delay", ScalarArgs(imm=66)),
        Instruction(
            "vmul.bf16", VectorArgs(vd=6, vs2=4)
        ),  # (v6, v7) = (v0, v1) * (v4, v5)
        Instruction("delay", ScalarArgs(imm=66)),
        Instruction("vstore", VectorArgs(vd=6, rs1=5)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vstore", VectorArgs(vd=7, rs1=6)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("dma.store.ch<N>", DmaArgs(rd=11, rs1=5, rs2=13, channel=0)),
        Instruction("dma.store.ch<N>", DmaArgs(rd=12, rs1=6, rs2=13, channel=1)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_A_H0, INPUT_A[:, :16].contiguous()),
        (DRAM_A_H1, INPUT_A[:, 16:].contiguous()),
        (DRAM_B_H0, INPUT_B[:, :16].contiguous()),
        (DRAM_B_H1, INPUT_B[:, 16:].contiguous()),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUT_H0,
        EXPECTED_STACKED,
    )
