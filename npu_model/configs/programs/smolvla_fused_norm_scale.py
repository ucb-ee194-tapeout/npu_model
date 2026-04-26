"""SmolVLA fused_norm_scale kernel: rsqrt(variance) then matrix * rsqrt.

MLIR sources (benchmarks/SaturnNPU/kernels/fused_norm_scale):
    op_a.mlir — rsqrt on a bf16 1-D vector           tensor<241xbf16>
    op_b.mlir — elementwise_mul matrix × rsqrt (broadcast along col axis)
                tensor<241x960xbf16>
Pattern appears 64× in the model. Fusing eliminates the intermediate
rsqrt vector materialisation.

Precision: both ops are bf16 end-to-end. npu_model has no VRSQRT, so
rsqrt is done as ``vrecip.bf16(vsqrt.bf16(x))``.

MLIR → ISA mapping (pair-op BF16):
    op_a: math.rsqrt → vsqrt.bf16 + vrecip.bf16
    op_b: arith.mulf → vmul.bf16

Demo tile: (32, 32) bf16 for both variance and matrix. Full-model
shape is variance [241], matrix [241, 960]; this Program exercises
the op graph on one tile where the variance has been pre-broadcast
to the matrix shape on the host side.
"""

import os

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs


# ═══════════════════════════════════════════════════════════════════════════
# 1. MLIR definition (f32 so IREE llvm-cpu can lower it).
# ═══════════════════════════════════════════════════════════════════════════

FUSED_NORM_SCALE_MLIR = """\
func.func @fused_norm_scale(
    %var: tensor<32x32xf32>, %mat: tensor<32x32xf32>
) -> tensor<32x32xf32> {
  %empty0 = tensor.empty() : tensor<32x32xf32>
  %rsqrt_v = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%var : tensor<32x32xf32>) outs(%empty0 : tensor<32x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %res = math.rsqrt %in : f32
    linalg.yield %res : f32
  } -> tensor<32x32xf32>
  %empty1 = tensor.empty() : tensor<32x32xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%mat, %rsqrt_v : tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%empty1 : tensor<32x32xf32>) {
  ^bb0(%in_mat: f32, %in_rsqrt: f32, %out: f32):
    %res = arith.mulf %in_mat, %in_rsqrt : f32
    linalg.yield %res : f32
  } -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}
"""


# ═══════════════════════════════════════════════════════════════════════════
# 2. PyTorch reference.
# ═══════════════════════════════════════════════════════════════════════════


def fused_norm_scale_reference(
    variance: torch.Tensor, matrix: torch.Tensor
) -> torch.Tensor:
    """output[i, j] = matrix[i, j] * rsqrt(variance[i, j]).

    Matches the two-step kernel: bf16 sqrt, bf16 recip, bf16 mul.
    """
    sqrt_v = torch.sqrt(variance.float()).to(torch.bfloat16)
    rsqrt_v = (1.0 / sqrt_v.float()).to(torch.bfloat16)
    return (matrix.float() * rsqrt_v.float()).to(matrix.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Golden data.
# ═══════════════════════════════════════════════════════════════════════════

torch.manual_seed(42)
# Variance must be positive; abs + small eps avoids div-by-zero.
VARIANCE = (torch.randn(32, 32).abs() + 0.1).to(torch.bfloat16)
MATRIX = torch.randn(32, 32, dtype=torch.bfloat16)
EXPECTED = fused_norm_scale_reference(VARIANCE, MATRIX)

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
            FUSED_NORM_SCALE_MLIR,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
        _cfg = runtime.Config("local-task")
        _ctx = runtime.SystemContext(config=_cfg)
        _ctx.add_vm_module(runtime.VmModule.copy_buffer(_ctx.instance, _vmfb))
        _iree_out = _ctx.modules.module["fused_norm_scale"](
            VARIANCE.float().numpy(), MATRIX.float().numpy()
        )
        _iree_bf16 = torch.from_numpy(np.array(_iree_out)).to(torch.bfloat16)
        _diff = (EXPECTED.float() - _iree_bf16.float()).abs().max().item()
        # f32 rsqrt vs bf16 stepwise → loose tolerance.
        assert _diff < 0.1, f"MLIR vs PyTorch mismatch: {_diff}"
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 4. Memory layout.
#
# Variance and matrix are each stored as (32, 32) bf16 = 2048 B in DRAM,
# split into two 32x16 halves back-to-back. Output mirrors that layout.
# ═══════════════════════════════════════════════════════════════════════════

DRAM_VAR_BASE = 0x0000
DRAM_MAT_BASE = 0x0800
DRAM_OUT_BASE = 0x1000
TILE_BYTES = 2048


# ═══════════════════════════════════════════════════════════════════════════
# 5. NPU ISA program (pair-op BF16).
#
# MRF map:
#   (m0, m1) = variance        (m2, m3) = matrix
#   (m4, m5) = sqrt(variance)  (m6, m7) = rsqrt(variance) = 1 / sqrt
#   (m8, m9) = matrix * rsqrt  (= output)
#
# Scalar reg map:
#   x1 = VMEM var base   x2 = VMEM matrix base   x3 = VMEM out base
#   x4 = DRAM var base   x5 = DRAM matrix base   x6 = DRAM out base
#   x7 = transfer size 2048
# ═══════════════════════════════════════════════════════════════════════════


class SmolVLAFusedNormScaleProgram(Program):
    """fused_norm_scale: out[i,j] = matrix[i,j] * rsqrt(variance[i,j]).

    Optimization: after var DMA completes (ch0, ~1028cy), load and process variance
    (vsqrt + vrecip, ~200cy) while the matrix DMA (ch1, ~1028cy) is still in flight.
    This hides ~200cy of VPU compute behind the sequential DMA engine.

    cycles: ~3312
    """

    instructions: List[Instruction[Any]] = [
        # Scalar setup
        Instruction(mnemonic="lui", args=ScalarArgs(rd=1, imm=0x2)),  # VMEM_VAR = 0x2000
        Instruction(mnemonic="lui", args=ScalarArgs(rd=2, imm=0x3)),  # VMEM_MAT = 0x3000
        Instruction(mnemonic="lui", args=ScalarArgs(rd=3, imm=0x4)),  # VMEM_OUT = 0x4000
        Instruction(mnemonic="addi", args=ScalarArgs(rd=4, rs1=0, imm=DRAM_VAR_BASE)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=5, imm=0x1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=5, rs1=5, imm=-2048)),  # DRAM_MAT = 0x0800
        Instruction(mnemonic="lui", args=ScalarArgs(rd=6, imm=0x1)),            # DRAM_OUT = 0x1000
        Instruction(mnemonic="lui", args=ScalarArgs(rd=7, imm=0x1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=7, rs1=7, imm=-2048)),  # transfer size = 2048
        # Issue var (ch0) and mat (ch1) DMA loads. With a sequential DMA engine,
        # ch1 starts after ch0 finishes (~1028cy), completing at ~2056cy total.
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=1)),
        Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=1, rs1=4, rs2=7, channel=0)),
        Instruction(mnemonic="dma.load.ch<N>", args=DmaArgs(rd=2, rs1=5, rs2=7, channel=1)),
        # Wait for var, then compute rsqrt(var) while matrix DMA is still in flight.
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),   # m0 = var low
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=1, rs1=1, imm12=32)),  # m1 = var high
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(mnemonic="vsqrt.bf16", args=VectorArgs(vd=4, vs1=0)),       # m4/m5 = sqrt(var)
        Instruction(mnemonic="delay", args=ScalarArgs(imm=66)),
        Instruction(mnemonic="vrecip.bf16", args=VectorArgs(vd=6, vs1=4)),      # m6/m7 = rsqrt(var)
        Instruction(mnemonic="delay", args=ScalarArgs(imm=66)),
        # Matrix DMA finishes ~2056cy from start; by now ~200cy of that window is used.
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=2, rs1=2, imm12=0)),   # m2 = mat low
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=3, rs1=2, imm12=32)),  # m3 = mat high
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=8, vs1=2, vs2=6)), # m8/m9 = mat * rsqrt
        Instruction(mnemonic="delay", args=ScalarArgs(imm=66)),
        # Store output
        Instruction(mnemonic="vstore", args=VectorArgs(vd=8, rs1=3, imm12=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=9, rs1=3, imm12=32)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(mnemonic="dma.store.ch<N>", args=DmaArgs(rd=6, rs1=3, rs2=7, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_VAR_BASE, VARIANCE),
        (DRAM_MAT_BASE, MATRIX),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUT_BASE,
        EXPECTED,
    )
