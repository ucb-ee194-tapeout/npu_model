"""SmolVLA RMS-norm kernel.

Computes ``y = x * rsqrt(mean(x^2) + eps)`` on a 32x32 bf16 tile.
Appears 139 times across SmolVLA (pre-attention and pre-MLP norms in
every Gemma block; LayerNorm in SigLIP blocks). 3 shape variants total;
this Program implements the 32x32 canonical form.

Model context:
    RMSNorm(x) = x / sqrt(mean(x^2) + eps). The learnable weight scale
    (standard RMS-norm multiplies by a ``weight`` tensor at the end) is
    NOT included in this kernel — compose it with a subsequent
    ``elementwise_mul`` if you need the scaled form.

MLIR → ISA mapping:
    arith.mulf x x  → vmul.bf16(x, x)                  (square)
    linalg.reduce sum → vredsum.row.bf16                (row-sum of squares)
    arith.mulf  sum inv_dim → vmul.bf16                 (mean of squares)
    arith.addf  mean eps    → vadd.bf16                 (+ eps)
    math.rsqrt  denom        → vsqrt.bf16 + vrecip.bf16 (1/sqrt)
    arith.mulf  x inv_rms    → vmul.bf16                (normalize)

Run:
    uv run pytest tests/test_programs.py --sim-verbose -vv
    # Expect: OK   SmolVLARmsNormProgram
"""

import torch
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER

# ═══════════════════════════════════════════════════════════════════════════
# 1. MLIR definition.
# ═══════════════════════════════════════════════════════════════════════════

RMS_NORM_MLIR = """\
func.func @rms_norm(
    %x: tensor<32x32xf32>, %inv_dim: tensor<32xf32>, %eps: tensor<32xf32>
) -> tensor<32x32xf32> {
  %sq_empty = tensor.empty() : tensor<32x32xf32>
  %sq = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%x : tensor<32x32xf32>) outs(%sq_empty : tensor<32x32xf32>) {
  ^bb0(%xe: f32, %_o: f32):
    %p = arith.mulf %xe, %xe : f32
    linalg.yield %p : f32
  } -> tensor<32x32xf32>

  %row_empty = tensor.empty() : tensor<32xf32>
  %zero = arith.constant 0.0 : f32
  %row_init = linalg.fill ins(%zero : f32) outs(%row_empty : tensor<32xf32>) -> tensor<32xf32>
  %row_sum = linalg.reduce
      ins(%sq : tensor<32x32xf32>) outs(%row_init : tensor<32xf32>)
      dimensions = [1]
    (%e: f32, %acc: f32) {
      %s = arith.addf %acc, %e : f32
      linalg.yield %s : f32
    }

  %norm_empty = tensor.empty() : tensor<32x32xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%x, %row_sum, %inv_dim, %eps
        : tensor<32x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>)
    outs(%norm_empty : tensor<32x32xf32>) {
  ^bb0(%xe: f32, %rs: f32, %idm: f32, %ep: f32, %_o: f32):
    %mean = arith.mulf %rs, %idm : f32
    %denom = arith.addf %mean, %ep : f32
    %inv = math.rsqrt %denom : f32
    %y = arith.mulf %xe, %inv : f32
    linalg.yield %y : f32
  } -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}
"""


# ═══════════════════════════════════════════════════════════════════════════
# 2. PyTorch reference — mirrors the exact ISA arithmetic (bf16
#    accumulation, per-row rsqrt).
# ═══════════════════════════════════════════════════════════════════════════


def rms_norm_reference(
    x: torch.Tensor, inv_dim: float = 1.0 / 32.0, eps: float = 1e-6
) -> torch.Tensor:
    """y = x * (1 / sqrt(mean(x^2) + eps)). Matches pair-op ISA sequence.

    Steps (all in bf16 after each stage, mirroring the kernel):
        sq    = x * x                      (vsquare.bf16)
        sum   = sum over 32 cols            (vredsum.row.bf16)
        mean  = sum * inv_dim               (vmul.bf16)
        denom = mean + eps                  (vadd.bf16)
        root  = sqrt(denom)                 (vsqrt.bf16)
        inv   = 1 / root                    (vrecip.bf16)
        y     = x * inv                     (vmul.bf16)
    """
    xb = x.to(torch.bfloat16)
    sq = (xb * xb).to(torch.bfloat16)
    row_sum = sq.sum(dim=-1, keepdim=True).to(torch.bfloat16)
    inv_dim_t = torch.full_like(row_sum, inv_dim, dtype=torch.bfloat16)
    eps_t = torch.full_like(row_sum, eps, dtype=torch.bfloat16)
    mean = (row_sum * inv_dim_t).to(torch.bfloat16)
    denom = (mean + eps_t).to(torch.bfloat16)
    root = torch.sqrt(denom.float()).to(torch.bfloat16)
    inv = (1.0 / root.float()).to(torch.bfloat16)
    return (xb * inv).to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Golden data.
# ═══════════════════════════════════════════════════════════════════════════

torch.manual_seed(42)
INPUT = torch.randn(32, 32, dtype=torch.bfloat16)
EXPECTED = rms_norm_reference(INPUT)

# Cross-check: compile MLIR via IREE, feed f32 inputs (MLIR signature
# declares tensor<32x32xf32> + two tensor<32xf32> broadcasts for
# inv_dim and eps), compare to PyTorch's bf16-accumulated reference.
# Tolerance ~5e-2 accounts for the bf16-rounding in the PyTorch ref.
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
            RMS_NORM_MLIR,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
        _cfg = runtime.Config("local-task")
        _ctx = runtime.SystemContext(config=_cfg)
        _ctx.add_vm_module(runtime.VmModule.copy_buffer(_ctx.instance, _vmfb))
        _inv_dim = np.full((32,), 1.0 / 32.0, dtype=np.float32)
        _eps = np.full((32,), 1e-6, dtype=np.float32)
        _iree_out = _ctx.modules.module["rms_norm"](
            INPUT.float().numpy(), _inv_dim, _eps
        )
        _iree_bf16 = torch.from_numpy(np.array(_iree_out)).to(torch.bfloat16)
        _diff = (EXPECTED.float() - _iree_bf16.float()).abs().max().item()
        assert _diff < 5e-2, f"MLIR vs PyTorch mismatch: {_diff}"
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 4. Memory layout.
#
# The handwritten ISA treats the 32x32 bf16 input as TWO 32x16 halves
# (1024 B each). dram_in_2 / dram_in_3 carry the broadcast ``inv_dim``
# and ``eps`` constants as 32x16 bf16 tiles (1024 B each).
# ═══════════════════════════════════════════════════════════════════════════

DRAM_X_H0 = 0x80000000  # x's first half (cols 0-15), 1024 B
DRAM_X_H1 = 0x80000400  # x's second half (cols 16-31), 1024 B
DRAM_INV_DIM = 0x80000800  # broadcast 1/dim, 1024 B
DRAM_EPS = 0x80000C00  # broadcast eps, 1024 B
DRAM_OUT_H0 = 0x80001000  # y's first half, 1024 B
DRAM_OUT_H1 = 0x80001400  # y's second half, 1024 B
EXPECTED_STACKED = torch.cat((EXPECTED[:, :16], EXPECTED[:, 16:]), dim=0)

VMEM_X_H0 = 0x20002000
VMEM_X_H1 = 0x20002400
VMEM_INV_DIM = 0x20002800
VMEM_EPS = 0x20002C00
VMEM_OUT_H0 = 0x20003000
VMEM_OUT_H1 = 0x20003400
TILE_BYTES = 1024  # 32 * 16 * 2 (bf16 half)

_x_h0, _x_h1 = INPUT[:, :16].contiguous(), INPUT[:, 16:].contiguous()
_inv_dim = torch.full((32, 16), 1.0 / 32.0, dtype=torch.bfloat16)
_eps = torch.full((32, 16), 1e-6, dtype=torch.bfloat16)


# ═══════════════════════════════════════════════════════════════════════════
# 5. NPU ISA program.
#
# Register map:
#   x1 = VMEM_X_H0    x2 = VMEM_X_H1    x3 = VMEM_INV_DIM  x4 = VMEM_EPS
#   x5 = VMEM_OUT_H0  x6 = VMEM_OUT_H1
#   x7  = DRAM_X_H0   x8 = DRAM_X_H1    x9 = DRAM_INV_DIM  x10 = DRAM_EPS
#   x11 = DRAM_OUT_H0 x12 = DRAM_OUT_H1
#   x13 = 1024 (transfer size per half)
#
#   v0 = x_h0          v1 = x_h1          v2 = inv_dim       v3 = eps
#   v4 = x_h0^2        v5 = x_h1^2
#   v6 = rowsum(v4)    v7 = rowsum(v5)
#   v8 = v6 + v7       (= full row sum of x^2)
#   v9 = v8 * inv_dim  (= mean(x^2))
#   v10 = v9 + eps     (= var + eps, broadcast)
#   v11 = sqrt(v10)    v12 = 1 / v11  (= rsqrt(var + eps))
#   v13 = x_h0 * inv_rms   v14 = x_h1 * inv_rms  — output halves
# ═══════════════════════════════════════════════════════════════════════════


class SmolVLARmsNormProgram(Program):
    """y = x * rsqrt(mean(x^2) + eps). 32x32 bf16 tile, no learnable scale.

    Pair-op rewrite: (m0, m1) is the (32,32) input tile; vredsum.row.bf16
    reduces all 32 columns in one shot. Mreg layout:
      (m0, m1)   = X
      (m2, m3)   = X^2           via vsquare.bf16
      (m4, m5)   = row-sum(X^2)  via vredsum.row
      (m6, m7)   = inv_dim       (DMA-loaded constant tile, then Y)
      (m8, m9)   = eps           (DMA-loaded constant tile)
      (m10, m11) = mean(X^2) = sum * inv_dim
      (m12, m13) = mean + eps
      (m14, m15) = sqrt(mean + eps)
      (m4, m5)   = 1 / sqrt  (inv_rms, reuses pair 4/5)
      (m6, m7)   = X * inv_rms = Y (reuses pair 6/7)
    """

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'smolvla_rms_norm.S')

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_X_H0, _x_h0),
        (DRAM_X_H1, _x_h1),
        (DRAM_INV_DIM, _inv_dim),
        (DRAM_EPS, _eps),
    ]

    golden_result: list[tuple[int, torch.Tensor]] = [(
        DRAM_OUT_H0,
        EXPECTED_STACKED,
    )]
