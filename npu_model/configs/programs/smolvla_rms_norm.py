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
    uv run python scripts/test_programs.py --verbose
    # Expect: OK   SmolVLARmsNormProgram
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs


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
    """y = x * rsqrt(sum(x^2) * inv_dim + eps). No learnable scale."""
    sum_sq = (x.float() * x.float()).sum(dim=-1, keepdim=True)  # (32, 1)
    var = (sum_sq * inv_dim).to(torch.bfloat16).float()
    inv_rms = torch.rsqrt(var + eps).to(torch.bfloat16)
    return (x * inv_rms.to(torch.bfloat16)).to(x.dtype)


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
try:
    import numpy as np
    import iree.compiler as compiler
    import iree.runtime as runtime

    _vmfb = compiler.compile_str(RMS_NORM_MLIR, target_backends=["llvm-cpu"])
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

DRAM_X_H0 = 0x0000         # x's first half (cols 0-15), 1024 B
DRAM_X_H1 = 0x0400         # x's second half (cols 16-31), 1024 B
DRAM_INV_DIM = 0x0800      # broadcast 1/dim, 1024 B
DRAM_EPS = 0x0C00          # broadcast eps, 1024 B
DRAM_OUT_H0 = 0x1000       # y's first half, 1024 B
DRAM_OUT_H1 = 0x1400       # y's second half, 1024 B

VMEM_X_H0    = 0x2000
VMEM_X_H1    = 0x2400
VMEM_INV_DIM = 0x2800
VMEM_EPS     = 0x2C00
VMEM_OUT_H0  = 0x3000
VMEM_OUT_H1  = 0x3400
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
    """y = x * rsqrt(mean(x^2) + eps). 32x32 bf16 tile, no learnable scale."""

    instructions: List[Instruction[Any]] = [
        # VMEM addresses  (each + 1024 bytes = +0x400 from the previous)
        Instruction(mnemonic="lui",  args=ScalarArgs(rd=1, imm=0x2)),              # 0x2000 VMEM_X_H0
        Instruction(mnemonic="addi", args=ScalarArgs(rd=2, rs1=1, imm=1024)),      # 0x2400 VMEM_X_H1
        Instruction(mnemonic="addi", args=ScalarArgs(rd=3, rs1=2, imm=1024)),      # 0x2800 VMEM_INV_DIM
        Instruction(mnemonic="addi", args=ScalarArgs(rd=4, rs1=3, imm=1024)),      # 0x2C00 VMEM_EPS
        Instruction(mnemonic="addi", args=ScalarArgs(rd=5, rs1=4, imm=1024)),      # 0x3000 VMEM_OUT_H0
        Instruction(mnemonic="addi", args=ScalarArgs(rd=6, rs1=5, imm=1024)),      # 0x3400 VMEM_OUT_H1
        # DRAM addresses  (x_h0 at 0, each next + 0x400)
        Instruction(mnemonic="addi", args=ScalarArgs(rd=7,  rs1=0, imm=0)),        # 0x0000 DRAM_X_H0
        Instruction(mnemonic="addi", args=ScalarArgs(rd=8,  rs1=7, imm=1024)),     # 0x0400 DRAM_X_H1
        Instruction(mnemonic="addi", args=ScalarArgs(rd=9,  rs1=8, imm=1024)),     # 0x0800 DRAM_INV_DIM
        Instruction(mnemonic="addi", args=ScalarArgs(rd=10, rs1=9, imm=1024)),     # 0x0C00 DRAM_EPS
        Instruction(mnemonic="addi", args=ScalarArgs(rd=11, rs1=10, imm=1024)),    # 0x1000 DRAM_OUT_H0
        Instruction(mnemonic="addi", args=ScalarArgs(rd=12, rs1=11, imm=1024)),    # 0x1400 DRAM_OUT_H1
        Instruction(mnemonic="addi", args=ScalarArgs(rd=13, rs1=0, imm=1024)),     # transfer size = 1024
        # DMA loads
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=1)),
        Instruction(mnemonic="dma.load.ch<N>",   args=DmaArgs(rd=1, rs1=7,  rs2=13, channel=0)),
        Instruction(mnemonic="dma.load.ch<N>",   args=DmaArgs(rd=2, rs1=8,  rs2=13, channel=1)),
        Instruction(mnemonic="dma.wait.ch<N>",   args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>",   args=DmaArgs(channel=1)),
        Instruction(mnemonic="dma.load.ch<N>",   args=DmaArgs(rd=3, rs1=9,  rs2=13, channel=0)),
        Instruction(mnemonic="dma.load.ch<N>",   args=DmaArgs(rd=4, rs1=10, rs2=13, channel=1)),
        Instruction(mnemonic="dma.wait.ch<N>",   args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>",   args=DmaArgs(channel=1)),
        # Load MRF
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=1, rs1=2, imm12=0)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=2, rs1=3, imm12=0)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=3, rs1=4, imm12=0)),
        # Compute
        Instruction(mnemonic="vmul.bf16",        args=VectorArgs(vd=4, vs1=0, vs2=0)),  # v4 = x_h0^2
        Instruction(mnemonic="vmul.bf16",        args=VectorArgs(vd=5, vs1=1, vs2=1)),  # v5 = x_h1^2
        Instruction(mnemonic="delay",            args=ScalarArgs(imm=2)),
        Instruction(mnemonic="vredsum.row.bf16", args=VectorArgs(vd=6, vs1=4)),
        Instruction(mnemonic="vredsum.row.bf16", args=VectorArgs(vd=7, vs1=5)),
        Instruction(mnemonic="delay",            args=ScalarArgs(imm=8)),
        Instruction(mnemonic="vadd.bf16",        args=VectorArgs(vd=8, vs1=6, vs2=7)),  # sum(x^2) broadcast per row
        Instruction(mnemonic="delay",            args=ScalarArgs(imm=2)),
        Instruction(mnemonic="vmul.bf16",        args=VectorArgs(vd=9, vs1=8, vs2=2)),  # * inv_dim
        Instruction(mnemonic="delay",            args=ScalarArgs(imm=2)),
        Instruction(mnemonic="vadd.bf16",        args=VectorArgs(vd=10, vs1=9, vs2=3)), # + eps
        Instruction(mnemonic="delay",            args=ScalarArgs(imm=2)),
        Instruction(mnemonic="vsqrt.bf16",       args=VectorArgs(vd=11, vs1=10)),
        Instruction(mnemonic="delay",            args=ScalarArgs(imm=8)),
        Instruction(mnemonic="vrecip.bf16",      args=VectorArgs(vd=12, vs1=11)),
        Instruction(mnemonic="delay",            args=ScalarArgs(imm=8)),
        Instruction(mnemonic="vmul.bf16",        args=VectorArgs(vd=13, vs1=0, vs2=12)),  # y_h0 = x_h0 * inv_rms
        Instruction(mnemonic="vmul.bf16",        args=VectorArgs(vd=14, vs1=1, vs2=12)),  # y_h1
        Instruction(mnemonic="delay",            args=ScalarArgs(imm=2)),
        # Store
        Instruction(mnemonic="vstore", args=VectorArgs(vd=13, rs1=5, imm12=0)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=14, rs1=6, imm12=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=20)),
        Instruction(mnemonic="dma.store.ch<N>", args=DmaArgs(rd=11, rs1=5, rs2=13, channel=0)),
        Instruction(mnemonic="dma.store.ch<N>", args=DmaArgs(rd=12, rs1=6, rs2=13, channel=1)),
        Instruction(mnemonic="dma.wait.ch<N>",  args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>",  args=DmaArgs(channel=1)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_X_H0,    _x_h0),
        (DRAM_X_H1,    _x_h1),
        (DRAM_INV_DIM, _inv_dim),
        (DRAM_EPS,     _eps),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUT_H0,
        # The kernel writes two halves at DRAM_OUT_H0 and DRAM_OUT_H1.
        # Read the first half and compare against cols 0-15 of EXPECTED.
        EXPECTED[:, :16],
    )
