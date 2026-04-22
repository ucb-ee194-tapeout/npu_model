"""SmolVLA attention kernel.

Single-tile scaled dot-product attention: ``softmax((Q @ K) * scale) @ V``.
Q, K, V are fp8 32x32 tiles; ``scale`` is an elementwise bf16 mask/scale.
Writes the bf16 output as two 32x16 halves.
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs


# ═══════════════════════════════════════════════════════════════════════════
# 2. PyTorch reference.
# ═══════════════════════════════════════════════════════════════════════════


def attention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    qf, kf, vf = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
    sf = scale.to(torch.float32)
    scores = (qf @ kf) * sf
    scores = scores - scores.max(dim=-1, keepdim=True).values
    probs = scores.exp()
    probs = probs / probs.sum(dim=-1, keepdim=True)
    probs_fp8 = probs.to(torch.float8_e4m3fn).to(torch.float32)
    return (probs_fp8 @ vf).to(torch.bfloat16)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Golden data.
# ═══════════════════════════════════════════════════════════════════════════

import math

# ───────────────────────────────────────────────────────────────────────
# MLIR: single-tile SDPA (softmax((Q @ K) * scale) @ V), expressed in
# f32 for stock llvm-cpu compatibility. The hardware does the matmuls
# in fp8 with a bf16 accumulator and quantizes ``probs`` back to fp8
# before the second matmul; this MLIR omits the fp8 round-trip on
# probs, so the cross-check tolerance is loose.
# ───────────────────────────────────────────────────────────────────────

ATTENTION_MLIR = """\
func.func @attention(
    %q: tensor<32x32xf32>, %k: tensor<32x32xf32>,
    %v: tensor<32x32xf32>, %scale: tensor<32x32xf32>
) -> tensor<32x32xf32> {
  %zero = arith.constant 0.0 : f32
  %qk0 = tensor.empty() : tensor<32x32xf32>
  %qk_init = linalg.fill ins(%zero : f32) outs(%qk0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %qk = linalg.matmul ins(%q, %k : tensor<32x32xf32>, tensor<32x32xf32>)
                      outs(%qk_init : tensor<32x32xf32>) -> tensor<32x32xf32>
  %sc0 = tensor.empty() : tensor<32x32xf32>
  %scaled = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%qk, %scale : tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%sc0 : tensor<32x32xf32>) {
  ^bb0(%a: f32, %b: f32, %_: f32):
    %m = arith.mulf %a, %b : f32
    linalg.yield %m : f32
  } -> tensor<32x32xf32>
  %sm0 = tensor.empty() : tensor<32x32xf32>
  %probs = linalg.softmax dimension(1) ins(%scaled : tensor<32x32xf32>)
                                        outs(%sm0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %ov0 = tensor.empty() : tensor<32x32xf32>
  %ov_init = linalg.fill ins(%zero : f32) outs(%ov0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %out = linalg.matmul ins(%probs, %v : tensor<32x32xf32>, tensor<32x32xf32>)
                       outs(%ov_init : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %out : tensor<32x32xf32>
}
"""

torch.manual_seed(49)
Q = torch.randint(-4, 4, (32, 32), dtype=torch.int8).to(torch.float8_e4m3fn)
K = torch.randint(-4, 4, (32, 32), dtype=torch.int8).to(torch.float8_e4m3fn)
V = torch.randint(-4, 4, (32, 32), dtype=torch.int8).to(torch.float8_e4m3fn)
SCALE_TILE = torch.full((32, 32), 1.0 / math.sqrt(32.0), dtype=torch.bfloat16)
EXPECTED = attention_reference(Q, K, V, SCALE_TILE)

# Cross-check via IREE.
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
            ATTENTION_MLIR,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
        _cfg = runtime.Config("local-task")
        _ctx = runtime.SystemContext(config=_cfg)
        _ctx.add_vm_module(runtime.VmModule.copy_buffer(_ctx.instance, _vmfb))
        _iree_out = _ctx.modules.module["attention"](
            Q.to(torch.float32).numpy(),
            K.to(torch.float32).numpy(),
            V.to(torch.float32).numpy(),
            SCALE_TILE.float().numpy(),
        )
        _iree_bf16 = torch.from_numpy(np.array(_iree_out)).to(torch.bfloat16)
        _diff = (EXPECTED.float() - _iree_bf16.float()).abs().max().item()
        assert _diff < 1.5, f"MLIR vs PyTorch mismatch: {_diff}"
    except ImportError:
        pass

DRAM_Q = 0x0000
DRAM_K = 0x0400
DRAM_V = 0x0800
DRAM_SCALE = 0x0C00
DRAM_OUT_H0 = 0x1000
DRAM_OUT_H1 = 0x1400
EXPECTED_STACKED = torch.cat((EXPECTED[:, :16], EXPECTED[:, 16:]), dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# 4. NPU ISA program.
# ═══════════════════════════════════════════════════════════════════════════


class SmolVLAAttentionProgram(Program):
    """Auto-generated single-file Program for the ``attention`` kernel.

    ISA is lifted from the merlin kernel manifest (see
    ``benchmarks/SaturnNPU/kernel_library/manifest.json``). This Program
    mirrors the ``smolvla_silu.py`` template: self-contained, no cross-
    file helpers, torch-allclose golden check via ``pytest tests/test_programs.py``.
    """

    # Pair-op rewrite. npu_model _vmatmul computes activation @ weight
    # directly (no implicit transpose), so we push K as weight for Q@K and
    # V as weight for probs@V (no XLU needed).
    #
    # MRF layout:
    #   m0  = Q (fp8 tile)
    #   m2  = K (fp8 tile)
    #   m4  = V (fp8 tile)
    #   m6,m7   = scale (bf16 pair, duplicated halves)
    #   m10,m11 = scores (bf16 pair) = Q @ K
    #   m12,m13 = scaled
    #   m14,m15 = rowmax (broadcast)
    #   m16,m17 = scaled - rowmax
    #   m18,m19 = exp(scaled - rowmax)
    #   m20,m21 = rowsum(exp)
    #   m22,m23 = 1 / rowsum
    #   m24,m25 = probs (bf16)
    #   m26     = probs (fp8, same row ordering as m24,m25 via vpack)
    #   m28,m29 = out = probs @ V  (bf16 pair)
    #
    # VMEM staging (each tile = 1024 B = 32 mreg-words, stored back-to-back):
    #   x1 = 0x2000 Q             x2 = 0x2400 K            x3 = 0x2800 V
    #   x4 = 0x2C00 scale half 0  x5 = 0x3000 scale half 1
    #   x6 = 0x4000 OUT half 0    x7 = 0x4400 OUT half 1
    instructions: List[Instruction[Any]] = [
        # VMEM addresses
        Instruction("lui", ScalarArgs(rd=1, imm=0x2)),  # 0x2000
        Instruction("addi", ScalarArgs(rd=2, rs1=1, imm=1024)),  # 0x2400
        Instruction("addi", ScalarArgs(rd=3, rs1=2, imm=1024)),  # 0x2800
        Instruction("addi", ScalarArgs(rd=4, rs1=3, imm=1024)),  # 0x2C00
        Instruction("lui", ScalarArgs(rd=5, imm=0x3)),  # 0x3000
        Instruction("lui", ScalarArgs(rd=6, imm=0x4)),  # 0x4000
        Instruction("addi", ScalarArgs(rd=7, rs1=6, imm=1024)),  # 0x4400
        # DRAM addresses
        Instruction("addi", ScalarArgs(rd=8, rs1=0, imm=DRAM_Q)),  # 0x0000
        Instruction("addi", ScalarArgs(rd=9, rs1=8, imm=1024)),  # 0x0400 K
        Instruction("addi", ScalarArgs(rd=10, rs1=9, imm=1024)),  # 0x0800 V
        Instruction("addi", ScalarArgs(rd=11, rs1=10, imm=1024)),  # 0x0C00 scale
        Instruction("lui", ScalarArgs(rd=12, imm=0x1)),  # 0x1000 OUT_H0
        Instruction("addi", ScalarArgs(rd=13, rs1=12, imm=1024)),  # 0x1400 OUT_H1
        Instruction("addi", ScalarArgs(rd=14, rs1=0, imm=1024)),  # per-half size
        # DMA loads
        Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=8, rs2=14, channel=0)),  # Q
        Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=9, rs2=14, channel=1)),  # K
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=3, rs1=10, rs2=14, channel=0)),  # V
        Instruction(
            "dma.load.ch<N>", DmaArgs(rd=4, rs1=11, rs2=14, channel=1)
        ),  # scale half0
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        # scale is a full 32x32 broadcast tile (1024 B for each 32x16 half).
        # Source DRAM_SCALE is 32x32 bf16 = 2048 B. We already loaded first
        # 1024 B to VMEM[x4]; load the next 1024 B to VMEM[x5] from
        # DRAM_SCALE+1024.
        Instruction("addi", ScalarArgs(rd=15, rs1=11, imm=1024)),  # DRAM scale+1024
        Instruction("dma.load.ch<N>", DmaArgs(rd=5, rs1=15, rs2=14, channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        # Load MRF
        Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)),  # Q
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vload", VectorArgs(vd=2, rs1=2, imm12=0)),  # K
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vload", VectorArgs(vd=4, rs1=3, imm12=0)),  # V
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vload", VectorArgs(vd=6, rs1=4, imm12=0)),  # scale half 0
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vload", VectorArgs(vd=7, rs1=5, imm12=0)),  # scale half 1
        Instruction("delay", ScalarArgs(imm=34)),
        # Matmul 1: scores = Q @ K  (push K as weight; activation = Q)
        Instruction("vmatpush.weight.mxu0", VectorArgs(vd=0, vs1=2)),
        Instruction("delay", ScalarArgs(imm=32)),
        Instruction("vmatmul.mxu0", MatrixArgs(vd=0, vs1=0, vs2=0)),
        Instruction("delay", ScalarArgs(imm=96)),
        Instruction("vmatpop.bf16.acc.mxu0", MatrixArgs(vd=10, vs1=0)),  # (m10,m11)
        Instruction("delay", ScalarArgs(imm=32)),
        # Scale: (m12, m13) = scores * scale
        Instruction("vmul.bf16", VectorArgs(vd=12, vs1=10, vs2=6)),
        Instruction("delay", ScalarArgs(imm=66)),
        # Stable softmax
        Instruction("vredmax.row.bf16", VectorArgs(vd=14, vs1=12)),
        Instruction("delay", ScalarArgs(imm=69)),
        Instruction("vsub.bf16", VectorArgs(vd=16, vs1=12, vs2=14)),
        Instruction("delay", ScalarArgs(imm=66)),
        Instruction("vexp.bf16", VectorArgs(vd=18, vs1=16)),
        Instruction("delay", ScalarArgs(imm=66)),
        Instruction("vredsum.row.bf16", VectorArgs(vd=20, vs1=18)),
        Instruction("delay", ScalarArgs(imm=69)),
        Instruction("vrecip.bf16", VectorArgs(vd=22, vs1=20)),
        Instruction("delay", ScalarArgs(imm=66)),
        Instruction("vmul.bf16", VectorArgs(vd=24, vs1=18, vs2=22)),
        Instruction("delay", ScalarArgs(imm=66)),
        # Pack probs BF16 → FP8 into m26 with unit scale (same row layout).
        Instruction("seli", ScalarArgs(rd=0, imm=1)),
        Instruction("vpack.bf16.fp8", VectorArgs(vd=26, vs1=24, es1=0)),
        Instruction("delay", ScalarArgs(imm=66)),
        # Matmul 2: out = packed_probs @ V
        Instruction("vmatpush.weight.mxu0", VectorArgs(vd=0, vs1=4)),
        Instruction("delay", ScalarArgs(imm=32)),
        Instruction("vmatmul.mxu0", MatrixArgs(vd=0, vs1=26, vs2=0)),
        Instruction("delay", ScalarArgs(imm=96)),
        Instruction("vmatpop.bf16.acc.mxu0", MatrixArgs(vd=28, vs1=0)),  # (m28,m29)
        Instruction("delay", ScalarArgs(imm=32)),
        # Store out: m28 → VMEM[x6], m29 → VMEM[x7]; DMA both to DRAM.
        Instruction("vstore", VectorArgs(vd=28, rs1=6, imm12=0)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vstore", VectorArgs(vd=29, rs1=7, imm12=0)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("dma.store.ch<N>", DmaArgs(rd=12, rs1=6, rs2=14, channel=0)),
        Instruction("dma.store.ch<N>", DmaArgs(rd=13, rs1=7, rs2=14, channel=1)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_Q, Q),
        (DRAM_K, K),
        (DRAM_V, V),
        (DRAM_SCALE, SCALE_TILE),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUT_H0,
        EXPECTED_STACKED,
    )
    # fp8 quantization + softmax noise; allow looser tolerance.
    kernel_tolerance = (0.2, 0.2)
