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
try:
    import numpy as np
    import iree.compiler as compiler
    import iree.runtime as runtime

    _vmfb = compiler.compile_str(ATTENTION_MLIR, target_backends=["llvm-cpu"])
    _cfg = runtime.Config("local-task")
    _ctx = runtime.SystemContext(config=_cfg)
    _ctx.add_vm_module(runtime.VmModule.copy_buffer(_ctx.instance, _vmfb))
    _iree_out = _ctx.modules.module["attention"](
        Q.to(torch.float32).numpy(), K.to(torch.float32).numpy(),
        V.to(torch.float32).numpy(), SCALE_TILE.float().numpy(),
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
    file helpers, torch-allclose golden check via ``test_programs.py``.
    """

    instructions: List[Instruction[Any]] = [
        Instruction("lui", ScalarArgs(rd=1, imm=2)),
        Instruction("lui", ScalarArgs(rd=2, imm=2)),
        Instruction("addi", ScalarArgs(rd=2, rs1=2, imm=1024)),
        Instruction("lui", ScalarArgs(rd=3, imm=3)),
        Instruction("addi", ScalarArgs(rd=3, rs1=3, imm=-2048)),
        Instruction("lui", ScalarArgs(rd=4, imm=3)),
        Instruction("addi", ScalarArgs(rd=4, rs1=4, imm=-1024)),
        Instruction("lui", ScalarArgs(rd=5, imm=3)),
        Instruction("lui", ScalarArgs(rd=6, imm=3)),
        Instruction("addi", ScalarArgs(rd=6, rs1=6, imm=1024)),
        Instruction("addi", ScalarArgs(rd=7)),
        Instruction("addi", ScalarArgs(rd=8, imm=1024)),
        Instruction("lui", ScalarArgs(rd=9, imm=1)),
        Instruction("addi", ScalarArgs(rd=9, rs1=9, imm=-2048)),
        Instruction("lui", ScalarArgs(rd=10, imm=1)),
        Instruction("addi", ScalarArgs(rd=10, rs1=10, imm=-1024)),
        Instruction("lui", ScalarArgs(rd=11, imm=1)),
        Instruction("lui", ScalarArgs(rd=12, imm=1)),
        Instruction("addi", ScalarArgs(rd=12, rs1=12, imm=1024)),
        Instruction("addi", ScalarArgs(rd=13, imm=1024)),
        Instruction("dma.config.ch<N>", DmaArgs()),
        Instruction("dma.config.ch<N>", DmaArgs(channel=1)),
        Instruction("dma.wait.ch<N>", DmaArgs()),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=7, rs2=13)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=8, rs2=13, channel=1)),
        Instruction("dma.wait.ch<N>", DmaArgs()),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=3, rs1=9, rs2=13)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=4, rs1=10, rs2=13, channel=1)),
        Instruction("dma.wait.ch<N>", DmaArgs()),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        Instruction("vload", VectorArgs(rs1=1)),
        Instruction("vload", VectorArgs(vd=1, rs1=2)),
        Instruction("vload", VectorArgs(vd=2, rs1=3)),
        Instruction("vload", VectorArgs(vd=3, rs1=4)),
        Instruction("vmatpush.weight.mxu0", VectorArgs(vs1=1)),
        Instruction("delay", ScalarArgs(imm=17)),
        Instruction("vmatmul.mxu0", MatrixArgs()),
        Instruction("delay", ScalarArgs(imm=33)),
        Instruction("vmatpop.bf16.acc.mxu0", VectorArgs(vd=4)),
        Instruction("vmul.bf16", VectorArgs(vd=6, vs1=4, vs2=3)),
        Instruction("vmul.bf16", VectorArgs(vd=7, vs1=5, vs2=3)),
        Instruction("delay", ScalarArgs(imm=2)),
        Instruction("vredmax.row.bf16", VectorArgs(vd=8, vs1=6)),
        Instruction("vredmax.row.bf16", VectorArgs(vd=9, vs1=7)),
        Instruction("delay", ScalarArgs(imm=8)),
        Instruction("vmaximum.bf16", VectorArgs(vd=10, vs1=8, vs2=9)),
        Instruction("delay", ScalarArgs(imm=2)),
        Instruction("vsub.bf16", VectorArgs(vd=11, vs1=6, vs2=10)),
        Instruction("vsub.bf16", VectorArgs(vd=12, vs1=7, vs2=10)),
        Instruction("delay", ScalarArgs(imm=2)),
        Instruction("vexp.bf16", VectorArgs(vd=13, vs1=11)),
        Instruction("vexp.bf16", VectorArgs(vd=14, vs1=12)),
        Instruction("delay", ScalarArgs(imm=8)),
        Instruction("vredsum.row.bf16", VectorArgs(vd=15, vs1=13)),
        Instruction("vredsum.row.bf16", VectorArgs(vd=16, vs1=14)),
        Instruction("delay", ScalarArgs(imm=8)),
        Instruction("vadd.bf16", VectorArgs(vd=17, vs1=15, vs2=16)),
        Instruction("delay", ScalarArgs(imm=2)),
        Instruction("vrecip.bf16", VectorArgs(vd=18, vs1=17)),
        Instruction("delay", ScalarArgs(imm=8)),
        Instruction("vmul.bf16", VectorArgs(vd=19, vs1=13, vs2=18)),
        Instruction("vmul.bf16", VectorArgs(vd=20, vs1=14, vs2=18)),
        Instruction("delay", ScalarArgs(imm=2)),
        Instruction("seli", ScalarArgs(imm=1)),
        Instruction("vpack.bf16.fp8", VectorArgs(vd=21, vs1=19)),
        Instruction("delay", ScalarArgs(imm=8)),
        Instruction("vmatpush.weight.mxu0", VectorArgs(vs1=2)),
        Instruction("delay", ScalarArgs(imm=17)),
        Instruction("vmatmul.mxu0", MatrixArgs(vs1=21)),
        Instruction("delay", ScalarArgs(imm=33)),
        Instruction("vmatpop.bf16.acc.mxu0", VectorArgs(vd=22)),
        Instruction("vstore", VectorArgs(vd=22, rs1=5)),
        Instruction("vstore", VectorArgs(vd=23, rs1=6)),
        Instruction("dma.store.ch<N>", DmaArgs(rd=11, rs1=5, rs2=13)),
        Instruction("dma.store.ch<N>", DmaArgs(rd=12, rs1=6, rs2=13, channel=1)),
        Instruction("dma.wait.ch<N>", DmaArgs()),
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
