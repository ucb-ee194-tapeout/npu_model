"""SmolVLA attention kernel.

Single-tile scaled dot-product attention: ``softmax((Q @ K) * scale) @ V``.
Q, K, V are fp8 32x32 tiles; ``scale`` is an elementwise bf16 mask/scale.
Writes the bf16 output as two 32x16 halves.
"""

import torch
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER

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

DRAM_Q = 0x80000000
DRAM_K = 0x80000400
DRAM_V = 0x80000800
DRAM_SCALE = 0x80000C00
DRAM_OUT_H0 = 0x80001000
DRAM_OUT_H1 = 0x80001400
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

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'smolvla_attention.S')

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_Q, Q),
        (DRAM_K, K),
        (DRAM_V, V),
        (DRAM_SCALE, SCALE_TILE),
    ]

    golden_result: list[tuple[int, torch.Tensor]] = [(
        DRAM_OUT_H0,
        EXPECTED_STACKED,
    )]
    # fp8 quantization + softmax noise; allow looser tolerance.
    kernel_tolerance = (0.2, 0.2)
