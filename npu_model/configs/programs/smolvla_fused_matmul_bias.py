"""SmolVLA fused_matmul_bias kernel: fp8 matmul then bf16 bias add.

MLIR sources (benchmarks/SaturnNPU/kernels/fused_matmul_bias):
    op_a.mlir — quantized_matmul_fp8:
        A f8E4M3FN × B f8E4M3FN → bf16 (extf fp8→bf16, mulf bf16, addf bf16)
    op_b.mlir — elementwise_add: matmul(bf16) + bias(bf16) → bf16
Pattern appears 286× in the model. Fusing eliminates the intermediate
matmul output materialisation.

Precision matches the MLIR exactly: A and B are fp8 (stored pre-
quantized in DRAM, matching ``smolvla_matmul.py``); bias and output
are bf16.

MLIR → ISA mapping:
    op_a (fp8 matmul):
        vmatpush.weight.mxu0   — push B_fp8 into WB slot 0
        vmatmul.mxu0           — acc[0] = A_fp8 @ WB[0], bf16 accumulator
        vmatpop.bf16.acc.mxu0  — acc[0] → (m_vd, m_vd+1) bf16 pair
    op_b (bias add):
        vadd.bf16              — pair-op add of matmul result + bias

Demo tile: (32, 32). Full op_a shape is
    A: tensor<3x512x512xf8E4M3FN>, B: tensor<768x3x16x16xf8E4M3FN>,
    out: tensor<768x32x32xbf16>.
This Program exercises one 32x32 sub-block.
"""

import os
import torch
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER


# ═══════════════════════════════════════════════════════════════════════════
# 1. MLIR definition — op_a + op_b fused. f32 inputs for IREE llvm-cpu; the
# simulator's fp8 matmul vs PyTorch f32 matmul differ so the crosscheck
# tolerance is loose.
# ═══════════════════════════════════════════════════════════════════════════

FUSED_MATMUL_BIAS_MLIR = """\
func.func @fused_matmul_bias(
    %A: tensor<32x32xf32>, %B: tensor<32x32xf32>, %bias: tensor<32x32xf32>
) -> tensor<32x32xf32> {
  %zero = arith.constant 0.0 : f32
  %empty_mm = tensor.empty() : tensor<32x32xf32>
  %mm_init = linalg.fill ins(%zero : f32)
                         outs(%empty_mm : tensor<32x32xf32>) -> tensor<32x32xf32>
  %mm = linalg.matmul
      ins(%A, %B : tensor<32x32xf32>, tensor<32x32xf32>)
      outs(%mm_init : tensor<32x32xf32>) -> tensor<32x32xf32>
  %empty = tensor.empty() : tensor<32x32xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%mm, %bias : tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%empty : tensor<32x32xf32>) {
  ^bb0(%in_mm: f32, %in_b: f32, %out: f32):
    %res = arith.addf %in_mm, %in_b : f32
    linalg.yield %res : f32
  } -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}
"""


# ═══════════════════════════════════════════════════════════════════════════
# 2. PyTorch reference.
# ═══════════════════════════════════════════════════════════════════════════


def fused_matmul_bias_reference(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """A_fp8 @ B_fp8 → bf16, then + bias. Matches MXU semantics:
    extf fp8→bf16 inputs, bf16 accumulator, bf16 bias-add."""
    mm = (a.to(torch.float32) @ b.to(torch.float32)).to(torch.bfloat16)
    return (mm.float() + bias.float()).to(torch.bfloat16)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Golden data — fp8 A and B, bf16 bias.
# ═══════════════════════════════════════════════════════════════════════════

torch.manual_seed(42)
INPUT_A = torch.randint(-8, 8, (32, 32), dtype=torch.int8).to(torch.float8_e4m3fn)
INPUT_B = torch.randint(-8, 8, (32, 32), dtype=torch.int8).to(torch.float8_e4m3fn)
BIAS = torch.randn(32, 32, dtype=torch.bfloat16)
EXPECTED = fused_matmul_bias_reference(INPUT_A, INPUT_B, BIAS)
# MXU pair layout is col-blocked: m[vd] = cols 0:16, m[vd+1] = cols 16:32
# (see ``write_mrf_bf16_tile``). Both the bias input and the output must
# follow the same stacked-halves layout so element offsets line up.
BIAS_STACKED = torch.cat((BIAS[:, :16], BIAS[:, 16:]), dim=0)  # (64, 16) bf16
# Kernel writes output as two 32x16 halves stacked: cols 0-15 then 16-31.
EXPECTED_STACKED = torch.cat((EXPECTED[:, :16], EXPECTED[:, 16:]), dim=0)

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
            FUSED_MATMUL_BIAS_MLIR,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
        _cfg = runtime.Config("local-task")
        _ctx = runtime.SystemContext(config=_cfg)
        _ctx.add_vm_module(runtime.VmModule.copy_buffer(_ctx.instance, _vmfb))
        _iree_out = _ctx.modules.module["fused_matmul_bias"](
            INPUT_A.to(torch.float32).numpy(),
            INPUT_B.to(torch.float32).numpy(),
            BIAS.float().numpy(),
        )
        _iree_bf16 = torch.from_numpy(np.array(_iree_out)).to(torch.bfloat16)
        # IREE uses full f32 matmul; simulator uses fp8 × fp8 → bf16. Allow
        # a loose tolerance (fp8 rounding noise on top of bias add).
        _diff = (EXPECTED.float() - _iree_bf16.float()).abs().max().item()
        assert _diff < 4.0, f"MLIR vs PyTorch mismatch: {_diff}"
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 4. Memory layout.
#
# DRAM:
#   [0x0000..0x03FF] A_fp8  (32x32 fp8 = 1024 B)
#   [0x0400..0x07FF] B_fp8  (32x32 fp8 = 1024 B)
#   [0x0800..0x0FFF] bias   (32x32 bf16 = 2048 B, two 32x16 halves)
#   [0x1000..0x17FF] output (32x32 bf16 = 2048 B, column-blocked)
# ═══════════════════════════════════════════════════════════════════════════

DRAM_A = 0x0000
DRAM_B = 0x0400
DRAM_BIAS = 0x0800
DRAM_OUT = 0x1000
FP8_BYTES = 1024
BF16_BYTES = 2048


# ═══════════════════════════════════════════════════════════════════════════
# 5. NPU ISA program (pair-op BF16 for VPU ops; MXU for matmul).
#
# MRF map:
#   m0         = A fp8 (full 32x32 tile in one mreg)
#   m2         = B fp8
#   (m4, m5)   = bias bf16 pair
#   (m6, m7)   = matmul result bf16 (from vmatpop.bf16.acc.mxu0 vd=6)
#   (m8, m9)   = out = matmul + bias bf16 pair
#
# Scalar reg map:
#   x1 = VMEM A base      x2 = VMEM B base
#   x3 = VMEM bias base   x4 = VMEM out base
#   x5 = DRAM A base      x6 = DRAM B base      x7 = DRAM bias base
#   x8 = DRAM out base    x9 = 1024 (fp8 tile)  x10 = 2048 (bf16 tile)
# ═══════════════════════════════════════════════════════════════════════════


class SmolVLAFusedMatmulBiasProgram(Program):
    """fused_matmul_bias: (A_fp8 @ B_fp8)_bf16 + bias_bf16."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'smolvla_fused_matmul_bias.S')

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_A, INPUT_A),
        (DRAM_B, INPUT_B),
        (DRAM_BIAS, BIAS_STACKED),
    ]

    golden_result: list[tuple[int, torch.Tensor]] = [(
        DRAM_OUT,
        EXPECTED_STACKED,
    )]
