"""SmolVLA gelu_tanh kernel.

GELU activation with the tanh approximation:
    y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
Appears 36 times in SmolVLA (SigLIP MLPs).

Pair-op BF16 layout (LMUL=2):
  (m0, m1)   = x                  (m2, m3)   = 0.044715 broadcast pair
  (m4, m5)   = sqrt(2/pi) pair    (m6, m7)   = 0.5 pair
  m8 = m9    = 1.0  (via vli.all)
  (m10, m11) = x^2     (m12, m13) = x^3
  (m14, m15) = 0.044715*x^3       (m16, m17) = x + 0.044715*x^3
  (m18, m19) = sqrt(2/pi)*(x+0.044715*x^3)
  (m20, m21) = tanh(...)          (m22, m23) = 1 + tanh(...)
  (m24, m25) = 0.5*x              (m26, m27) = GELU(x) = 0.5*x*(1+tanh(...))

DRAM layout (1024 B per 32x16 tile):
  DRAM_X_H0    = 0x0000  x half 0
  DRAM_X_H1    = 0x0400  x half 1
  DRAM_C044    = 0x0800  0.044715 broadcast tile (loaded into m2 AND m3)
  DRAM_CSQRT   = 0x0C00  sqrt(2/pi) broadcast tile (loaded into m4 AND m5)
  DRAM_CHALF   = 0x1000  0.5 broadcast tile (loaded into m6 AND m7)
  DRAM_OUT_H0  = 0x1400  output half 0
  DRAM_OUT_H1  = 0x1800  output half 1
"""

import torch
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER


# ═══════════════════════════════════════════════════════════════════════════
# 2. PyTorch reference.
# ═══════════════════════════════════════════════════════════════════════════

import math


def gelu_tanh_reference(x: torch.Tensor) -> torch.Tensor:
    return (
        x.float()
        * 0.5
        * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x.float() + 0.044715 * x.float().pow(3.0))
            )
        )
    ).to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Golden data.
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────
# MLIR: GELU (tanh approximation). Transcribed from
# ``benchmarks/SaturnNPU/kernels/gelu_tanh/variant_0_1024_3072_bf16.mlir``
# with the shape reduced to a 32x32 tile and constants spelled out.
#   y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# ───────────────────────────────────────────────────────────────────────

GELU_TANH_MLIR = """\
func.func @gelu_tanh(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %c3    = arith.constant 3 : i64
  %k044  = arith.constant 0.044715 : f32
  %ksqrt = arith.constant 0.7978845 : f32
  %one   = arith.constant 1.0 : f32
  %half  = arith.constant 0.5 : f32
  %out0  = tensor.empty() : tensor<32x32xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%x : tensor<32x32xf32>) outs(%out0 : tensor<32x32xf32>) {
  ^bb0(%in: f32, %_: f32):
    %x3 = math.fpowi %in, %c3 : f32, i64
    %t1 = arith.mulf %x3, %k044 : f32
    %t2 = arith.addf %in, %t1 : f32
    %t3 = arith.mulf %t2, %ksqrt : f32
    %t4 = math.tanh %t3 : f32
    %t5 = arith.addf %t4, %one : f32
    %t6 = arith.mulf %t5, %half : f32
    %y  = arith.mulf %in, %t6 : f32
    linalg.yield %y : f32
  } -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}
"""

torch.manual_seed(42)
INPUT = torch.randn(32, 32, dtype=torch.bfloat16) * 0.5
EXPECTED = gelu_tanh_reference(INPUT)

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
            GELU_TANH_MLIR,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
        _cfg = runtime.Config("local-task")
        _ctx = runtime.SystemContext(config=_cfg)
        _ctx.add_vm_module(runtime.VmModule.copy_buffer(_ctx.instance, _vmfb))
        _iree_out = _ctx.modules.module["gelu_tanh"](INPUT.float().numpy())
        _iree_bf16 = torch.from_numpy(np.array(_iree_out)).to(torch.bfloat16)
        _diff = (EXPECTED.float() - _iree_bf16.float()).abs().max().item()
        assert _diff < 5e-2, f"MLIR vs PyTorch mismatch: {_diff}"
    except ImportError:
        pass
DRAM_X_H0 = 0x80000000
DRAM_X_H1 = 0x80000400
DRAM_C044 = 0x80000800  # 0.044715 broadcast tile
DRAM_CSQRT = 0x80000C00  # sqrt(2/pi) broadcast tile
DRAM_CHALF = 0x80001000  # 0.5 broadcast tile
DRAM_OUT_H0 = 0x80001400
DRAM_OUT_H1 = 0x80001800

_c044 = torch.full((32, 16), 0.044715, dtype=torch.bfloat16)
_csqrt = torch.full((32, 16), math.sqrt(2.0 / math.pi), dtype=torch.bfloat16)
_chalf = torch.full((32, 16), 0.5, dtype=torch.bfloat16)
EXPECTED_STACKED = torch.cat((EXPECTED[:, :16], EXPECTED[:, 16:]), dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# 4. NPU ISA program.
# ═══════════════════════════════════════════════════════════════════════════


class SmolVLAGeluTanhProgram(Program):
    """Auto-generated single-file Program for the ``gelu_tanh`` kernel.

    ISA is lifted from the merlin kernel manifest (see
    ``benchmarks/SaturnNPU/kernel_library/manifest.json``). This Program
    mirrors the ``smolvla_silu.py`` template: self-contained, no cross-
    file helpers, torch-allclose golden check via ``pytest tests/test_programs.py``.
    """

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'smolvla_gelu_tanh.S')

    # Leaving gelu_tanh inputs minimal — tabulated constants default
    # to zero, which means the numerical check below is a smoke
    # (will fail unless the original polynomial-table values are
    # supplied). Populate DRAM_C1..C3 with the autocomp constants to
    # enable torch.allclose.
    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_X_H0, INPUT[:, :16].contiguous()),
        (DRAM_X_H1, INPUT[:, 16:].contiguous()),
        (DRAM_C044, _c044),
        (DRAM_CSQRT, _csqrt),
        (DRAM_CHALF, _chalf),
    ]

    golden_result: list[tuple[int, torch.Tensor]] = [(DRAM_OUT_H0, EXPECTED_STACKED)]
