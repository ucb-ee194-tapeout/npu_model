"""SmolVLA gelu_tanh kernel.

GELU activation with the tanh approximation:
    y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
Appears 36 times in SmolVLA (SigLIP MLPs).

The handwritten ISA loads polynomial-table constants at
``dram_in_1..dram_in_4`` (four pre-computed scalars broadcast to
32x16 tiles). Inputs are the two split halves of x at
``dram_in_0`` and a companion address; outputs are the two
32x16 halves of GELU(x).
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
# The handwritten ISA expects 5 inputs (x plus 4 tabulated constants)
# at DRAM offsets 0x0, 0x400, 0x800, 0xc00, 0x1000. The golden fixture
# here is a placeholder — to numerically validate this kernel, populate
# all 5 regions with the actual polynomial-table values from the
# original AutoComp source. For now we ship a xfail-style smoke.
DRAM_X_H0 = 0x0000
DRAM_X_H1 = 0x0400
DRAM_C1 = 0x0800  # constant tables
DRAM_C2 = 0x0C00
DRAM_C3 = 0x1000
DRAM_OUT_H0 = 0x1400
DRAM_OUT_H1 = 0x1800


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
    ]

    # No golden_result set — kernel body depends on constants not yet
    # provided. Define ``golden_result`` once you have them.
    # golden_result = (DRAM_OUT_H0, EXPECTED[:, :16])
