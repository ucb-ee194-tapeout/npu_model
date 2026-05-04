"""SmolVLA elementwise_div kernel.

Elementwise divide. 217 instances in SmolVLA; 6 shape variants.
This Program implements the 32x32 canonical form. The ISA uses
``vrecip`` + ``vmul`` (no dedicated divide op).
"""

import torch
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER


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

DRAM_A_H0 = 0x80000000
DRAM_A_H1 = 0x80000400
DRAM_B_H0 = 0x80000800
# dram_in_3 (B_h1) at 0x80000C00; the kernel writes its first output half back
# at 0x80000C00 too (see manifest patch_points for elementwise_div). So
# DRAM_B_H1 and DRAM_OUT_H0 share the same address — the DMA store
# overwrites the B_h1 buffer in place once it's no longer needed.
DRAM_B_H1 = 0x80000C00
DRAM_OUT_H0 = 0x80000C00  # written after B_h1 is read into VMEM
DRAM_OUT_H1 = 0x80001000
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

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'smolvla_elementwise_div.S')

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_A_H0, INPUT_A[:, :16].contiguous()),
        (DRAM_A_H1, INPUT_A[:, 16:].contiguous()),
        (DRAM_B_H0, INPUT_B[:, :16].contiguous()),
        (DRAM_B_H1, INPUT_B[:, 16:].contiguous()),
    ]

    golden_result: list[tuple[int, torch.Tensor]] = [(
        DRAM_OUT_H0,
        EXPECTED_STACKED,
    )]
