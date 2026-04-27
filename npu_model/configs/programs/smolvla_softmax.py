"""SmolVLA softmax kernel.

Row-wise stable softmax on a 32x32 bf16 tile: subtract rowmax,
exponentiate, divide by rowsum. 23 instances in SmolVLA; 3
shape variants.

Writes output as two 32x16 halves at dram_out_0 / dram_out_1.
"""

import torch
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER


# ═══════════════════════════════════════════════════════════════════════════
# 2. PyTorch reference.
# ═══════════════════════════════════════════════════════════════════════════


def softmax_reference(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    xm = xf - xf.max(dim=-1, keepdim=True).values
    ex = xm.exp()
    return (ex / ex.sum(dim=-1, keepdim=True)).to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Golden data.
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────
# MLIR: row-wise softmax. Modeled after
# ``benchmarks/SaturnNPU/kernels/linalg.softmax/variant_*.mlir``
# (a single ``linalg.softmax dimension(2)`` call over a bf16 tensor).
# Retyped to f32 here so stock llvm-cpu can lower without bf16 buffer
# interop.
# ───────────────────────────────────────────────────────────────────────

SOFTMAX_MLIR = """\
func.func @softmax(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %out0 = tensor.empty() : tensor<32x32xf32>
  %result = linalg.softmax dimension(1) ins(%x : tensor<32x32xf32>)
                                         outs(%out0 : tensor<32x32xf32>)
                                         -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}
"""

torch.manual_seed(50)
INPUT = torch.randn(32, 32, dtype=torch.bfloat16) * 2.0
EXPECTED = softmax_reference(INPUT)

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
            SOFTMAX_MLIR,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
        _cfg = runtime.Config("local-task")
        _ctx = runtime.SystemContext(config=_cfg)
        _ctx.add_vm_module(runtime.VmModule.copy_buffer(_ctx.instance, _vmfb))
        _iree_out = _ctx.modules.module["softmax"](INPUT.float().numpy())
        _iree_bf16 = torch.from_numpy(np.array(_iree_out)).to(torch.bfloat16)
        _diff = (EXPECTED.float() - _iree_bf16.float()).abs().max().item()
        assert _diff < 5e-2, f"MLIR vs PyTorch mismatch: {_diff}"
    except ImportError:
        pass

DRAM_X_H0 = 0x0000
DRAM_X_H1 = 0x0400
DRAM_OUT_H0 = 0x0B00
DRAM_OUT_H1 = 0x0F00
EXPECTED_STACKED = torch.cat((EXPECTED[:, :16], EXPECTED[:, 16:]), dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# 4. NPU ISA program.
# ═══════════════════════════════════════════════════════════════════════════


class SmolVLASoftmaxProgram(Program):
    """Auto-generated single-file Program for the ``softmax`` kernel.

    ISA is lifted from the merlin kernel manifest (see
    ``benchmarks/SaturnNPU/kernel_library/manifest.json``). This Program
    mirrors the ``smolvla_silu.py`` template: self-contained, no cross-
    file helpers, torch-allclose golden check via ``pytest tests/test_programs.py``.
    """

    # Pair-op BF16 layout:
    #   (m0, m1)   = X                         (m2, m3)   = rowmax(X)
    #   (m4, m5)   = X - rowmax                (m6, m7)   = exp(X - rowmax)
    #   (m8, m9)   = rowsum(exp)               (m10, m11) = 1 / rowsum
    #   (m12, m13) = exp * inv_rowsum = Y
    #
    # VMEM layout:
    #   x1 = VMEM X base (two 1024-B halves stacked: m0 at +0, m1 at +1024)
    #   x2 = VMEM OUT base (m12 at +0, m13 at +1024 → DMA both as one block)
    #   x3 = DRAM X_H0, x4 = DRAM X_H1
    #   x5 = DRAM OUT_H0 (m12 lands here, m13 lands at +1024 == OUT_H1)
    #   x6 = 1024  (per-half transfer size)
    #   x7 = x1 + 1024  (second-half VMEM addr for DMA.LOAD)
    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'smolvla_softmax.S')

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_X_H0, INPUT[:, :16].contiguous()),
        (DRAM_X_H1, INPUT[:, 16:].contiguous()),
    ]

    golden_result: list[tuple[int, torch.Tensor]] = [(
        DRAM_OUT_H0,
        EXPECTED_STACKED,
    )]
