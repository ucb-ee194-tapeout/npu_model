"""SmolVLA softmax kernel.

Row-wise stable softmax on a 32x32 bf16 tile: subtract rowmax,
exponentiate, divide by rowsum. 23 instances in SmolVLA; 3
shape variants.

Writes output as two 32x16 halves at dram_out_0 / dram_out_1.
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs


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
    instructions: List[Instruction[Any]] = [
        Instruction("lui", ScalarArgs(rd=1, imm=0x2)),  # x1 = 0x2000 VMEM X base
        Instruction("lui", ScalarArgs(rd=2, imm=0x3)),  # x2 = 0x3000 VMEM OUT base
        Instruction("addi", ScalarArgs(rd=3, rs1=0, imm=DRAM_X_H0)),  # x3 = 0x0000
        Instruction("addi", ScalarArgs(rd=4, rs1=3, imm=1024)),  # x4 = 0x0400
        Instruction("lui", ScalarArgs(rd=5, imm=0x1)),  # x5 = 0x1000
        Instruction(
            "addi", ScalarArgs(rd=5, rs1=5, imm=-1280)
        ),  # x5 = 0x0B00 DRAM_OUT_H0
        Instruction("addi", ScalarArgs(rd=6, rs1=0, imm=1024)),  # x6 = 1024
        Instruction("addi", ScalarArgs(rd=7, rs1=1, imm=1024)),  # x7 = x1 + 1024
        # DMA X → VMEM: two halves back-to-back (m0 at +0, m1 at +1024)
        Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=0)),
        Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=1)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=3, rs2=6, channel=0)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=7, rs1=4, rs2=6, channel=1)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        # Load X pair
        Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)),
        Instruction("delay", ScalarArgs(imm=16)),
        Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=32)),
        Instruction("delay", ScalarArgs(imm=16)),
        # (m2, m3) = rowmax(X) broadcast
        Instruction("vredmax.row.bf16", VectorArgs(vd=2, vs1=0)),
        Instruction("delay", ScalarArgs(imm=4)),
        # (m4, m5) = X - rowmax
        Instruction("vsub.bf16", VectorArgs(vd=4, vs1=0, vs2=2)),
        Instruction("delay", ScalarArgs(imm=4)),
        # (m6, m7) = exp(X - rowmax)
        Instruction("vexp.bf16", VectorArgs(vd=6, vs1=4)),
        Instruction("delay", ScalarArgs(imm=16)),
        # (m8, m9) = rowsum(exp)
        Instruction("vredsum.row.bf16", VectorArgs(vd=8, vs1=6)),
        Instruction("delay", ScalarArgs(imm=4)),
        # (m10, m11) = 1 / rowsum
        Instruction("vrecip.bf16", VectorArgs(vd=10, vs1=8)),
        Instruction("delay", ScalarArgs(imm=16)),
        # (m12, m13) = exp * inv_rowsum
        Instruction("vmul.bf16", VectorArgs(vd=12, vs1=6, vs2=10)),
        Instruction("delay", ScalarArgs(imm=4)),
        # Store m12 and m13 back-to-back in VMEM, DMA both 1024-B halves.
        Instruction("vstore", VectorArgs(vd=12, rs1=2, imm12=0)),
        Instruction("delay", ScalarArgs(imm=16)),
        Instruction("vstore", VectorArgs(vd=13, rs1=2, imm12=32)),
        Instruction("delay", ScalarArgs(imm=16)),
        # Two DMA stores of 1024 bytes each (m12 → OUT_H0, m13 → OUT_H1).
        Instruction("addi", ScalarArgs(rd=8, rs1=5, imm=1024)),  # x8 = DRAM_OUT_H1
        Instruction("addi", ScalarArgs(rd=9, rs1=2, imm=1024)),  # x9 = VMEM m13 addr
        Instruction("dma.store.ch<N>", DmaArgs(rd=5, rs1=2, rs2=6, channel=0)),
        Instruction("dma.store.ch<N>", DmaArgs(rd=8, rs1=9, rs2=6, channel=1)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_X_H0, INPUT[:, :16].contiguous()),
        (DRAM_X_H1, INPUT[:, 16:].contiguous()),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUT_H0,
        EXPECTED_STACKED,
    )
