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
try:
    import numpy as np
    import iree.compiler as compiler
    import iree.runtime as runtime

    _vmfb = compiler.compile_str(SOFTMAX_MLIR, target_backends=["llvm-cpu"])
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


# ═══════════════════════════════════════════════════════════════════════════
# 4. NPU ISA program.
# ═══════════════════════════════════════════════════════════════════════════

class SmolVLASoftmaxProgram(Program):
    """Auto-generated single-file Program for the ``softmax`` kernel.

    ISA is lifted from the merlin kernel manifest (see
    ``benchmarks/SaturnNPU/kernel_library/manifest.json``). This Program
    mirrors the ``smolvla_silu.py`` template: self-contained, no cross-
    file helpers, torch-allclose golden check via ``test_programs.py``.
    """

    instructions: List[Instruction[Any]] = [
        Instruction("lui", ScalarArgs(rd=1, imm=2)),
        Instruction("addi", ScalarArgs(rd=2, rs1=1, imm=1024)),
        Instruction("lui", ScalarArgs(rd=3, imm=3)),
        Instruction("addi", ScalarArgs(rd=4, rs1=3, imm=1024)),
        Instruction("addi", ScalarArgs(rd=5)),
        Instruction("addi", ScalarArgs(rd=6, imm=1024)),
        Instruction("addi", ScalarArgs(rd=7, imm=2047)),
        Instruction("addi", ScalarArgs(rd=7, rs1=7, imm=769)),
        Instruction("addi", ScalarArgs(rd=8, rs1=7, imm=1024)),
        Instruction("addi", ScalarArgs(rd=9, imm=1024)),
        Instruction("dma.config.ch<N>", DmaArgs()),
        Instruction("dma.config.ch<N>", DmaArgs(channel=1)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=5, rs2=9)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=6, rs2=9, channel=1)),
        Instruction("dma.wait.ch<N>", DmaArgs()),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        Instruction("vload", VectorArgs(rs1=1)),
        Instruction("vload", VectorArgs(vd=1, rs1=2)),
        Instruction("vredmax.row.bf16", VectorArgs(vd=2)),
        Instruction("vredmax.row.bf16", VectorArgs(vd=3, vs1=1)),
        Instruction("delay", ScalarArgs(imm=8)),
        Instruction("vmaximum.bf16", VectorArgs(vd=4, vs1=2, vs2=3)),
        Instruction("delay", ScalarArgs(imm=2)),
        Instruction("vsub.bf16", VectorArgs(vd=5, vs2=4)),
        Instruction("vsub.bf16", VectorArgs(vd=6, vs1=1, vs2=4)),
        Instruction("delay", ScalarArgs(imm=2)),
        Instruction("vexp.bf16", VectorArgs(vd=7, vs1=5)),
        Instruction("vexp.bf16", VectorArgs(vd=8, vs1=6)),
        Instruction("delay", ScalarArgs(imm=8)),
        Instruction("vredsum.row.bf16", VectorArgs(vd=9, vs1=7)),
        Instruction("vredsum.row.bf16", VectorArgs(vd=10, vs1=8)),
        Instruction("delay", ScalarArgs(imm=8)),
        Instruction("vadd.bf16", VectorArgs(vd=11, vs1=9, vs2=10)),
        Instruction("delay", ScalarArgs(imm=2)),
        Instruction("vrecip.bf16", VectorArgs(vd=12, vs1=11)),
        Instruction("delay", ScalarArgs(imm=8)),
        Instruction("vmul.bf16", VectorArgs(vd=13, vs1=7, vs2=12)),
        Instruction("vmul.bf16", VectorArgs(vd=14, vs1=8, vs2=12)),
        Instruction("delay", ScalarArgs(imm=2)),
        Instruction("vstore", VectorArgs(vd=13, rs1=3)),
        Instruction("vstore", VectorArgs(vd=14, rs1=4)),
        Instruction("dma.store.ch<N>", DmaArgs(rd=7, rs1=3, rs2=9)),
        Instruction("dma.store.ch<N>", DmaArgs(rd=8, rs1=4, rs2=9, channel=1)),
        Instruction("dma.wait.ch<N>", DmaArgs()),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_X_H0, INPUT[:, :16].contiguous()),
        (DRAM_X_H1, INPUT[:, 16:].contiguous()),
    ]


    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUT_H0,
        EXPECTED[:, :16],
    )

