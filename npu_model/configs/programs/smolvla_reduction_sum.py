"""SmolVLA reduction_sum kernel.

Row-wise sum of a 32x32 bf16 tile, broadcast across cols to a
32x16 output tile. 214 instances in SmolVLA; 3 shape variants.
The kernel adds the two 32x16 halves elementwise first, then
reduces rows. Match that accumulation order in the reference.
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs


# ═══════════════════════════════════════════════════════════════════════════
# 2. PyTorch reference.
# ═══════════════════════════════════════════════════════════════════════════


def reduction_sum_reference(x: torch.Tensor) -> torch.Tensor:
    # Pair-op vredsum.row.bf16 sums all 32 columns of the (32,32) pair in
    # one shot and broadcasts the scalar row-sum back to both 16-col halves.
    row_sum = x.sum(dim=-1, keepdim=True).to(torch.bfloat16)
    return row_sum.expand(-1, 16).contiguous().to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Golden data.
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────
# MLIR: row-wise sum-reduce. Modeled after
# ``benchmarks/SaturnNPU/kernels/reduction_sum/variant_*.mlir``
# (linalg.generic reduction over d1), adapted to 32x32 tile + broadcast
# of the scalar row-sum back out to 16 columns (the hardware stores a
# single column of row-sums replicated).
# ───────────────────────────────────────────────────────────────────────

REDUCTION_SUM_MLIR = """\
func.func @reduction_sum(%x: tensor<32x32xf32>) -> tensor<32x16xf32> {
  %init0 = tensor.empty() : tensor<32xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32)
                      outs(%init0 : tensor<32xf32>) -> tensor<32xf32>
  %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
  } ins(%x : tensor<32x32xf32>) outs(%init : tensor<32xf32>) {
  ^bb0(%in: f32, %acc: f32):
    %s = arith.addf %acc, %in : f32
    linalg.yield %s : f32
  } -> tensor<32xf32>
  %out0 = tensor.empty() : tensor<32x16xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%sum : tensor<32xf32>) outs(%out0 : tensor<32x16xf32>) {
  ^bb0(%in: f32, %_: f32):
    linalg.yield %in : f32
  } -> tensor<32x16xf32>
  return %result : tensor<32x16xf32>
}
"""

torch.manual_seed(46)
INPUT = torch.randn(32, 32, dtype=torch.bfloat16)
EXPECTED = reduction_sum_reference(INPUT)

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
            REDUCTION_SUM_MLIR,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
        _cfg = runtime.Config("local-task")
        _ctx = runtime.SystemContext(config=_cfg)
        _ctx.add_vm_module(runtime.VmModule.copy_buffer(_ctx.instance, _vmfb))
        _iree_out = _ctx.modules.module["reduction_sum"](INPUT.float().numpy())
        _iree_f32 = torch.from_numpy(np.array(_iree_out)).to(torch.bfloat16)
        _diff = (EXPECTED.float() - _iree_f32.float()).abs().max().item()
        assert _diff < 5e-1, f"MLIR vs PyTorch mismatch: {_diff}"
    except ImportError:
        pass

DRAM_X_H0 = 0x0000
DRAM_X_H1 = 0x0400
DRAM_OUT = 0x0B00


# ═══════════════════════════════════════════════════════════════════════════
# 4. NPU ISA program.
# ═══════════════════════════════════════════════════════════════════════════


class SmolVLAReductionSumProgram(Program):
    """Auto-generated single-file Program for the ``reduction_sum`` kernel.

    ISA is lifted from the merlin kernel manifest (see
    ``benchmarks/SaturnNPU/kernel_library/manifest.json``). This Program
    mirrors the ``smolvla_silu.py`` template: self-contained, no cross-
    file helpers, torch-allclose golden check via ``pytest tests/test_programs.py``.
    """

    # Pair-op BF16 layout:
    #   (m0, m1) = X  (32x32 tile: m0 = cols 0:16, m1 = cols 16:32)
    #   (m4, m5) = vredsum.row broadcast over X
    # Only the first half m4 (== m5 under broadcast) is stored to DRAM_OUT
    # since EXPECTED is shape (32, 16).
    #
    # VMEM layout:
    #   x1  = VMEM base for X half0 (both halves staged at same VMEM base,
    #         using imm12=0 for m0 and imm12=32 for m1 so they occupy the
    #         next 32 mreg-words i.e. +1024 bytes).
    #   x2  = VMEM out base
    #   x3  = DRAM X_H0 = 0x0000
    #   x4  = DRAM X_H1 = 0x0400
    #   x5  = DRAM OUT  = 0x0B00
    #   x6  = 1024 (transfer size per half)
    instructions: List[Instruction[Any]] = [
        # VMEM addresses
        Instruction(
            "lui", ScalarArgs(rd=1, imm=0x2)
        ),  # x1 = 0x2000 (VMEM X pair, 2048 B)
        Instruction("lui", ScalarArgs(rd=2, imm=0x3)),  # x2 = 0x3000 (VMEM OUT base)
        # DRAM addresses
        Instruction("addi", ScalarArgs(rd=3, rs1=0, imm=DRAM_X_H0)),  # x3 = 0x0000
        Instruction("addi", ScalarArgs(rd=4, rs1=3, imm=1024)),  # x4 = 0x0400
        Instruction("lui", ScalarArgs(rd=5, imm=0x1)),  # x5 = 0x1000? we want 0x0B00
        Instruction(
            "addi", ScalarArgs(rd=5, rs1=5, imm=-1280)
        ),  # x5 = 0x1000 - 0x500 = 0x0B00
        Instruction("addi", ScalarArgs(rd=6, rs1=0, imm=1024)),  # x6 = 1024
        # DMA loads: X_H0 → VMEM[x1]; X_H1 → VMEM[x1 + 1024]
        Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=0)),
        Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=1)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=3, rs2=6, channel=0)),
        # Second DMA lands at x1 + 1024. Build that address in x7.
        Instruction("addi", ScalarArgs(rd=7, rs1=1, imm=1024)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=7, rs1=4, rs2=6, channel=1)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        # Load X pair into (m0, m1)
        Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=32)),
        Instruction("delay", ScalarArgs(imm=34)),
        # (m4, m5) = row-sum broadcast over (m0, m1)
        Instruction("vredsum.row.bf16", VectorArgs(vd=4, vs1=0)),
        Instruction("delay", ScalarArgs(imm=69)),
        # Store m4 (first half) to VMEM[x2], then DMA to DRAM_OUT.
        Instruction("vstore", VectorArgs(vd=4, rs1=2, imm12=0)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("dma.store.ch<N>", DmaArgs(rd=5, rs1=2, rs2=6, channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_X_H0, INPUT[:, :16].contiguous()),
        (DRAM_X_H1, INPUT[:, 16:].contiguous()),
    ]

    golden_result: tuple[int, torch.Tensor] = (DRAM_OUT, EXPECTED)
