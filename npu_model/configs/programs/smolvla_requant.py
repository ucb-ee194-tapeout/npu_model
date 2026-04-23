"""SmolVLA requant kernel.

bf16 → fp8 requantization with unit scale (seli imm=1).
Reads two 32x16 bf16 halves and packs to a contiguous 32x32
fp8 tile via vpack.bf16.fp8.
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs


# ═══════════════════════════════════════════════════════════════════════════
# 2. PyTorch reference.
# ═══════════════════════════════════════════════════════════════════════════


def requant_reference(x: torch.Tensor) -> torch.Tensor:
    # Naive cast — kernel's seli=1 is the unit-scale path.
    return x.to(torch.float8_e4m3fn)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Golden data.
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────
# MLIR: bf16 → fp8 → bf16 round-trip (matches kernel's seli=1 unit-scale
# cast). Output dtype is bf16 so IREE doesn't need fp8 buffer support
# at the runtime boundary; the fp8 round happens via arith.truncf inside
# the linalg body.
# ───────────────────────────────────────────────────────────────────────

REQUANT_MLIR = """\
func.func @requant(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %out0 = tensor.empty() : tensor<32x32xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%x : tensor<32x32xf32>) outs(%out0 : tensor<32x32xf32>) {
  ^bb0(%in: f32, %_: f32):
    %t = arith.truncf %in : f32 to f8E4M3FN
    %u = arith.extf %t : f8E4M3FN to f32
    linalg.yield %u : f32
  } -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}
"""

torch.manual_seed(48)
# Keep values in the fp8_e4m3 representable range.
INPUT = torch.randn(32, 32, dtype=torch.bfloat16) * 0.5
EXPECTED = requant_reference(INPUT)

# Cross-check via IREE. PyTorch reference returns fp8; IREE returns
# fp8-rounded f32; cast both to f32 for comparison.
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
            REQUANT_MLIR,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
        _cfg = runtime.Config("local-task")
        _ctx = runtime.SystemContext(config=_cfg)
        _ctx.add_vm_module(runtime.VmModule.copy_buffer(_ctx.instance, _vmfb))
        _iree_out = _ctx.modules.module["requant"](INPUT.float().numpy())
        _iree_arr = np.array(_iree_out)
        _iree_f32 = torch.from_numpy(_iree_arr)
        _diff = (EXPECTED.to(torch.float32) - _iree_f32).abs().max().item()
        # Bf16-rounded input vs f32-then-fp8 path can differ by 1 bf16 ULP.
        assert _diff < 5e-2, f"MLIR vs PyTorch mismatch: {_diff}"
    except ImportError:
        pass

DRAM_X_H0 = 0x0000
DRAM_X_H1 = 0x0400
DRAM_OUT = 0x0B00


# ═══════════════════════════════════════════════════════════════════════════
# 4. NPU ISA program.
# ═══════════════════════════════════════════════════════════════════════════


class SmolVLARequantProgram(Program):
    """Auto-generated single-file Program for the ``requant`` kernel.

    ISA is lifted from the merlin kernel manifest (see
    ``benchmarks/SaturnNPU/kernel_library/manifest.json``). This Program
    mirrors the ``smolvla_silu.py`` template: self-contained, no cross-
    file helpers, torch-allclose golden check via ``pytest tests/test_programs.py``.
    """

    instructions: List[Instruction[Any]] = [
        Instruction("lui", ScalarArgs(rd=1, imm=2)),
        Instruction("addi", ScalarArgs(rd=2, rs1=1, imm=1024)),
        Instruction("lui", ScalarArgs(rd=3, imm=3)),
        Instruction("addi", ScalarArgs(rd=4)),
        Instruction("addi", ScalarArgs(rd=5, imm=1024)),
        Instruction("addi", ScalarArgs(rd=6, imm=2047)),
        Instruction("addi", ScalarArgs(rd=6, rs1=6, imm=769)),
        Instruction("addi", ScalarArgs(rd=7, imm=1024)),
        Instruction("seli", ScalarArgs(rd=5, imm=1)),
        Instruction("dma.config.ch<N>", DmaArgs()),
        Instruction("dma.config.ch<N>", DmaArgs(channel=1)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=4, rs2=7, channel=0)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=5, rs2=7, channel=1)),
        Instruction("dma.wait.ch<N>", DmaArgs()),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        Instruction("vload", VectorArgs(rs1=1)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vload", VectorArgs(vd=1, rs1=2)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vpack.bf16.fp8", VectorArgs(vd=2, es1=5)),
        Instruction("delay", ScalarArgs(imm=66)),
        Instruction("vstore", VectorArgs(vd=2, rs1=3)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("dma.store.ch<N>", DmaArgs(rd=6, rs1=3, rs2=7, channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs()),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_X_H0, INPUT[:, :16].contiguous()),
        (DRAM_X_H1, INPUT[:, 16:].contiguous()),
    ]

    golden_result: tuple[int, torch.Tensor] = (DRAM_OUT, EXPECTED)
