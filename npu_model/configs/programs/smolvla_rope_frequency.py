"""SmolVLA rope_frequency kernel.

Per-element cosine on a 32x32 bf16 tile. Used as a building
block for RoPE (rotary position embeddings); pair with a
sin kernel to form the full rotary transform.
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs


# ═══════════════════════════════════════════════════════════════════════════
# 2. PyTorch reference.
# ═══════════════════════════════════════════════════════════════════════════


def rope_frequency_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.cos(x.float()).to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Golden data.
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────
# MLIR: elementwise cos. The benchmark MLIRs under
# ``benchmarks/…/rope_frequency/`` stage the x^2 precompute; our kernel
# runs the trailing cos op that consumes RoPE frequencies.
# ───────────────────────────────────────────────────────────────────────

ROPE_FREQUENCY_MLIR = """\
func.func @rope_frequency(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %out0 = tensor.empty() : tensor<32x32xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%x : tensor<32x32xf32>) outs(%out0 : tensor<32x32xf32>) {
  ^bb0(%in: f32, %_: f32):
    %c = math.cos %in : f32
    linalg.yield %c : f32
  } -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}
"""

torch.manual_seed(47)
# Keep values in a range where bf16 vcos is well-behaved.
INPUT = torch.randn(32, 32, dtype=torch.bfloat16) * 0.5
EXPECTED = rope_frequency_reference(INPUT)

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
            ROPE_FREQUENCY_MLIR,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
        _cfg = runtime.Config("local-task")
        _ctx = runtime.SystemContext(config=_cfg)
        _ctx.add_vm_module(runtime.VmModule.copy_buffer(_ctx.instance, _vmfb))
        _iree_out = _ctx.modules.module["rope_frequency"](INPUT.float().numpy())
        _iree_bf16 = torch.from_numpy(np.array(_iree_out)).to(torch.bfloat16)
        _diff = (EXPECTED.float() - _iree_bf16.float()).abs().max().item()
        assert _diff < 5e-2, f"MLIR vs PyTorch mismatch: {_diff}"
    except ImportError:
        pass

DRAM_X = 0x0000
DRAM_OUT = 0x0B00


# ═══════════════════════════════════════════════════════════════════════════
# 4. NPU ISA program.
# ═══════════════════════════════════════════════════════════════════════════


class SmolVLARopeFrequencyProgram(Program):
    """Per-element cosine on a 32x32 bf16 tile (RoPE frequency precompute). cycles: ~202"""

    instructions: List[Instruction[Any]] = [
        Instruction("lui", ScalarArgs(rd=1, imm=2)),
        Instruction("addi", ScalarArgs(rd=1, rs1=1)),
        Instruction("addi", ScalarArgs(rd=4)),
        Instruction("lui", ScalarArgs(rd=5, imm=1)),
        Instruction("addi", ScalarArgs(rd=5, rs1=5, imm=-1280)),
        Instruction("lui", ScalarArgs(rd=6, imm=1)),
        Instruction("addi", ScalarArgs(rd=6, rs1=6, imm=-2048)),
        Instruction("dma.config.ch<N>", DmaArgs()),
        Instruction("dma.wait.ch<N>", DmaArgs()),
        Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=4, rs2=6)),
        Instruction("dma.wait.ch<N>", DmaArgs()),
        Instruction("vload", VectorArgs(rs1=1)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=32)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vcos.bf16", VectorArgs(vd=2)),  # (v2, v3) = cos(v0, v1)
        Instruction("delay", ScalarArgs(imm=66)),
        Instruction("vstore", VectorArgs(vd=2, rs1=1)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vstore", VectorArgs(vd=3, rs1=1, imm12=32)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("dma.store.ch<N>", DmaArgs(rd=5, rs1=1, rs2=6)),
        Instruction("dma.wait.ch<N>", DmaArgs(rs2=6)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_X, INPUT),
    ]

    golden_result: tuple[int, torch.Tensor] = (DRAM_OUT, EXPECTED)
