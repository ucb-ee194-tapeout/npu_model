"""SmolVLA rope_frequency kernel.

Per-element cosine on a 32x32 bf16 tile. Used as a building
block for RoPE (rotary position embeddings); pair with a
sin kernel to form the full rotary transform.
"""

import torch
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER

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
    """Auto-generated single-file Program for the ``rope_frequency`` kernel.

    ISA is lifted from the merlin kernel manifest (see
    ``benchmarks/SaturnNPU/kernel_library/manifest.json``). This Program
    mirrors the ``smolvla_silu.py`` template: self-contained, no cross-
    file helpers, torch-allclose golden check via ``pytest tests/test_programs.py``.
    """

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'smolvla_rope_frequency.S')

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_X, INPUT),
    ]

    golden_result: list[tuple[int, torch.Tensor]] = [(DRAM_OUT, EXPECTED)]
