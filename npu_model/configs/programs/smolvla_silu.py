"""SmolVLA SiLU/Swish activation kernel.

This program demonstrates how a SmolVLA MLIR op maps to NPU ISA instructions.
Use it as a template for implementing other SmolVLA kernels.

Everything lives in this one file:
    - The MLIR op definition (as a string, compilable with iree.compiler)
    - The PyTorch reference implementation
    - The NPU ISA program
    - The golden result (computed from PyTorch, optionally cross-checked with
      MLIR when NPU_MODEL_ENABLE_IREE_CROSSCHECK=1)

Model context:
    SiLU appears 32 times in SmolVLA (once per Gemma MLP layer).

MLIR → ISA mapping:
    arith.negf %x       → vmul.bf16(x, -1)         negate via multiply
    math.exp %neg        → vexp.bf16(neg_x)          vector exp
    arith.addf %one %exp → vadd.bf16(exp, ones)      add constant 1.0
    arith.divf %x %denom → vrecip.bf16 + vmul.bf16   reciprocal then multiply

How to add your own SmolVLA kernel:
    1. Copy this file.
    2. Find your MLIR in merlin/benchmarks/SaturnNPU/kernels/<type>/.
    3. Replace SILU_MLIR, silu_reference, and the ISA instructions.
    4. Run: uv run pytest tests/test_programs.py --sim-verbose -vv
"""

import os
import torch
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER


# ═══════════════════════════════════════════════════════════════════════════
# 1. MLIR definition — the exact op from SmolVLA's global-optimization IR.
#    This can be compiled standalone with: iree.compiler.compile_str(SILU_MLIR)
# ═══════════════════════════════════════════════════════════════════════════

SILU_MLIR = """\
#hal.executable.target<cpu="host">
func.func @silu(%arg0: tensor<32x16xf32>) -> tensor<32x16xf32> {
  %empty = tensor.empty() : tensor<32x16xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<32x32xf32>) outs(%empty : tensor<32x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %neg = arith.negf %in : f32
    %exp = math.exp %neg : f32
    %one = arith.constant 1.0 : f32
    %denom = arith.addf %one, %exp : f32
    %res = arith.divf %in, %denom : f32
    linalg.yield %res : f32
  } -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}
"""


# ═══════════════════════════════════════════════════════════════════════════
# 2. PyTorch reference — computes the golden output.
# ═══════════════════════════════════════════════════════════════════════════


def silu_reference(x: torch.Tensor) -> torch.Tensor:
    """SiLU(x) = x * sigmoid(x). Matches the MLIR linalg.generic above."""
    return (x.float() * torch.sigmoid(x.float())).to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Golden data — deterministic input + expected output.
# ═══════════════════════════════════════════════════════════════════════════


def _iree_crosscheck_enabled() -> bool:
    value = os.environ.get("NPU_MODEL_ENABLE_IREE_CROSSCHECK", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _maybe_crosscheck_with_iree(expected: torch.Tensor) -> torch.Tensor:
    """Optionally validate the MLIR with IREE without impacting normal startup."""
    if not _iree_crosscheck_enabled():
        return expected

    try:
        import numpy as np
        import iree.compiler as compiler
        import iree.runtime as runtime
    except ImportError:
        return expected

    vmfb = compiler.compile_str(
        SILU_MLIR,
        target_backends=["llvm-cpu"],
        extra_args=["--iree-llvmcpu-target-cpu=generic"],
    )
    config = runtime.Config("local-task")
    ctx = runtime.SystemContext(config=config)
    ctx.add_vm_module(runtime.VmModule.copy_buffer(ctx.instance, vmfb))
    iree_out = ctx.modules.module["silu"](INPUT.float().numpy())
    iree_expected = torch.from_numpy(np.array(iree_out)).to(torch.bfloat16)
    diff = (expected.float() - iree_expected.float()).abs().max().item()
    assert diff < 1e-3, f"MLIR vs PyTorch mismatch: {diff}"
    return iree_expected


torch.manual_seed(42)
INPUT = torch.randn(32, 32, dtype=torch.bfloat16)

# Primary golden: PyTorch reference
EXPECTED = silu_reference(INPUT)
EXPECTED = _maybe_crosscheck_with_iree(EXPECTED)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Memory layout
# ═══════════════════════════════════════════════════════════════════════════

DRAM_INPUT_BASE = 0x80000000
DRAM_OUTPUT_BASE = 0x80000800
VMEM_INPUT_BASE = 0x20002000
VMEM_OUTPUT_BASE = 0x20002800
TILE_BYTES = 2048  # 32 * 32 * 2 (bf16)


# ═══════════════════════════════════════════════════════════════════════════
# 5. NPU ISA program — the kernel implementation under test.
# ═══════════════════════════════════════════════════════════════════════════


class SmolVLASiluProgram(Program):
    """SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'smolvla_silu.S')

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_INPUT_BASE, INPUT),
    ]

    golden_result: list[tuple[int, torch.Tensor]] = [(
        DRAM_OUTPUT_BASE,
        EXPECTED,
    )]
