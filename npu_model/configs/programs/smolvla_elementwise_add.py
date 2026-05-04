"""SmolVLA elementwise addition kernel.

Adds two 32x32 bf16 tiles elementwise. Appears 539 times across SmolVLA
(residual adds, bias adds, etc.). 16 shape variants in the full model;
this Program implements the 32x32 canonical form. Other shapes use the
same pattern with different transfer sizes.

Everything lives in this one file:
    - The MLIR op definition (as a string, compilable with iree.compiler)
    - The PyTorch reference implementation
    - The NPU ISA program
    - The golden result (computed from PyTorch, cross-checked with MLIR if IREE available)

Model context:
    Elementwise add is the most common op in SmolVLA (residuals in every
    transformer block + bias adds + positional adds).

MLIR → ISA mapping:
    arith.addf %a %b → vadd.bf16(a, b)   pair-op: (vd, vd+1) = (vs1, vs1+1) + (vs2, vs2+1)
"""

import torch
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER

# ═══════════════════════════════════════════════════════════════════════════
# 1. MLIR definition — the exact op from SmolVLA's global-optimization IR.
# ═══════════════════════════════════════════════════════════════════════════

ELEMENTWISE_ADD_MLIR = """\
func.func @elementwise_add(
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
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %sum = arith.addf %lhs, %rhs : f32
    linalg.yield %sum : f32
  } -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}
"""


# ═══════════════════════════════════════════════════════════════════════════
# 2. PyTorch reference — computes the golden output.
# ═══════════════════════════════════════════════════════════════════════════


def elementwise_add_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise add: y[i, j] = a[i, j] + b[i, j]. Cast back to input dtype."""
    return (a.float() + b.float()).to(a.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Golden data — deterministic inputs + expected output.
# ═══════════════════════════════════════════════════════════════════════════

torch.manual_seed(42)
INPUT_A = torch.randn(32, 32, dtype=torch.bfloat16)
INPUT_B = torch.randn(32, 32, dtype=torch.bfloat16)

EXPECTED = elementwise_add_reference(INPUT_A, INPUT_B)

# Optional cross-check via IREE (mirrors smolvla_silu.py).
#
# Gated behind NPU_MODEL_ENABLE_IREE_CROSSCHECK because iree-runtime's
# nanobind bindings leak MappedMemory / HalBufferView instances at
# interpreter shutdown (the C extension's type objects finalize before
# our module globals release their refs). CI leaves the env var unset
# so the test process exits without triggering nanobind's leak report.
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
            ELEMENTWISE_ADD_MLIR,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
        _config = runtime.Config("local-task")
        _ctx = runtime.SystemContext(config=_config)
        _ctx.add_vm_module(runtime.VmModule.copy_buffer(_ctx.instance, _vmfb))
        _iree_out = _ctx.modules.module["elementwise_add"](
            INPUT_A.float().numpy(), INPUT_B.float().numpy()
        )
        _iree_expected = torch.from_numpy(np.array(_iree_out)).to(torch.bfloat16)
        _diff = (EXPECTED.float() - _iree_expected.float()).abs().max().item()
        assert _diff < 1e-3, f"MLIR vs PyTorch mismatch: {_diff}"
        EXPECTED = _iree_expected
    except ImportError:
        pass  # IREE not available — use PyTorch reference (fine for CI)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Memory layout.
#
# The kernel treats a 32x32 bf16 tile as two [32, 16] halves packed
# consecutively in memory (bytes 0..1023 = cols 0-15, bytes 1024..2047 =
# cols 16-31). For a plain torch.randn(32, 32).contiguous(), this
# interpretation maps row-major bytes to the two halves as "first 16
# rows then next 16 rows" — the elementwise add is layout-invariant, so
# reading the output as [32, 32] row-major recovers the arithmetic sum.
# ═══════════════════════════════════════════════════════════════════════════

DRAM_A_BASE = 0x80000000
DRAM_B_BASE = 0x80001000
DRAM_OUTPUT_BASE = 0x80002000
VMEM_A_BASE = 0x20004000
VMEM_B_BASE = 0x20005000
VMEM_OUTPUT_BASE = 0x20006000
TILE_BYTES = 2048  # 32 * 32 * 2 (bf16)


# ═══════════════════════════════════════════════════════════════════════════
# 5. NPU ISA program.
#
# Two vloads per operand (imm12=0 / imm12=32 picks the first / second
# 32x16 half), two vadds, two vstores, one DMA store.
# ═══════════════════════════════════════════════════════════════════════════


class SmolVLAElementwiseAddProgram(Program):
    """y = a + b on two 32x32 bf16 tiles."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'smolvla_elementwise_add.S') 

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_A_BASE, INPUT_A),
        (DRAM_B_BASE, INPUT_B),
    ]

    golden_result: list[tuple[int, torch.Tensor]] = [(
        DRAM_OUTPUT_BASE,
        EXPECTED,
    )]
