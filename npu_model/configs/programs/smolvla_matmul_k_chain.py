"""SmolVLA matmul K-tile chain kernel.

Composed matmul that accumulates across the reduction dimension (K).
The per-tile ``smolvla_matmul`` kernel computes a single
``[32x32] @ [32x32]`` MXU product.  Real layers have K >> 32, so the
compiler tiles K into 32-wide slices and chains the per-tile kernel
with ``vmatmul.mxu0`` (first tile, resets accumulator) followed by
``vmatmul.acc.mxu0`` (subsequent tiles, accumulate).

This Program tiles K=64 into **two** 32-wide slices to exercise the
accumulator state machine end-to-end:

    C[32, 32] = A[32, 0:32]  @ B[0:32, 32]      (reset acc)
              + A[32, 32:64] @ B[32:64, 32]     (accumulate)

It is the npu_model-side equivalent of merlin's
``matmul_acc_first`` + ``matmul_acc_last`` chain: self-contained, no
cross-file helpers, seeded golden tensor, torch-allclose validation
via ``pytest tests/test_programs.py``.

Per-tile arithmetic mirrors the MXU semantics exactly:
    acc_fp16_1   = A_k0 @ B_k0   (fp16 × fp16)
    acc_bf16_1   = acc_fp16_1.to(bf16)
    acc_fp16_2   = A_k1 @ B_k1  + acc_bf16_1.to(fp16)
    out_bf16     = acc_fp16_2.to(bf16)

So the torch reference below rounds to bf16 at each accumulator store
to capture the bf16 round-trip the hardware performs between tiles.
"""

import torch
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER

# ═══════════════════════════════════════════════════════════════════════════
# 1. MLIR definition.
# ═══════════════════════════════════════════════════════════════════════════

MATMUL_K_CHAIN_MLIR = """\
func.func @matmul_k_chain(
    %a: tensor<32x64xf32>, %b: tensor<64x32xf32>
) -> tensor<32x32xf32> {
  %empty = tensor.empty() : tensor<32x32xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32)
                      outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  %result = linalg.matmul
      ins(%a, %b : tensor<32x64xf32>, tensor<64x32xf32>)
      outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}
"""
# Note: MLIR uses f32 for IREE compatibility. Hardware computes in fp8
# with bf16 accumulator; PyTorch reference mirrors the hardware. The
# IREE cross-check below tolerates a loose tol (~2.0) to cover the
# fp8-quantization gap between the f32 MLIR and bf16 hardware ref.


# ═══════════════════════════════════════════════════════════════════════════
# 2. PyTorch reference.
# ═══════════════════════════════════════════════════════════════════════════


def matmul_k_chain_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Mirror MXU per-tile semantics: fp8 → fp16 multiply, bf16 accumulate.
    a0, a1 = a[:, :32].to(torch.float16), a[:, 32:].to(torch.float16)
    b0, b1 = b[:32, :].to(torch.float16), b[32:, :].to(torch.float16)
    acc_bf16 = (a0 @ b0).to(torch.bfloat16)  # first tile writes acc
    acc_fp16 = (a1 @ b1) + acc_bf16.to(torch.float16)  # accumulate second tile
    return acc_fp16.to(torch.bfloat16)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Golden data.
# ═══════════════════════════════════════════════════════════════════════════

torch.manual_seed(44)
INPUT_A = torch.randint(-8, 8, (32, 64), dtype=torch.int8).to(torch.float8_e4m3fn)
INPUT_B = torch.randint(-8, 8, (64, 32), dtype=torch.int8).to(torch.float8_e4m3fn)
EXPECTED = matmul_k_chain_reference(INPUT_A, INPUT_B)
# Kernel writes the 32x32 bf16 output as two 32x16 halves stacked: cols
# 0-15 first, then cols 16-31. Golden is stacked to match byte layout.
EXPECTED_STACKED = torch.cat((EXPECTED[:, :16], EXPECTED[:, 16:]), dim=0)

# Cross-check: compile the (f32) MLIR via IREE and compare to the
# hardware-semantic PyTorch reference. Expect a gap (fp8 quantization
# in the hardware ref vs exact f32 in MLIR) on order of magnitudes.
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
            MATMUL_K_CHAIN_MLIR,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
        _cfg = runtime.Config("local-task")
        _ctx = runtime.SystemContext(config=_cfg)
        _ctx.add_vm_module(runtime.VmModule.copy_buffer(_ctx.instance, _vmfb))
        _iree_out = _ctx.modules.module["matmul_k_chain"](
            INPUT_A.to(torch.float32).numpy(), INPUT_B.to(torch.float32).numpy()
        )
        _iree_bf16 = torch.from_numpy(np.array(_iree_out)).to(torch.bfloat16)
        # Compare f32-MLIR vs fp8-accumulator-bf16 hardware reference on
        # bf16 values. Tolerance: a few bf16 ULPs per accumulated term.
        _diff = (EXPECTED.float() - _iree_bf16.float()).abs().max().item()
        assert _diff < 8.0, f"MLIR vs PyTorch mismatch: {_diff}"
    except ImportError:
        pass

# DRAM layout — all per-tile fp8 buffers are 32x32 = 1024 B; output is bf16 = 2048 B.
DRAM_A_K0 = 0x80000000
DRAM_A_K1 = 0x80000400
DRAM_B_K0 = 0x80000800
DRAM_B_K1 = 0x80000C00
DRAM_OUT = 0x80001000

# VMEM mirrors DRAM layout (distinct address space).
VMEM_A_K0 = 0x20000000
VMEM_A_K1 = 0x20000400
VMEM_B_K0 = 0x20000800
VMEM_B_K1 = 0x20000C00
VMEM_OUT_H0 = 0x20001000
VMEM_OUT_H1 = 0x20001400  # second 32x16 half of the bf16 output tile


# ═══════════════════════════════════════════════════════════════════════════
# 4. NPU ISA program.
# ═══════════════════════════════════════════════════════════════════════════


class SmolVLAMatmulKChainProgram(Program):
    """Two-tile K-chain matmul composed from the per-tile matmul kernel.

    Uses dual DMA channels to parallelize A-tile and B-tile loads.
    Chains ``vmatmul.mxu0`` (first K-tile) with ``vmatmul.acc.mxu0``
    (second K-tile); both target the same accumulator slot so the
    final ``vmatpop`` yields A_k0@B_k0 + A_k1@B_k1.
    """

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'smolvla_matmul_k_chain.S')

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_A_K0, INPUT_A[:, :32].contiguous()),
        (DRAM_A_K1, INPUT_A[:, 32:].contiguous()),
        (DRAM_B_K0, INPUT_B[:32, :].contiguous()),
        (DRAM_B_K1, INPUT_B[32:, :].contiguous()),
    ]

    golden_result: list[tuple[int, torch.Tensor]] = [(
        DRAM_OUT,
        EXPECTED_STACKED,
    )]
