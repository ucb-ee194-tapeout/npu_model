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

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs


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
DRAM_A_K0 = 0x0000
DRAM_A_K1 = 0x0400
DRAM_B_K0 = 0x0800
DRAM_B_K1 = 0x0C00
DRAM_OUT = 0x1000

# VMEM mirrors DRAM layout (distinct address space).
VMEM_A_K0 = 0x0000
VMEM_A_K1 = 0x0400
VMEM_B_K0 = 0x0800
VMEM_B_K1 = 0x0C00
VMEM_OUT_H0 = 0x1000
VMEM_OUT_H1 = 0x1400  # second 32x16 half of the bf16 output tile


# ═══════════════════════════════════════════════════════════════════════════
# 4. NPU ISA program.
# ═══════════════════════════════════════════════════════════════════════════


class SmolVLAMatmulKChainProgram(Program):
    """Two-tile K-chain matmul (K=64 split into 2×32). Uses MXU0 with acc.

    Chains vmatmul.mxu0 + vmatmul.acc.mxu0 for the two K-tiles.

    DMA pipelining: A_K1 and B_K1 are issued right after K0 data is ready.
    The K0 compute (~196cy decoder, 96cy MXU0) overlaps with the DMA engine
    loading A_K1 (~516cy), giving MXU0+DMA concurrent execution.
    """

    instructions: List[Instruction[Any]] = [
        # Scalar register setup
        Instruction("addi", ScalarArgs(rd=1, imm=DRAM_A_K0)),
        Instruction("addi", ScalarArgs(rd=2, rs1=1, imm=1024)),   # DRAM_A_K1
        Instruction("addi", ScalarArgs(rd=3, rs1=2, imm=1024)),   # DRAM_B_K0
        Instruction("addi", ScalarArgs(rd=4, rs1=3, imm=1024)),   # DRAM_B_K1
        Instruction("addi", ScalarArgs(rd=5, rs1=4, imm=1024)),   # DRAM_OUT
        Instruction("addi", ScalarArgs(rd=6, imm=VMEM_A_K0)),
        Instruction("addi", ScalarArgs(rd=7, rs1=6, imm=1024)),   # VMEM_A_K1
        Instruction("addi", ScalarArgs(rd=8, rs1=7, imm=1024)),   # VMEM_B_K0
        Instruction("addi", ScalarArgs(rd=9, rs1=8, imm=1024)),   # VMEM_B_K1
        Instruction("addi", ScalarArgs(rd=10, rs1=9, imm=1024)),  # VMEM_OUT_H0
        Instruction("addi", ScalarArgs(rd=11, rs1=10, imm=1024)), # VMEM_OUT_H1
        Instruction("addi", ScalarArgs(rd=12, imm=1024)),          # tile size
        Instruction("addi", ScalarArgs(rd=13, rs1=12, imm=1024)), # output size (2048)
        Instruction("dma.config.ch<N>", DmaArgs()),
        Instruction("dma.wait.ch<N>", DmaArgs()),
        # Load K0 tiles: A_K0 (ch0, 516cy) → B_K0 (ch1, 516cy, sequential).
        Instruction("dma.load.ch<N>", DmaArgs(rd=6, rs1=1, rs2=12)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=8, rs1=3, rs2=12, channel=1)),
        Instruction("dma.wait.ch<N>", DmaArgs()),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        # K0 data ready. Issue K1 loads async now so A_K1 starts immediately
        # behind B_K0 in the DMA engine. K0 compute (~196cy) overlaps with
        # A_K1 loading (~516cy), giving MXU0+DMA concurrent execution.
        Instruction("dma.load.ch<N>", DmaArgs(rd=7, rs1=2, rs2=12)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=9, rs1=4, rs2=12, channel=1)),
        # K tile 0: acc = A_K0 @ B_K0 (reset accumulator)
        Instruction("vload", VectorArgs(vd=0, rs1=6)),          # mrf[0] ← A_K0
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vload", VectorArgs(vd=1, rs1=8)),          # mrf[1] ← B_K0
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vmatpush.weight.mxu0", VectorArgs(vs1=1)), # wb[0] ← mrf[1]
        Instruction("delay", ScalarArgs(imm=32)),
        Instruction("vmatmul.mxu0", MatrixArgs()),              # acc[0] = mrf[0] @ wb[0], 96cy
        Instruction("delay", ScalarArgs(imm=96)),
        # K1 tiles arrive during K0 compute. Wait now — should be near-instant
        # for A_K1 (started ~1032cy ago, takes 516cy) and a short stall for B_K1.
        Instruction("dma.wait.ch<N>", DmaArgs()),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        # K tile 1: acc += A_K1 @ B_K1
        Instruction("vload", VectorArgs(vd=2, rs1=7)),          # mrf[2] ← A_K1
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vload", VectorArgs(vd=3, rs1=9)),          # mrf[3] ← B_K1
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vmatpush.weight.mxu0", VectorArgs(vs1=3)), # wb[0] ← mrf[3]
        Instruction("delay", ScalarArgs(imm=32)),
        Instruction("vmatmul.acc.mxu0", MatrixArgs(vs1=2)),     # acc[0] += mrf[2] @ wb[0]
        Instruction("delay", ScalarArgs(imm=96)),
        # Pop accumulator and store both halves before DMA (dma.store covers
        # the full 2048B region so both vstores must complete first).
        Instruction("vmatpop.bf16.acc.mxu0", VectorArgs(vd=4)), # mrf[4..5] ← acc[0]
        Instruction("delay", ScalarArgs(imm=32)),
        Instruction("vstore", VectorArgs(vd=4, rs1=10)),        # vmem[OUT_H0] ← mrf[4]
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vstore", VectorArgs(vd=5, rs1=11)),        # vmem[OUT_H1] ← mrf[5]
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("dma.store.ch<N>", DmaArgs(rd=5, rs1=10, rs2=13, channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_A_K0, INPUT_A[:, :32].contiguous()),
        (DRAM_A_K1, INPUT_A[:, 32:].contiguous()),
        (DRAM_B_K0, INPUT_B[:32, :].contiguous()),
        (DRAM_B_K1, INPUT_B[32:, :].contiguous()),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUT,
        EXPECTED_STACKED,
    )
