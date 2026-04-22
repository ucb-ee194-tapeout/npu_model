"""SmolVLA matmul kernel.

One-tile fp8 matmul: ``C = A @ B`` where A and B are 32x32 fp8
tiles. The kernel pops the bf16 accumulator as two 32x16 halves
(cols 0-15 and cols 16-31) written at DRAM_OUTPUT and
DRAM_OUTPUT + 0x400.
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs


# ═══════════════════════════════════════════════════════════════════════════
# 1. MLIR definition. Modeled after
# ``benchmarks/SaturnNPU/kernels/quantized_matmul_fp8/variant_*`` — the
# benchmark uses fp8 inputs lifted to bf16; stock llvm-cpu cannot lower
# fp8 matmul, so we promote inputs/outputs to f32 for IREE. Hardware
# still computes in fp8; the loose cross-check tolerance accommodates.
# ═══════════════════════════════════════════════════════════════════════════

MATMUL_MLIR = """\
func.func @matmul(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>)
    -> tensor<32x32xf32> {
  %empty = tensor.empty() : tensor<32x32xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32)
                      outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  %result = linalg.matmul
      ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>)
      outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}
"""


# ═══════════════════════════════════════════════════════════════════════════
# 2. PyTorch reference.
# ═══════════════════════════════════════════════════════════════════════════


def matmul_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.to(torch.float32) @ b.to(torch.float32)).to(torch.bfloat16)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Golden data.
# ═══════════════════════════════════════════════════════════════════════════

torch.manual_seed(42)
INPUT_A = torch.randint(-8, 8, (32, 32), dtype=torch.int8).to(torch.float8_e4m3fn)
INPUT_B = torch.randint(-8, 8, (32, 32), dtype=torch.int8).to(torch.float8_e4m3fn)
EXPECTED = matmul_reference(INPUT_A, INPUT_B)
# Kernel writes output as two 32x16 halves stacked: cols 0-15, then cols 16-31.
EXPECTED_STACKED = torch.cat((EXPECTED[:, :16], EXPECTED[:, 16:]), dim=0)

# Cross-check via IREE. Hardware uses fp8; MLIR is f32 for IREE compatibility.
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
            MATMUL_MLIR,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
        _cfg = runtime.Config("local-task")
        _ctx = runtime.SystemContext(config=_cfg)
        _ctx.add_vm_module(runtime.VmModule.copy_buffer(_ctx.instance, _vmfb))
        _iree_out = _ctx.modules.module["matmul"](
            INPUT_A.to(torch.float32).numpy(), INPUT_B.to(torch.float32).numpy()
        )
        _iree_bf16 = torch.from_numpy(np.array(_iree_out)).to(torch.bfloat16)
        _diff = (EXPECTED.float() - _iree_bf16.float()).abs().max().item()
        assert _diff < 4.0, f"MLIR vs PyTorch mismatch: {_diff}"
    except ImportError:
        pass

DRAM_A = 0x0
DRAM_B = 0x500
DRAM_OUT = 0xB00


# ═══════════════════════════════════════════════════════════════════════════
# 4. NPU ISA program.
# ═══════════════════════════════════════════════════════════════════════════


class SmolVLAMatmulProgram(Program):
    """Auto-generated single-file Program for the ``matmul`` kernel.

    ISA is lifted from the merlin kernel manifest (see
    ``benchmarks/SaturnNPU/kernel_library/manifest.json``). This Program
    mirrors the ``smolvla_silu.py`` template: self-contained, no cross-
    file helpers, torch-allclose golden check via ``pytest tests/test_programs.py``.
    """

    instructions: List[Instruction[Any]] = [
        Instruction("addi", ScalarArgs(rd=1)),
        Instruction("addi", ScalarArgs(rd=2, imm=1280)),
        Instruction("lui", ScalarArgs(rd=3, imm=1)),
        Instruction("addi", ScalarArgs(rd=3, rs1=3, imm=-1280)),
        Instruction("lui", ScalarArgs(rd=4, imm=2)),
        Instruction("addi", ScalarArgs(rd=4, rs1=4)),
        Instruction("addi", ScalarArgs(rd=5, rs1=4, imm=1024)),
        Instruction("addi", ScalarArgs(rd=6, rs1=5, imm=1024)),
        Instruction("addi", ScalarArgs(rd=7, imm=1024)),
        Instruction("lui", ScalarArgs(rd=8, imm=1)),
        Instruction("addi", ScalarArgs(rd=8, rs1=8, imm=-2048)),
        Instruction("dma.config.ch<N>", DmaArgs(channel=0)),
        Instruction(
            "dma.wait.ch<N>", DmaArgs(channel=0)
        ),  # let config settle before issuing loads
        Instruction("dma.load.ch<N>", DmaArgs(rd=4, rs1=1, rs2=7, channel=0)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=5, rs1=2, rs2=7, channel=1)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        Instruction("vload", VectorArgs(rs1=4)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vload", VectorArgs(vd=1, rs1=5)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vmatpush.weight.mxu0", VectorArgs(vd=0,vs1=1)),
        Instruction("delay", ScalarArgs(imm=32)),
        Instruction("vmatmul.mxu0", MatrixArgs(vd=0, vs1=0, vs2=0)),
        Instruction("delay", ScalarArgs(imm=96)),
        Instruction("vmatpop.bf16.acc.mxu0", VectorArgs(vd=2)),
        Instruction("delay", ScalarArgs(imm=32)),
        Instruction("vstore", VectorArgs(vd=2, rs1=6)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("addi", ScalarArgs(rd=9, rs1=6, imm=1024)),
        Instruction("vstore", VectorArgs(vd=3, rs1=9)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("dma.store.ch<N>", DmaArgs(rd=3, rs1=6, rs2=8,channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("ecall", ScalarArgs()),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_A, INPUT_A),
        (DRAM_B, INPUT_B),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUT,
        EXPECTED_STACKED,
    )
