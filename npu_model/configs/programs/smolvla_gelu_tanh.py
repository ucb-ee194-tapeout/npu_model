"""SmolVLA gelu_tanh kernel.

GELU activation with the tanh approximation:
    y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
Appears 36 times in SmolVLA (SigLIP MLPs).

Pair-op BF16 layout (LMUL=2):
  (m0, m1)   = x                  (m2, m3)   = 0.044715 broadcast pair
  (m4, m5)   = sqrt(2/pi) pair    (m6, m7)   = 0.5 pair
  m8 = m9    = 1.0  (via vli.all)
  (m10, m11) = x^2     (m12, m13) = x^3
  (m14, m15) = 0.044715*x^3       (m16, m17) = x + 0.044715*x^3
  (m18, m19) = sqrt(2/pi)*(x+0.044715*x^3)
  (m20, m21) = tanh(...)          (m22, m23) = 1 + tanh(...)
  (m24, m25) = 0.5*x              (m26, m27) = GELU(x) = 0.5*x*(1+tanh(...))

DRAM layout (1024 B per 32x16 tile):
  DRAM_X_H0    = 0x0000  x half 0
  DRAM_X_H1    = 0x0400  x half 1
  DRAM_C044    = 0x0800  0.044715 broadcast tile (loaded into m2 AND m3)
  DRAM_CSQRT   = 0x0C00  sqrt(2/pi) broadcast tile (loaded into m4 AND m5)
  DRAM_CHALF   = 0x1000  0.5 broadcast tile (loaded into m6 AND m7)
  DRAM_OUT_H0  = 0x1400  output half 0
  DRAM_OUT_H1  = 0x1800  output half 1
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs


# ═══════════════════════════════════════════════════════════════════════════
# 2. PyTorch reference.
# ═══════════════════════════════════════════════════════════════════════════

import math


def gelu_tanh_reference(x: torch.Tensor) -> torch.Tensor:
    return (
        x.float()
        * 0.5
        * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x.float() + 0.044715 * x.float().pow(3.0))
            )
        )
    ).to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Golden data.
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────
# MLIR: GELU (tanh approximation). Transcribed from
# ``benchmarks/SaturnNPU/kernels/gelu_tanh/variant_0_1024_3072_bf16.mlir``
# with the shape reduced to a 32x32 tile and constants spelled out.
#   y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# ───────────────────────────────────────────────────────────────────────

GELU_TANH_MLIR = """\
func.func @gelu_tanh(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %c3    = arith.constant 3 : i64
  %k044  = arith.constant 0.044715 : f32
  %ksqrt = arith.constant 0.7978845 : f32
  %one   = arith.constant 1.0 : f32
  %half  = arith.constant 0.5 : f32
  %out0  = tensor.empty() : tensor<32x32xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%x : tensor<32x32xf32>) outs(%out0 : tensor<32x32xf32>) {
  ^bb0(%in: f32, %_: f32):
    %x3 = math.fpowi %in, %c3 : f32, i64
    %t1 = arith.mulf %x3, %k044 : f32
    %t2 = arith.addf %in, %t1 : f32
    %t3 = arith.mulf %t2, %ksqrt : f32
    %t4 = math.tanh %t3 : f32
    %t5 = arith.addf %t4, %one : f32
    %t6 = arith.mulf %t5, %half : f32
    %y  = arith.mulf %in, %t6 : f32
    linalg.yield %y : f32
  } -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}
"""

torch.manual_seed(42)
INPUT = torch.randn(32, 32, dtype=torch.bfloat16) * 0.5
EXPECTED = gelu_tanh_reference(INPUT)

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
            GELU_TANH_MLIR,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
        _cfg = runtime.Config("local-task")
        _ctx = runtime.SystemContext(config=_cfg)
        _ctx.add_vm_module(runtime.VmModule.copy_buffer(_ctx.instance, _vmfb))
        _iree_out = _ctx.modules.module["gelu_tanh"](INPUT.float().numpy())
        _iree_bf16 = torch.from_numpy(np.array(_iree_out)).to(torch.bfloat16)
        _diff = (EXPECTED.float() - _iree_bf16.float()).abs().max().item()
        assert _diff < 5e-2, f"MLIR vs PyTorch mismatch: {_diff}"
    except ImportError:
        pass
DRAM_X_H0 = 0x0000
DRAM_X_H1 = 0x0400
DRAM_C044 = 0x0800  # 0.044715 broadcast tile
DRAM_CSQRT = 0x0C00  # sqrt(2/pi) broadcast tile
DRAM_CHALF = 0x1000  # 0.5 broadcast tile
DRAM_OUT_H0 = 0x1400
DRAM_OUT_H1 = 0x1800

_c044 = torch.full((32, 16), 0.044715, dtype=torch.bfloat16)
_csqrt = torch.full((32, 16), math.sqrt(2.0 / math.pi), dtype=torch.bfloat16)
_chalf = torch.full((32, 16), 0.5, dtype=torch.bfloat16)
EXPECTED_STACKED = torch.cat((EXPECTED[:, :16], EXPECTED[:, 16:]), dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# 4. NPU ISA program.
# ═══════════════════════════════════════════════════════════════════════════


class SmolVLAGeluTanhProgram(Program):
    """Auto-generated single-file Program for the ``gelu_tanh`` kernel.

    ISA is lifted from the merlin kernel manifest (see
    ``benchmarks/SaturnNPU/kernel_library/manifest.json``). This Program
    mirrors the ``smolvla_silu.py`` template: self-contained, no cross-
    file helpers, torch-allclose golden check via ``pytest tests/test_programs.py``.
    """

    # Scalar register map:
    #   x1 = VMEM base for x pair        (0x2000, both halves at +0 and +1024)
    #   x2 = VMEM base for C044 pair     (0x3000)
    #   x3 = VMEM base for Csqrt pair    (0x4000)
    #   x4 = VMEM base for Chalf pair    (0x5000)
    #   x5 = VMEM OUT base               (0x6000)
    #   x6 = DRAM_X_H0 = 0              x7 = DRAM_X_H1 = 0x0400
    #   x8 = DRAM_C044 = 0x0800         x9 = DRAM_CSQRT = 0x0C00
    #   x10 = DRAM_CHALF = 0x1000       x11 = DRAM_OUT_H0 = 0x1400
    #   x12 = DRAM_OUT_H1 = 0x1800      x13 = 1024 (per-half size)
    #   x14 = x1+1024 (second VMEM half for x)
    #   x15 = x2+1024, x16 = x3+1024, x17 = x4+1024 (second halves for constants)
    instructions: List[Instruction[Any]] = [
        # VMEM base addresses
        Instruction("lui", ScalarArgs(rd=1, imm=0x2)),   # 0x2000 VMEM x
        Instruction("lui", ScalarArgs(rd=2, imm=0x3)),   # 0x3000 VMEM C044
        Instruction("lui", ScalarArgs(rd=3, imm=0x4)),   # 0x4000 VMEM Csqrt
        Instruction("lui", ScalarArgs(rd=4, imm=0x5)),   # 0x5000 VMEM Chalf
        Instruction("lui", ScalarArgs(rd=5, imm=0x6)),   # 0x6000 VMEM OUT
        # DRAM addresses
        Instruction("addi", ScalarArgs(rd=6, rs1=0, imm=DRAM_X_H0)),    # 0x0000
        Instruction("addi", ScalarArgs(rd=7, rs1=6, imm=1024)),           # 0x0400
        Instruction("lui", ScalarArgs(rd=8, imm=0x1)),
        Instruction("addi", ScalarArgs(rd=8, rs1=8, imm=-2048)),          # 0x0800
        Instruction("addi", ScalarArgs(rd=9, rs1=8, imm=1024)),           # 0x0C00
        Instruction("lui", ScalarArgs(rd=10, imm=0x1)),                   # 0x1000
        Instruction("lui", ScalarArgs(rd=11, imm=0x1)),
        Instruction("addi", ScalarArgs(rd=11, rs1=11, imm=1024)),         # 0x1400
        Instruction("addi", ScalarArgs(rd=12, rs1=11, imm=1024)),         # 0x1800
        Instruction("addi", ScalarArgs(rd=13, rs1=0, imm=1024)),          # 1024
        # Secondary VMEM offsets for second halves of each pair
        Instruction("addi", ScalarArgs(rd=14, rs1=1, imm=1024)),          # VMEM x +1024
        Instruction("addi", ScalarArgs(rd=15, rs1=2, imm=1024)),          # VMEM C044 +1024
        Instruction("addi", ScalarArgs(rd=16, rs1=3, imm=1024)),          # VMEM Csqrt +1024
        Instruction("addi", ScalarArgs(rd=17, rs1=4, imm=1024)),          # VMEM Chalf +1024
        # DMA: load x halves
        Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=0)),
        Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=1)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=6, rs2=13, channel=0)),   # x_h0 → VMEM[x1]
        Instruction("dma.load.ch<N>", DmaArgs(rd=14, rs1=7, rs2=13, channel=1)),  # x_h1 → VMEM[x1+1024]
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        # DMA: load C044 into both halves of the pair
        Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=8, rs2=13, channel=0)),   # C044 → m2
        Instruction("dma.load.ch<N>", DmaArgs(rd=15, rs1=8, rs2=13, channel=1)),  # C044 → m3
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        # DMA: load Csqrt into both halves, and Chalf into both halves
        Instruction("dma.load.ch<N>", DmaArgs(rd=3, rs1=9, rs2=13, channel=0)),   # Csqrt → m4
        Instruction("dma.load.ch<N>", DmaArgs(rd=16, rs1=9, rs2=13, channel=1)),  # Csqrt → m5
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=4, rs1=10, rs2=13, channel=0)),  # Chalf → m6
        Instruction("dma.load.ch<N>", DmaArgs(rd=17, rs1=10, rs2=13, channel=1)), # Chalf → m7
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
        # Load x into (m0, m1): two vloads (vload is per-register)
        Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)),   # m0 = x_h0
        Instruction("delay", ScalarArgs(imm=16)),
        Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=32)),  # m1 = x_h1
        Instruction("delay", ScalarArgs(imm=16)),
        # Load constant pairs (vload is per-register)
        Instruction("vload", VectorArgs(vd=2, rs1=2, imm12=0)),   # m2 = C044
        Instruction("delay", ScalarArgs(imm=16)),
        Instruction("vload", VectorArgs(vd=3, rs1=2, imm12=32)),  # m3 = C044 (pair)
        Instruction("delay", ScalarArgs(imm=16)),
        Instruction("vload", VectorArgs(vd=4, rs1=3, imm12=0)),   # m4 = Csqrt
        Instruction("delay", ScalarArgs(imm=16)),
        Instruction("vload", VectorArgs(vd=5, rs1=3, imm12=32)),  # m5 = Csqrt (pair)
        Instruction("delay", ScalarArgs(imm=16)),
        Instruction("vload", VectorArgs(vd=6, rs1=4, imm12=0)),   # m6 = Chalf
        Instruction("delay", ScalarArgs(imm=16)),
        Instruction("vload", VectorArgs(vd=7, rs1=4, imm12=32)),  # m7 = Chalf (pair)
        Instruction("delay", ScalarArgs(imm=16)),
        # 1.0 constants via vli.all (vli.all is per-register)
        Instruction("vli.all", VectorArgs(vd=8, imm=1)),   # m8 = 1.0
        Instruction("delay", ScalarArgs(imm=65)),
        Instruction("vli.all", VectorArgs(vd=9, imm=1)),   # m9 = 1.0 (pair)
        Instruction("delay", ScalarArgs(imm=65)),
        # ── Pair-op GELU computation ──────────────────────────────────────────
        # (m10, m11) = x^2
        Instruction("vmul.bf16", VectorArgs(vd=10, vs1=0, vs2=0)),
        Instruction("delay", ScalarArgs(imm=66)),
        # (m12, m13) = x^3 = x^2 * x
        Instruction("vmul.bf16", VectorArgs(vd=12, vs1=10, vs2=0)),
        Instruction("delay", ScalarArgs(imm=66)),
        # (m14, m15) = 0.044715 * x^3
        Instruction("vmul.bf16", VectorArgs(vd=14, vs1=12, vs2=2)),
        Instruction("delay", ScalarArgs(imm=66)),
        # (m16, m17) = x + 0.044715*x^3
        Instruction("vadd.bf16", VectorArgs(vd=16, vs1=0, vs2=14)),
        Instruction("delay", ScalarArgs(imm=66)),
        # (m18, m19) = sqrt(2/pi) * (x + 0.044715*x^3)
        Instruction("vmul.bf16", VectorArgs(vd=18, vs1=16, vs2=4)),
        Instruction("delay", ScalarArgs(imm=66)),
        # (m20, m21) = tanh(sqrt(2/pi) * (...))
        Instruction("vtanh.bf16", VectorArgs(vd=20, vs1=18)),
        Instruction("delay", ScalarArgs(imm=66)),
        # (m22, m23) = 1 + tanh(...)
        Instruction("vadd.bf16", VectorArgs(vd=22, vs1=8, vs2=20)),
        Instruction("delay", ScalarArgs(imm=66)),
        # (m24, m25) = 0.5 * x
        Instruction("vmul.bf16", VectorArgs(vd=24, vs1=0, vs2=6)),
        Instruction("delay", ScalarArgs(imm=66)),
        # (m26, m27) = GELU(x) = 0.5*x * (1 + tanh(...))
        Instruction("vmul.bf16", VectorArgs(vd=26, vs1=24, vs2=22)),
        Instruction("delay", ScalarArgs(imm=66)),
        # ── Store: (m26, m27) → VMEM → DRAM ─────────────────────────────────
        # x18 = VMEM OUT + 1024 (second half of output pair)
        Instruction("addi", ScalarArgs(rd=18, rs1=5, imm=1024)),
        Instruction("vstore", VectorArgs(vd=26, rs1=5, imm12=0)),
        Instruction("delay", ScalarArgs(imm=16)),
        Instruction("vstore", VectorArgs(vd=27, rs1=5, imm12=32)),
        Instruction("delay", ScalarArgs(imm=16)),
        Instruction("dma.store.ch<N>", DmaArgs(rd=11, rs1=5, rs2=13, channel=0)),   # m26 → DRAM_OUT_H0
        Instruction("dma.store.ch<N>", DmaArgs(rd=12, rs1=18, rs2=13, channel=1)),  # m27 → DRAM_OUT_H1
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_X_H0, INPUT[:, :16].contiguous()),
        (DRAM_X_H1, INPUT[:, 16:].contiguous()),
        (DRAM_C044, _c044),
        (DRAM_CSQRT, _csqrt),
        (DRAM_CHALF, _chalf),
    ]

    golden_result: tuple[int, torch.Tensor] = (DRAM_OUT_H0, EXPECTED_STACKED)
