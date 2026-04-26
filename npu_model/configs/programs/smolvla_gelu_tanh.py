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
    """GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).

    Each constant tile (C044, Csqrt, Chalf) is loaded once from DRAM into
    VMEM. Both pair halves (m2/m3, m4/m5, m6/m7) are populated by vloading
    from the same VMEM address — correct since the constant is broadcast
    across the full tile. Saves 3 DMA loads vs the naive two-per-pair approach.

    DMA schedule (sequential engine): x_H0 → x_H1 → C044 → Csqrt → (Chalf
    queued on ch0 after x_H0 clears). Compute pipelines into each wait window.

    Scalar register map:
      x1=VMEM x (0x2000)  x2=VMEM C044 (0x3000)  x3=VMEM Csqrt (0x4000)
      x4=VMEM Chalf (0x5000)  x5=VMEM OUT (0x6000)
      x6=DRAM_X_H0  x7=DRAM_X_H1  x8=DRAM_C044  x9=DRAM_CSQRT
      x10=DRAM_CHALF  x11=DRAM_OUT_H0  x12=DRAM_OUT_H1  x13=1024
      x14=VMEM x+1024 (x_H1 DMA dest)  x18=VMEM OUT+1024 (H1 store src)
    """

    instructions: List[Instruction[Any]] = [
        # VMEM base addresses
        Instruction("lui", ScalarArgs(rd=1, imm=0x2)),   # 0x2000 VMEM x
        Instruction("lui", ScalarArgs(rd=2, imm=0x3)),   # 0x3000 VMEM C044
        Instruction("lui", ScalarArgs(rd=3, imm=0x4)),   # 0x4000 VMEM Csqrt
        Instruction("lui", ScalarArgs(rd=4, imm=0x5)),   # 0x5000 VMEM Chalf
        Instruction("lui", ScalarArgs(rd=5, imm=0x6)),   # 0x6000 VMEM OUT
        # DRAM addresses
        Instruction("addi", ScalarArgs(rd=6, rs1=0, imm=DRAM_X_H0)),   # 0x0000
        Instruction("addi", ScalarArgs(rd=7, rs1=6, imm=1024)),         # 0x0400 x_H1
        Instruction("lui", ScalarArgs(rd=8, imm=0x1)),
        Instruction("addi", ScalarArgs(rd=8, rs1=8, imm=-2048)),        # 0x0800 C044
        Instruction("addi", ScalarArgs(rd=9, rs1=8, imm=1024)),         # 0x0C00 Csqrt
        Instruction("lui", ScalarArgs(rd=10, imm=0x1)),                 # 0x1000 Chalf
        Instruction("lui", ScalarArgs(rd=11, imm=0x1)),
        Instruction("addi", ScalarArgs(rd=11, rs1=11, imm=1024)),       # 0x1400 OUT_H0
        Instruction("addi", ScalarArgs(rd=12, rs1=11, imm=1024)),       # 0x1800 OUT_H1
        Instruction("addi", ScalarArgs(rd=13, rs1=0, imm=1024)),        # 1024
        Instruction("addi", ScalarArgs(rd=14, rs1=1, imm=1024)),        # VMEM x+1024
        Instruction("addi", ScalarArgs(rd=18, rs1=5, imm=1024)),        # VMEM OUT+1024
        # Queue all 4 loads: x_H0 (ch0,516cy) → x_H1 (ch1,516cy) → C044
        # (ch2,516cy) → Csqrt (ch3,516cy). Constants loaded once only —
        # both pair halves are populated by vloading the same VMEM address.
        Instruction("dma.config.ch<N>", DmaArgs(rs1=0, channel=0)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=1,  rs1=6,  rs2=13, channel=0)),  # x_H0
        Instruction("dma.load.ch<N>", DmaArgs(rd=14, rs1=7,  rs2=13, channel=1)),  # x_H1
        Instruction("dma.load.ch<N>", DmaArgs(rd=2,  rs1=8,  rs2=13, channel=2)),  # C044
        Instruction("dma.load.ch<N>", DmaArgs(rd=3,  rs1=9,  rs2=13, channel=3)),  # Csqrt
        # x_H0 done at ~516cy. Reuse ch0 for Chalf; queues behind Csqrt,
        # so Chalf starts at ~2064cy and finishes at ~2580cy.
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),
        Instruction("dma.load.ch<N>", DmaArgs(rd=4,  rs1=10, rs2=13, channel=0)),  # Chalf
        Instruction("dma.wait.ch<N>", DmaArgs(channel=1)),  # x_H1 done at ~1032cy
        # x ready. C044 loading (~1032-1548cy). Compute x², x³ and fill
        # the wait bubble with vli.all for the 1.0 constants.
        Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)),   # m0 = x_H0
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=32)),  # m1 = x_H1
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vmul.bf16", VectorArgs(vd=10, vs1=0, vs2=0)),   # (m10,m11) = x²
        Instruction("delay", ScalarArgs(imm=66)),
        Instruction("vmul.bf16", VectorArgs(vd=12, vs1=10, vs2=0)),  # (m12,m13) = x³
        Instruction("delay", ScalarArgs(imm=66)),
        Instruction("vli.all", VectorArgs(vd=8, imm=1)),   # m8 = 1.0
        Instruction("delay", ScalarArgs(imm=65)),
        Instruction("vli.all", VectorArgs(vd=9, imm=1)),   # m9 = 1.0 (pair)
        Instruction("delay", ScalarArgs(imm=65)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=2)),  # C044 done at ~1548cy
        # Load C044 into both pair halves from the same VMEM address.
        # Csqrt loading (~1548-2064cy). Compute 0.044715*x³ and x+0.044715*x³.
        Instruction("vload", VectorArgs(vd=2, rs1=2, imm12=0)),  # m2 = C044
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vload", VectorArgs(vd=3, rs1=2, imm12=0)),  # m3 = C044 (same addr)
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vmul.bf16", VectorArgs(vd=14, vs1=12, vs2=2)),  # (m14,m15) = 0.044715*x³
        Instruction("delay", ScalarArgs(imm=66)),
        Instruction("vadd.bf16", VectorArgs(vd=16, vs1=0, vs2=14)),  # (m16,m17) = x+0.044715*x³
        Instruction("delay", ScalarArgs(imm=66)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=3)),  # Csqrt done at ~2064cy
        # Load Csqrt into both pair halves from same VMEM address.
        # Chalf loading (~2064-2580cy). Compute sqrt*(...), tanh, 1+tanh.
        Instruction("vload", VectorArgs(vd=4, rs1=3, imm12=0)),  # m4 = Csqrt
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vload", VectorArgs(vd=5, rs1=3, imm12=0)),  # m5 = Csqrt (same addr)
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vmul.bf16", VectorArgs(vd=18, vs1=16, vs2=4)),  # (m18,m19) = sqrt(2/pi)*(x+...)
        Instruction("delay", ScalarArgs(imm=66)),
        Instruction("vtanh.bf16", VectorArgs(vd=20, vs1=18)),         # (m20,m21) = tanh(...)
        Instruction("delay", ScalarArgs(imm=66)),
        Instruction("vadd.bf16", VectorArgs(vd=22, vs1=8, vs2=20)),  # (m22,m23) = 1+tanh(...)
        Instruction("delay", ScalarArgs(imm=66)),
        Instruction("dma.wait.ch<N>", DmaArgs(channel=0)),  # Chalf done at ~2580cy
        # Load Chalf into both pair halves from same VMEM address.
        Instruction("vload", VectorArgs(vd=6, rs1=4, imm12=0)),  # m6 = Chalf
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vload", VectorArgs(vd=7, rs1=4, imm12=0)),  # m7 = Chalf (same addr)
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("vmul.bf16", VectorArgs(vd=24, vs1=0, vs2=6)),   # (m24,m25) = 0.5*x
        Instruction("delay", ScalarArgs(imm=66)),
        Instruction("vmul.bf16", VectorArgs(vd=26, vs1=24, vs2=22)), # (m26,m27) = GELU(x)
        Instruction("delay", ScalarArgs(imm=66)),
        # Store H0, fire dma.store H0 async, then vstore H1 via LSU while
        # DMA H0 is running (different EXUs, non-overlapping VMEM regions).
        Instruction("vstore", VectorArgs(vd=26, rs1=5, imm12=0)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("dma.store.ch<N>", DmaArgs(rd=11, rs1=5,  rs2=13, channel=0)),
        Instruction("vstore", VectorArgs(vd=27, rs1=5, imm12=32)),
        Instruction("delay", ScalarArgs(imm=34)),
        Instruction("dma.store.ch<N>", DmaArgs(rd=12, rs1=18, rs2=13, channel=1)),
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
