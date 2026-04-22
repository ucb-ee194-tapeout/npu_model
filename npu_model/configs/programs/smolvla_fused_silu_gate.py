"""SmolVLA fused_silu_gate kernel: sigmoid(x) then x * sigmoid(x) = silu(x).

MLIR sources (benchmarks/SaturnNPU/kernels/fused_silu_gate):
    op_a.mlir — sigmoid(x) = 1 / (1 + exp(-x))           bf16
    op_b.mlir — elementwise_mul(op_a(x), x)               bf16
Pattern appears 32× in the model. Fusing eliminates the intermediate
sigmoid materialisation. Functionally identical to ``smolvla_silu``.

MLIR → ISA mapping (pair-op BF16):
    op_a (sigmoid):
        arith.negf          → vmul.bf16(x, -1)           pair-op
        math.exp            → vexp.bf16
        arith.addf 1.0      → vadd.bf16(exp(-x), +1)
        arith.divf 1.0      → vrecip.bf16(denom)
    op_b (silu = sigmoid * x):
        arith.mulf          → vmul.bf16(sigmoid, x)

Demo tile: one (32, 32) bf16 tile. The full MLIR shape is (50, 720);
this program exercises the op-graph on a tile-sized slice.
"""

import os

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs


# ═══════════════════════════════════════════════════════════════════════════
# 1. MLIR definition — op_a + op_b fused into one func for IREE.
# ═══════════════════════════════════════════════════════════════════════════

FUSED_SILU_GATE_MLIR = """\
func.func @fused_silu_gate(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %empty0 = tensor.empty() : tensor<32x32xf32>
  %sig = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<32x32xf32>) outs(%empty0 : tensor<32x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %neg = arith.negf %in : f32
    %exp = math.exp %neg : f32
    %one = arith.constant 1.0 : f32
    %den = arith.addf %exp, %one : f32
    %res = arith.divf %one, %den : f32
    linalg.yield %res : f32
  } -> tensor<32x32xf32>
  %empty1 = tensor.empty() : tensor<32x32xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%sig, %arg0 : tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%empty1 : tensor<32x32xf32>) {
  ^bb0(%in_sig: f32, %in_x: f32, %out: f32):
    %res = arith.mulf %in_sig, %in_x : f32
    linalg.yield %res : f32
  } -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}
"""


# ═══════════════════════════════════════════════════════════════════════════
# 2. PyTorch reference.
# ═══════════════════════════════════════════════════════════════════════════


def fused_silu_gate_reference(x: torch.Tensor) -> torch.Tensor:
    """SiLU(x) = x * sigmoid(x). Matches the op_a + op_b chain."""
    xf = x.float()
    return (xf * torch.sigmoid(xf)).to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Golden data.
# ═══════════════════════════════════════════════════════════════════════════

torch.manual_seed(42)
INPUT = torch.randn(32, 32, dtype=torch.bfloat16)
EXPECTED = fused_silu_gate_reference(INPUT)

# Optional IREE crosscheck. Gated so CI never imports iree-runtime (the
# extension leaks nanobind MappedMemory / HalBufferView at shutdown).
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
            FUSED_SILU_GATE_MLIR,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
        _cfg = runtime.Config("local-task")
        _ctx = runtime.SystemContext(config=_cfg)
        _ctx.add_vm_module(runtime.VmModule.copy_buffer(_ctx.instance, _vmfb))
        _iree_out = _ctx.modules.module["fused_silu_gate"](INPUT.float().numpy())
        _iree_bf16 = torch.from_numpy(np.array(_iree_out)).to(torch.bfloat16)
        _diff = (EXPECTED.float() - _iree_bf16.float()).abs().max().item()
        assert _diff < 1e-2, f"MLIR vs PyTorch mismatch: {_diff}"
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 4. Memory layout.
#
# (32, 32) bf16 tile = 2048 B, handled as two 32x16 halves back-to-back
# in DRAM (first 1024 B = cols 0:16, next 1024 B = cols 16:32).
# ═══════════════════════════════════════════════════════════════════════════

DRAM_X_BASE = 0x0000
DRAM_OUT_BASE = 0x0800
TILE_BYTES = 2048


# ═══════════════════════════════════════════════════════════════════════════
# 5. NPU ISA program (pair-op BF16).
#
# MRF map:
#   (m0, m1)   = x                 (m2, m3)   = -1.0 const
#   (m4, m5)   = +1.0 const        (m6, m7)   = -x
#   (m8, m9)   = exp(-x)           (m10, m11) = 1 + exp(-x)
#   (m12, m13) = sigmoid(x)        (m14, m15) = silu(x) = sigmoid(x) * x
#
# Scalar reg map:
#   x1 = VMEM x base    x2 = VMEM out base
#   x3 = DRAM x base    x4 = DRAM out base    x5 = transfer size (2048)
# ═══════════════════════════════════════════════════════════════════════════


class SmolVLAFusedSiluGateProgram(Program):
    """fused_silu_gate: silu(x) = x * sigmoid(x) on a 32x32 bf16 tile."""

    instructions: List[Instruction[Any]] = [
        # Scalar setup
        Instruction(mnemonic="lui", args=ScalarArgs(rd=1, imm=0x2)),  # 0x2000
        Instruction(mnemonic="lui", args=ScalarArgs(rd=2, imm=0x3)),  # 0x3000
        Instruction(mnemonic="addi", args=ScalarArgs(rd=3, rs1=0, imm=DRAM_X_BASE)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=4, imm=0x1)),  # 0x1000
        Instruction(mnemonic="addi", args=ScalarArgs(rd=4, rs1=4, imm=-2048)),  # 0x0800
        # Transfer size = 2048 via lui + addi (exceeds signed 12-bit immediate).
        Instruction(mnemonic="lui", args=ScalarArgs(rd=5, imm=0x1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=5, rs1=5, imm=-2048)),
        # DMA: DRAM → VMEM
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=1, rs1=3, rs2=5, channel=0)
        ),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        # Load both halves of x into (m0, m1)
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=1, rs1=1, imm12=32)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        # Broadcast constants (vli.all is per-register, so two each)
        Instruction(mnemonic="vli.all", args=VectorArgs(vd=2, imm=-1)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=65)),
        Instruction(mnemonic="vli.all", args=VectorArgs(vd=3, imm=-1)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=65)),
        Instruction(mnemonic="vli.all", args=VectorArgs(vd=4, imm=1)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=65)),
        Instruction(mnemonic="vli.all", args=VectorArgs(vd=5, imm=1)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=65)),
        # op_a: sigmoid(x) = 1 / (1 + exp(-x))
        Instruction(
            mnemonic="vmul.bf16", args=VectorArgs(vd=6, vs1=0, vs2=2)
        ),  # (m6, m7) = -x
        Instruction(mnemonic="delay", args=ScalarArgs(imm=66)),
        Instruction(
            mnemonic="vexp.bf16", args=VectorArgs(vd=8, vs1=6)
        ),  # (m8, m9) = exp(-x)
        Instruction(mnemonic="delay", args=ScalarArgs(imm=66)),
        Instruction(
            mnemonic="vadd.bf16", args=VectorArgs(vd=10, vs1=8, vs2=4)
        ),  # (m10, m11) = 1 + exp(-x)
        Instruction(mnemonic="delay", args=ScalarArgs(imm=66)),
        Instruction(
            mnemonic="vrecip.bf16", args=VectorArgs(vd=12, vs1=10)
        ),  # (m12, m13) = sigmoid(x)
        Instruction(mnemonic="delay", args=ScalarArgs(imm=66)),
        # op_b: silu(x) = sigmoid(x) * x
        Instruction(
            mnemonic="vmul.bf16", args=VectorArgs(vd=14, vs1=12, vs2=0)
        ),  # (m14, m15) = silu(x)
        Instruction(mnemonic="delay", args=ScalarArgs(imm=66)),
        # Store both halves
        Instruction(mnemonic="vstore", args=VectorArgs(vd=14, rs1=2, imm12=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=15, rs1=2, imm12=32)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(
            mnemonic="dma.store.ch<N>", args=DmaArgs(rd=4, rs1=2, rs2=5, channel=0)
        ),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_X_BASE, INPUT),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUT_BASE,
        EXPECTED,
    )
