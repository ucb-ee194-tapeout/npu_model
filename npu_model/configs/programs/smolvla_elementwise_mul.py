"""SmolVLA elementwise multiply kernel.

Multiplies two 32x32 bf16 tiles elementwise. Appears 664 times across
SmolVLA (gate/up fusion in MLPs, norm×scale, attention score×mask). 19
shape variants in the full model; this Program implements the 32x32
canonical form.

MLIR → ISA mapping:
    arith.mulf %a %b → vmul.bf16(a, b)   pair-op: (vd, vd+1) = (vs1, vs1+1) * (vs2, vs2+1)
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs


ELEMENTWISE_MUL_MLIR = """\
func.func @elementwise_mul(
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
    %p = arith.mulf %lhs, %rhs : f32
    linalg.yield %p : f32
  } -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}
"""


def elementwise_mul_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() * b.float()).to(a.dtype)


torch.manual_seed(43)
INPUT_A = torch.randn(32, 32, dtype=torch.bfloat16)
INPUT_B = torch.randn(32, 32, dtype=torch.bfloat16)

EXPECTED = elementwise_mul_reference(INPUT_A, INPUT_B)

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
            ELEMENTWISE_MUL_MLIR,
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
        _config = runtime.Config("local-task")
        _ctx = runtime.SystemContext(config=_config)
        _ctx.add_vm_module(runtime.VmModule.copy_buffer(_ctx.instance, _vmfb))
        _iree_out = _ctx.modules.module["elementwise_mul"](
            INPUT_A.float().numpy(), INPUT_B.float().numpy()
        )
        _iree_expected = torch.from_numpy(np.array(_iree_out)).to(torch.bfloat16)
        _diff = (EXPECTED.float() - _iree_expected.float()).abs().max().item()
        assert _diff < 1e-3, f"MLIR vs PyTorch mismatch: {_diff}"
        EXPECTED = _iree_expected
    except ImportError:
        pass


DRAM_A_BASE = 0x0000
DRAM_B_BASE = 0x1000
DRAM_OUTPUT_BASE = 0x2000
VMEM_A_BASE = 0x4000
VMEM_B_BASE = 0x5000
VMEM_OUTPUT_BASE = 0x6000
TILE_BYTES = 2048  # 32 * 32 * 2 (bf16)


class SmolVLAElementwiseMulProgram(Program):
    """y = a * b on two 32x32 bf16 tiles (elementwise). cycles: ~270"""

    instructions: List[Instruction[Any]] = [
        Instruction(mnemonic="lui", args=ScalarArgs(rd=1, imm=0x4)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=2, imm=0x5)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=3, imm=0x6)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=4, rs1=0, imm=DRAM_A_BASE)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=5, imm=0x1)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=6, imm=0x2)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=7, imm=0x1)),
        Instruction(
            mnemonic="addi", args=ScalarArgs(rd=7, rs1=7, imm=-2048)
        ),  # x7 = 2048
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=1, rs1=4, rs2=7, channel=0)
        ),
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=2, rs1=5, rs2=7, channel=1)
        ),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=1, rs1=1, imm12=32)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=2, rs1=2, imm12=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=3, rs1=2, imm12=32)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(
            mnemonic="vmul.bf16", args=VectorArgs(vd=4, vs1=0, vs2=2)
        ),  # (v4, v5) = (v0, v1) * (v2, v3)
        Instruction(mnemonic="delay", args=ScalarArgs(imm=66)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=4, rs1=3, imm12=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=5, rs1=3, imm12=32)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(
            mnemonic="dma.store.ch<N>", args=DmaArgs(rd=6, rs1=3, rs2=7, channel=0)
        ),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_A_BASE, INPUT_A),
        (DRAM_B_BASE, INPUT_B),
    ]

    golden_result: tuple[int, torch.Tensor] = (DRAM_OUTPUT_BASE, EXPECTED)
