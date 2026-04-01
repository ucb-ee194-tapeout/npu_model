from typing import List, Tuple
import torch
from npu_model.isa import (
    DmaArgs,
    MatrixArgs,
    ScalarArgs,
    VectorArgs,
)
from ...software import Instruction, Program

# Constants for memory layout
DRAM_ACTIVATION_BASE = 0x0000
DRAM_WEIGHT_BASE = 0x0400
DRAM_OUTPUT_BASE = 0x0800
VMEM_ACTIVATION_BASE = 0x2000
VMEM_WEIGHT_BASE = 0x2400
VMEM_OUTPUT_BASE = 0x2800

# Mock data for matmul verification
ACTIVATION_DATA = torch.eye(32, 32, dtype=torch.float8_e4m3fn)
WEIGHT_DATA = (2 * torch.eye(32, 32, dtype=torch.float32)).to(torch.float8_e4m3fn)
MATMUL_RESULT = (ACTIVATION_DATA.to(torch.float32) @ WEIGHT_DATA.to(torch.float32)).to(
    torch.bfloat16
)


class MatmulProgram(Program):
    """
    Rewritten Matmul test using structured Args dataclasses.
    """

    instructions: List[Instruction] = [
        Instruction(
            mnemonic="addi", args=ScalarArgs(rd=1, rs1=0, imm=VMEM_ACTIVATION_BASE)
        ),
        Instruction(
            mnemonic="addi", args=ScalarArgs(rd=2, rs1=0, imm=VMEM_WEIGHT_BASE)
        ),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=3, rs1=0, imm=VMEM_OUTPUT_BASE)),
        Instruction(
            mnemonic="dma.load",
            args=DmaArgs(rd=8, base=DRAM_ACTIVATION_BASE, size=1024, flag=0),
        ),
        Instruction(
            mnemonic="dma.load",
            args=DmaArgs(rd=9, base=DRAM_WEIGHT_BASE, size=1024, flag=1),
        ),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=0)),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=1)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=8, rs1=1, offset=0)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=9, rs1=2, offset=0)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, offset=0)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=1, rs1=2, offset=0)),
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=0, vs1=1)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=0), delay=16),
        Instruction(mnemonic="vmatmul.mxu0", args=VectorArgs(vd=0, vs1=0, vs2=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=0), delay=32),
        Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=2, vs2=0)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=2, rs1=3, offset=0)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=3, rs1=3, offset=32)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=10, rs1=3, offset=0)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=11, rs1=3, offset=32)),
        Instruction(mnemonic="delay", args=VectorArgs(imm=0), delay=16),
        Instruction(
            mnemonic="dma.store",
            args=DmaArgs(rs1=10, base=DRAM_OUTPUT_BASE, size=1024, flag=0),
        ),
        Instruction(
            mnemonic="dma.store",
            args=DmaArgs(rs1=11, base=DRAM_OUTPUT_BASE + 1024, size=1024, flag=1),
        ),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=0)),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=1)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_ACTIVATION_BASE, ACTIVATION_DATA),
        (DRAM_WEIGHT_BASE, WEIGHT_DATA),
    ]

    golden_result: Tuple[int, torch.Tensor] = (
        DRAM_OUTPUT_BASE,
        torch.cat((MATMUL_RESULT[:, :16], MATMUL_RESULT[:, 16:]), dim=0),
    )
