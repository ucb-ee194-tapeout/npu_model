from typing import List, Tuple
import torch
from npu_model.isa import (
    DmaArgs,
    MatrixArgs, # type: ignore (unused)
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
        Instruction(
            mnemonic="addi", args=ScalarArgs(rd=3, rs1=0, imm=VMEM_OUTPUT_BASE)
        ),
        # dram weights/activations -> vmem
        Instruction(
            mnemonic="addi", args=ScalarArgs(rd=4, rs1=0, imm=DRAM_ACTIVATION_BASE)
        ),
        Instruction(
            mnemonic="addi", args=ScalarArgs(rd=5, rs1=0, imm=DRAM_WEIGHT_BASE)
        ),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=6, rs1=0, imm=1024)),
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(
            mnemonic="dma.load.ch<N>",
            args=DmaArgs(rd=1, rs1=4, rs2=6, channel=0),
        ),
        Instruction(
            mnemonic="dma.load.ch<N>",
            args=DmaArgs(rd=2, rs1=5, rs2=6, channel=1),
        ),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
        # load weights/activations from vmem
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=2, imm12=0)),
        # push to weight buffer, matmul, and pop from accumulation buffer
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=0, vs1=1)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=0), delay=16),
        Instruction(mnemonic="vmatmul.mxu0", args=VectorArgs(vd=0, vs1=0, vs2=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=0), delay=32),
        Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=2, vs2=0)),
        # store to vmem
        Instruction(mnemonic="vstore", args=VectorArgs(vd=2, rs1=3, imm12=0)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=3, rs1=3, imm12=32)),
        # store to dram
        Instruction(
            mnemonic="addi", args=ScalarArgs(rd=10, rs1=0, imm=DRAM_OUTPUT_BASE)
        ),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=11, rs1=0, imm=1024)),
        Instruction(
            mnemonic="dma.store.ch<N>", args=DmaArgs(rd=10, rs1=3, rs2=6, channel=0)
        ),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=3, rs1=3, imm=1024)),
        Instruction(
            mnemonic="dma.store.ch<N>", args=DmaArgs(rd=11, rs1=3, rs2=6, channel=1)
        ),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_ACTIVATION_BASE, ACTIVATION_DATA),
        (DRAM_WEIGHT_BASE, WEIGHT_DATA),
    ]

    golden_result: Tuple[int, torch.Tensor] = (
        DRAM_OUTPUT_BASE,
        torch.cat((MATMUL_RESULT[:, :16], MATMUL_RESULT[:, 16:]), dim=0),
    )
