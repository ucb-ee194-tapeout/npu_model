from typing import List, Tuple, Any
import torch
from npu_model.isa import (
    DmaArgs,
    MatrixArgs,  # type: ignore (unused)
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

    instructions: List[Instruction[Any]] = [
        # DRAM_ACTIVATION_BASE = 0x0000 (Fits in addi)
        Instruction(mnemonic="addi", args=ScalarArgs(rd=4, rs1=0, imm=0x000)),
        # DRAM_WEIGHT_BASE = 0x0400 (Fits in addi: 1024)
        Instruction(mnemonic="addi", args=ScalarArgs(rd=5, rs1=0, imm=0x400)),
        # VMEM_ACTIVATION_BASE = 0x2000
        Instruction(mnemonic="lui", args=ScalarArgs(rd=1, imm=0x2)),
        # VMEM_WEIGHT_BASE = 0x2400
        Instruction(mnemonic="lui", args=ScalarArgs(rd=2, imm=0x2)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=2, rs1=2, imm=0x400)),
        # VMEM_OUTPUT_BASE = 0x2800
        # Note: 0x800 is -2048 in signed 12-bit.
        # To get 0x2800, we load 0x3000 (lui 3) then subtract 0x800.
        Instruction(mnemonic="lui", args=ScalarArgs(rd=3, imm=0x3)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=3, rs1=3, imm=-2048)),
        # x6 = 1024 (size for vmem store)
        Instruction(mnemonic="addi", args=ScalarArgs(rd=6, rs1=0, imm=1024)),
        # set DMA base
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        # store activation into VMEM (vmem[x1] = activation)
        Instruction(
            mnemonic="dma.load.ch<N>",
            args=DmaArgs(rd=1, rs1=4, rs2=6, channel=0),
        ),
        # store weight into vmem (vmem[x2] = weight)
        Instruction(
            mnemonic="dma.load.ch<N>",
            args=DmaArgs(rd=2, rs1=5, rs2=6, channel=1),
        ),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
        # load weights/activations from vmem
        Instruction(
            mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)
        ),  # mrf[v0] = activations
        Instruction(
            mnemonic="vload", args=VectorArgs(vd=1, rs1=2, imm12=0)
        ),  # mrf[v1] = weights
        # push to weight buffer, matmul, and pop from accumulation buffer
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=0, vs1=1)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=16)),
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=0, vs2=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=32)),
        Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=2, vs1=0)),
        # store to vmem
        Instruction(mnemonic="vstore", args=VectorArgs(vd=2, rs1=3, imm12=0)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=3, rs1=3, imm12=32)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=2048)),
        # store to dram
        # DRAM_OUTPUT_BASE = 0x0800 (2048)
        # To get 0x800: LUI 1 (0x1000) + ADDI -2048 = 0x0800
        Instruction(mnemonic="lui", args=ScalarArgs(rd=10, imm=1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=10, rs1=10, imm=-2048)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=11, rs1=10, imm=1024)),
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
