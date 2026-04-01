from typing import List, Tuple
from npu_model.software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs
import torch


class DMAStallProgram(Program):
    """
    A simple program demonstrating DMA loads, stalling logic, and matrix multiplication
    updated for the latest npu_model ISA.
    """

    instructions: List[Instruction] = [
        # Load things w/ Matmul
        Instruction(mnemonic="dma.load", args=DmaArgs(rd=2, base=0, size=1024, flag=0)),
        Instruction(
            mnemonic="dma.load.mxu0", args=DmaArgs(rd=1, base=1024, size=1024, flag=1)
        ),
        Instruction(
            mnemonic="dma.wait", args=DmaArgs(flag=1)
        ),  # Wait to get these things
        # Do unnecessary loads
        Instruction(mnemonic="dma.load", args=DmaArgs(rd=3, base=0, size=1024, flag=0)),
        Instruction(
            mnemonic="dma.load.mxu0", args=DmaArgs(rd=0, base=1024, size=1024, flag=1)
        ),
        # Do matmul
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=2, vs2=1)),
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=2, vs2=1)),
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=2, vs2=1)),
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=2, vs2=1)),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=1)),  # Wait to finish loads
        Instruction(mnemonic="delay", args=ScalarArgs(imm=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0, torch.eye(32, 32, dtype=torch.float8_e4m3fn)),
        (1024, torch.eye(32, 32, dtype=torch.float8_e4m3fn)),
    ]
