from typing import List, Tuple
from npu_model.isa import ScalarArgs, VectorArgs, MatrixArgs, DmaArgs
from ...software import Instruction, Program
import torch


class MatmulProgram(Program):
    """
    matmul test
    """

    instructions: List[Instruction] = [
        Instruction(
            mnemonic="dma.load", args=DmaArgs(rd=2, base=0, size=2048, flag=0)
        ),  # a full 64x16 matrix of bf16s (0-2048)
        Instruction(mnemonic="dma.wait", args=DmaArgs(rd=0, base=0, size=0, flag=0)),
        Instruction(
            mnemonic="dma.load.mxu0",
            args=DmaArgs(rd=1, base=2048, size=512, flag=0),
        ),  # a full 64x16 matrix of bf16s (ones)
        Instruction(mnemonic="dma.wait", args=DmaArgs(rd=0, base=0, size=0, flag=0)),
        Instruction(mnemonic="matmul.mxu0", args=MatrixArgs(mrd=0, mrs1=2, mrs2=1)),
        Instruction(mnemonic="matmul.mxu0", args=MatrixArgs(mrd=0, mrs1=2, mrs2=1)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        # A = 64x32 matrix with increasing values
        (0, (torch.eye(64, 32, dtype=torch.float8_e4m3fn))),
        # B = 32x16 matrix (identity-like: first 16 columns of 32x32 identity)
        # So result is (64x32) @ (32x16) -> (64x16)
        (2048, (torch.eye(32, 16, dtype=torch.float8_e4m3fn))),
    ]
