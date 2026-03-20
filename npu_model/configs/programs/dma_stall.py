from typing import List, Tuple
from ...software import (
    Instruction,
    Program,
)
import torch


class DMAStallProgram(Program):
    """
    A simple addi program with a branch and a matmul.
    """

    instructions: List[Instruction] = [
        # Load things w/ Matmul
        Instruction(
            mnemonic="dma.load", args={"rd": 2, "base": 0, "size": 1024, "flag": 0}
        ),
        Instruction(
            mnemonic="dma.load.mxu0", args={"rd": 1, "base": 1024, "size": 1024, "flag": 1}
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 1}), # Wait to get these things
        
        # Do unnecessary loads
        Instruction(
            mnemonic="dma.load", args={"rd": 3, "base": 0, "size": 1024, "flag": 0}
        ),
        Instruction(
            mnemonic="dma.load.mxu0", args={"rd": 0, "base": 1024, "size": 1024, "flag": 1}
        ),

        # Do matmul
        Instruction(mnemonic="matmul.mxu0", args={"rd": 0, "rs1": 2, "rs2": 1}),
        Instruction(mnemonic="matmul.mxu0", args={"rd": 0, "rs1": 2, "rs2": 1}),
        Instruction(mnemonic="matmul.mxu0", args={"rd": 0, "rs1": 2, "rs2": 1}),
        Instruction(mnemonic="matmul.mxu0", args={"rd": 0, "rs1": 2, "rs2": 1}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}), # Wait to finish loads
        Instruction(mnemonic="delay", args={"imm": 0}),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0, torch.eye(32, 32, dtype=torch.float8_e4m3fn)),
        (1024, torch.eye(32, 32, dtype=torch.float8_e4m3fn)),
    ]
