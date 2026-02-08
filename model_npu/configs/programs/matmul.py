from typing import List, Tuple
from ...software import (
    Instruction,
    Program,
)
import torch


class MatmulProgram(Program):
    """
    A simple matmul program.
    """

    instructions: List[Instruction] = [
        Instruction(mnemonic="addi", args={"rd": 1, "rs1": 1, "imm": 3}),
        Instruction(
            mnemonic="dma.load",
            args={"rd": 1, "base": 0, "size": 16 * 16 * 4, "flag": 0},
        ),
        Instruction(
            mnemonic="dma.loadw",
            args={"rd": 0, "base": 0, "size": 16 * 16 * 4, "flag": 1},
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="matmul", args={"rd": 0, "rs1": 1, "rs2": 0}, delay=0),
        Instruction(
            mnemonic="dma.load",
            args={"rd": 2, "base": 0, "size": 16 * 16 * 4, "flag": 0},
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),
        Instruction(mnemonic="matmul", args={"rd": 3, "rs1": 1, "rs2": 0}, delay=0),
        Instruction(
            mnemonic="dma.store",
            args={"rs1": 0, "base": 16 * 16 * 4, "size": 16 * 16 * 4, "flag": 2},
            delay=15,
        ),  # stall for 15 cycles before dispatching me
        Instruction(mnemonic="dma.wait", args={"flag": 2}),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0, torch.ones((16, 16), dtype=torch.float32)),
        (16 * 16 * 4, torch.ones((16, 16), dtype=torch.float32)),
    ]
