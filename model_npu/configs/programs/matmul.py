from typing import List, Tuple
from ...software import (
    Instruction,
    Program,
)
import numpy as np


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
            mnemonic="dma.load",
            args={"rd": 2, "base": 0, "size": 16 * 16 * 4, "flag": 1},
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="matmul", args={"rd": 0, "rs1": 1, "rs2": 2}, delay=0),
        Instruction(
            mnemonic="dma.load",
            args={"rd": 4, "base": 0, "size": 16 * 16 * 4, "flag": 0},
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),
        Instruction(mnemonic="matmul", args={"rd": 3, "rs1": 1, "rs2": 4}, delay=0),
        Instruction(
            mnemonic="dma.store",
            args={"rs1": 0, "base": 16 * 16 * 4, "size": 16 * 16 * 4, "flag": 2},
            delay=15,
        ),  # stall for 15 cycles before dispatching me
        Instruction(mnemonic="dma.wait", args={"flag": 2}),
    ]

    memory_regions: List[Tuple[int, np.ndarray]] = [
        (0, np.ones((16, 16)).astype(np.float32)),
        (16 * 16 * 4, np.ones((16, 16)).astype(np.float32)),
    ]
