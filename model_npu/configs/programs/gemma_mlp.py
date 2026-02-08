from typing import List, Tuple
from ...software import (
    Instruction,
    Program,
)
import numpy as np


class GemmaMlpProgram(Program):
    """
    Gemma MLP kernel program.
    """

    instructions: List[Instruction] = [
        # load gate_proj_weight tile
        Instruction(
            mnemonic="dma.load",
            args={"rd": 0, "base": 0, "size": 16 * 32 * 1, "flag": 0},
        ),
        # load up_proj_weight tile
        Instruction(
            mnemonic="dma.load",
            args={"rd": 1, "base": 0, "size": 16 * 32 * 1, "flag": 1},
        ),

        # set loop bound and counter
        Instruction(mnemonic="addi", args={"rd": 1, "rs1": 0, "imm": 8}),
        Instruction(mnemonic="addi", args={"rd": 2, "rs1": 0, "imm": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),

        # load x tile
        Instruction(
            mnemonic="dma.load",
            args={"rd": 0, "base": 0, "size": 64 * 32 * 1, "flag": 0},
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),

        # gate projection
        Instruction(mnemonic="matmul", args={"rd": 1, "rs1": 0, "rs2": 0}, delay=0),
        # up projection
        Instruction(mnemonic="matmul", args={"rd": 2, "rs1": 0, "rs2": 1}, delay=0),

        # loop
        Instruction(mnemonic="blt", args={"rs1": 2, "rs2": 1, "imm": -4}, delay=0),
        Instruction(mnemonic="addi", args={"rd": 2, "rs1": 2, "imm": 1}),
        Instruction(mnemonic="nop", args={}),

        Instruction(mnemonic="vlibroadcast", args={"rd": 4, "imm": 0.7978845608028654}),
        Instruction(mnemonic="vlibroadcast", args={"rd": 5, "imm": 0.044715}),
        # reset loop counter
        Instruction(mnemonic="addi", args={"rd": 2, "rs1": 0, "imm": 0}),

        # pow(x, 3)
        Instruction(mnemonic="vmul", args={"rd": 6, "rs1": 1, "rs2": 1}),
        Instruction(mnemonic="vmul", args={"rd": 6, "rs1": 6, "rs2": 1}),
        # 0.044715 * x_pow3
        Instruction(mnemonic="vmul", args={"rd": 6, "rs1": 5, "rs2": 6}),
        # x + x_pow3
        Instruction(mnemonic="vadd", args={"rd": 6, "rs1": 1, "rs2": 6}),
        # sqrt_constant * x
        Instruction(mnemonic="vmul", args={"rd": 6, "rs1": 4, "rs2": 6}),

        # loop
        Instruction(mnemonic="blt", args={"rs1": 2, "rs2": 1, "imm": -1}, delay=0),
        Instruction(mnemonic="addi", args={"rd": 2, "rs1": 2, "imm": 1}),
        Instruction(mnemonic="nop", args={}),

        Instruction(
            mnemonic="dma.store",
            args={"rs1": 0, "base": 0, "size": 64 * 16 * 1, "flag": 2},
            delay=15,
        ),  # stall for 15 cycles before dispatching me
        Instruction(mnemonic="dma.wait", args={"flag": 2}),
    ]

    memory_regions: List[Tuple[int, np.ndarray]] = [
        (0, np.ones((64, 16)).astype(np.float32)),
        (0, np.ones((64, 16)).astype(np.float32)),
        (0, np.ones((64, 16)).astype(np.float32)),
        (0, np.ones((64, 16)).astype(np.float32)),
    ]
