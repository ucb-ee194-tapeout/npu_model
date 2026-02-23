from typing import List
from ...software import (
    Instruction,
    Program,
)


class AddiProgram(Program):
    """
    A simple addi program with a branch and a matmul.
    """

    instructions: List[Instruction] = [
        Instruction(mnemonic="addi", args={"rd": 2, "rs1": 0, "imm": 0}),
        Instruction(mnemonic="addi", args={"rd": 1, "rs1": 1, "imm": 0}),
        Instruction(mnemonic="addi", args={"rd": 2, "rs1": 2, "imm": 8}),
        Instruction(mnemonic="addi", args={"rd": 1, "rs1": 1, "imm": 1}),
        Instruction(mnemonic="blt", args={"rs1": 1, "rs2": 2, "imm": -1}),
        Instruction(mnemonic="matmul.mxu1", args={"rd": 1, "rs1": 1, "rs2": 1}),
        # Instruction(mnemonic="nop", args={}),
        Instruction(mnemonic="addi", args={"rd": 4, "rs1": 4, "imm": 1}),
        Instruction(mnemonic="addi", args={"rd": 5, "rs1": 5, "imm": 1}),
        # Instruction(mnemonic="nop", args={}),
    ]
