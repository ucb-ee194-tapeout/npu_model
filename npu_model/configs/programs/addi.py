from typing import List
from model_npu.isa import ScalarArgs, VectorArgs, MatrixArgs, DmaArgs
from ...software import (
    Instruction,
    Program,
)


class AddiProgram(Program):
    """
    A simple addi program with a branch and a matmul.
    """

    instructions: List[Instruction] = [
        Instruction(mnemonic="addi", args=ScalarArgs(rd=1, rs1=1, imm=0)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=2, rs1=2, imm=8)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=1, rs1=1, imm=1)),
        Instruction(mnemonic="blt", args=ScalarArgs(rs1=1, rs2=2, imm=-1)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=0)),
        Instruction(mnemonic="matmul.mxu1", args=MatrixArgs(mrd=1, mrs1=1, mrs2=1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=4, rs1=4, imm=1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=5, rs1=5, imm=1)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=0)),
    ]
