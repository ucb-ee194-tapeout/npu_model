from typing import List, Any
from npu_model.isa import *
from ...software import (
    Instruction,
    Program,
)


class AddiProgram(Program):
    """
    A simple addi program with a branch and a matmul.
    """

    instructions: List[Instruction[Any]] = [
        Instruction(mnemonic="addi", args=ScalarArgs(rd=1, rs1=1, imm=0)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=2, rs1=2, imm=8)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=1, rs1=1, imm=1)),
        Instruction(mnemonic="blt", args=ScalarArgs(rs1=1, rs2=2, imm=-1)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=0)),
        Instruction(mnemonic="vmatmul.mxu1", args=MatrixArgs(vd=1, vs1=1, vs2=1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=4, rs1=4, imm=1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=5, rs1=5, imm=1)),
        Instruction(mnemonic="delay", args=ScalarArgs()),
    ]