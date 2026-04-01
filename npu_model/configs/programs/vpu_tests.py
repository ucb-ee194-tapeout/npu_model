from typing import List, Tuple
from ...software import Instruction, Program
import torch
from npu_model.isa import (

    VectorArgs,
)


class VectorArithmeticProgram(Program):
    """
    Basic arithmetic correctness
    """

    instructions: List[Instruction] = [
        Instruction("vload", args=VectorArgs(vd=0, rs1=0, offset=0)),
        Instruction("vadd", args=VectorArgs(vrd=1, vs1=0, vs2=0)),
        Instruction("vsub", args=VectorArgs(vrd=2, vs1=1, vs2=0)),
        Instruction("vmul", args=VectorArgs(vrd=3, vs1=2, vs2=0)),
        Instruction("vstore", args=VectorArgs(vd=3, rs1=0, offset=1)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0, torch.arange(16, dtype=torch.bfloat16)),
        (16 * 2, torch.zeros(16, dtype=torch.bfloat16)),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        16 * 2,
        (torch.arange(16, dtype=torch.bfloat16) ** 2),
    )
