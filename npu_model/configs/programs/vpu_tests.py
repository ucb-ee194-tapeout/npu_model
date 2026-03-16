from typing import List, Tuple
from ...software import Instruction, Program
import torch
from npu_model.isa import VectorArgs, DmaArgs


class VectorArithmeticProgram(Program):
    """
    Basic arithmetic correctness
    """

    instructions: List[Instruction] = [
        Instruction("dma.load", DmaArgs(rd=0, base=0, size=16 * 2, flag=0)),
        Instruction("dma.wait", DmaArgs(flag=0)),
        Instruction("vadd", VectorArgs(vrd=1, vrs1=0, vrs2=0)),
        Instruction("vsub", VectorArgs(vrd=2, vrs1=1, vrs2=0)),
        Instruction("vmul", VectorArgs(vrd=3, vrs1=2, vrs2=0)),
        Instruction("dma.store", DmaArgs(rs1=3, base=16 * 2, size=16 * 2, flag=1)),
        Instruction("dma.wait", DmaArgs(flag=1)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0, torch.arange(16, dtype=torch.bfloat16)),
        (16 * 2, torch.zeros(16, dtype=torch.bfloat16)),
    ]
