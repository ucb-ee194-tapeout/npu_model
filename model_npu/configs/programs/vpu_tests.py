from typing import List, Tuple
from ...software import Instruction, Program
import torch


class VectorArithmeticProgram(Program):
    """
    Basic arithmetic correctness
    """

    instructions: List[Instruction] = [
        Instruction("dma.load.m", {"rd": 0, "base": 0, "size": 16 * 2, "flag": 0}),
        Instruction("dma.wait", {"flag": 0}),

        Instruction("vadd", {"vrd": 1, "vs1": 0, "vs2": 0}),
        Instruction("vsub", {"vrd": 2, "vs1": 1, "vs2": 0}),
        Instruction("vmul", {"vrd": 3, "vs1": 2, "vs2": 0}),

        Instruction("dma.store.m", {"rs1": 3, "base": 16 * 2, "size": 16 * 2, "flag": 1}),
        Instruction("dma.wait", {"flag": 1}),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0, torch.arange(16, dtype=torch.bfloat16)),
        (16 * 2, torch.zeros(16, dtype=torch.bfloat16)),
    ]


