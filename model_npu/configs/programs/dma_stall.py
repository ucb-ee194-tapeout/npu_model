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
        Instruction("dma.load.m", {"rd": 0, "base": 0, "size": 16 * 2, "flag": 0}, delay=5),
        Instruction("dma.load.m", {"rd": 1, "base": 16 * 2, "size": 16 * 2, "flag": 1}, delay=5),
        Instruction("dma.load.m", {"rd": 2, "base": 16 * 3, "size": 16 * 2, "flag": 2}, delay=5),
        # Instruction("dma.wait", {"flag": 2}),
        Instruction(mnemonic="addi", args={"rd": 5, "rs1": 0, "imm": 10}),
        Instruction(mnemonic="dma.wait", args={"flag": 2}),
        Instruction(
            mnemonic="dma.store.m",
            args={"rs1": 3, "base": 16 * 4, "size": 32, "flag": 1},
            delay=15,
        ),  # stall for 15 cycles before dispatching me
        Instruction(mnemonic="dma.wait", args={"flag": 1}),

    ]
    
    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0, torch.ones((16, 16), dtype=torch.float32)),
        (32, torch.ones((16, 16), dtype=torch.float32)),
        (48, torch.ones((16, 16), dtype=torch.float32)),
        (16 * 4, torch.ones((16, 16), dtype=torch.float32)),
    ]
