from typing import List, Tuple
from .instruction import Instruction
import numpy as np


class Program:
    """
    A program is a sequence of instructions to be executed.
    """

    instructions: List[Instruction] = []
    memory_regions: List[Tuple[int, np.ndarray]] = []

    def __len__(self) -> int:
        return len(self.instructions)

    def __getitem__(self, idx: int) -> Instruction:
        return self.instructions[idx]

    def get_instruction(self, pc: int) -> Instruction:
        """
        Get the instruction at `pc`.
        """
        return self.instructions[pc]

    def is_finished(self, pc: int) -> bool:
        """Check if program execution is complete."""
        return pc >= len(self.instructions)
