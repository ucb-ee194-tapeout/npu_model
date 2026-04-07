from typing import List, Tuple, Any
from .instruction import Instruction
import torch


class Program:
    """
    A program is a sequence of instructions to be executed.
    """

    instructions: List[Instruction[Any]] = []
    memory_regions: List[Tuple[int, torch.Tensor]] = []

    def __len__(self) -> int:
        return len(self.instructions)

    def __getitem__(self, idx: int) -> Instruction[Any]:
        return self.instructions[idx]

    def get_instruction(self, pc: int) -> Instruction[Any]:
        """
        Get the instruction at `pc`.
        """
        return self.instructions[pc]

    def is_finished(self, pc: int) -> bool:
        """Check if program execution is complete."""
        return pc >= len(self.instructions)
    
    def assemble(self) -> list[int]:
        bytecode: list[int] = []
        for instr in self.instructions:
            bytecode.append(instr.assemble())
        return bytecode

class InstantiableProgram(Program):
    def __init__(self, instructions: list[Instruction[Any]]):
        self.instructions = instructions