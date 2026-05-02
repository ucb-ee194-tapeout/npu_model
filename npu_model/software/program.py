import torch
from pathlib import Path

from .instruction import Instruction

ASM_FOLDER = Path("./npu_model/configs/programs/asm/")

class Program:
    """
    A program is a sequence of instructions to be executed.
    """

    instructions: list[Instruction] = []
    """The set of instructions to execute in the program, in Python IR."""
    memory_regions: list[tuple[int, torch.Tensor]] = []
    """The memory to load into DRAM, starting at dram base (0)"""
    golden_result: list[tuple[int, torch.Tensor]] = []
    """The expected value after the program has completed."""
    timeout: int | None = 10000
    """The number of cycles to run before timing out. Does nothing in the perf model, just for assembly into a .C file"""
    
    def __len__(self) -> int:
        return len(self.instructions)

    def __getitem__(self, idx: int) -> Instruction:
        return self.instructions[idx]

    def get_instruction(self, pc: int) -> Instruction:
        """
        Get the instruction at byte address `pc`.
        """
        return self.instructions[pc // 4]

    def is_finished(self, pc: int) -> bool:
        """Check if program execution is complete."""
        return pc >= len(self.instructions) * 4

    def assemble(self) -> list[int]:
        bytecode: list[int] = []
        for instr in self.instructions:
            bytecode.append(instr.to_bytecode())
        return bytecode


class InstantiableProgram(Program):
    def __init__(self, instructions: list[Instruction], memory_regions: list[tuple[int, torch.Tensor]] = [], golden_result: list[tuple[int, torch.Tensor]] = [], timeout: int | None = 10000):
        self.instructions = instructions
        self.memory_regions = memory_regions
        self.golden_result = golden_result
        self.timeout = timeout
