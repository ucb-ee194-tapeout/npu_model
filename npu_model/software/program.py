import torch
from pathlib import Path

from .instruction import Instruction

ASM_FOLDER = Path("./npu_model/configs/programs/asm/")

class Program:
    """
    A program is a sequence of instructions to be executed.
    """

    instructions: list[Instruction] = []
    memory_regions: list[tuple[int, torch.Tensor]] = []
    """(offset, bytes) preloaded at ``dram_base + offset`` before the program runs."""
    dram_base: int = 1 << 32
    """Physical address that ``memory_regions`` and ``golden_result`` offsets are
    relative to. The supplied workloads program ``dma.config`` with base 1, so
    their 32-bit DMA operands address DRAM from 4 GiB: inside the window the
    RTL simulation maps (64 GiB at 0x8000_0000) and clear of the host program."""

    def __len__(self) -> int:
        return len(self.instructions)

    def __getitem__(self, idx: int) -> Instruction:
        return self.instructions[idx]

    def get_instruction(self, pc: int) -> Instruction:
        """
        Get the instruction at RTL word-index program counter `pc`.
        """
        return self.instructions[pc]

    def is_finished(self, pc: int) -> bool:
        """Check if program execution is complete."""
        return pc >= len(self.instructions)

    def assemble(self) -> list[int]:
        bytecode: list[int] = []
        for instr in self.instructions:
            bytecode.append(instr.to_bytecode())
        return bytecode


class InstantiableProgram(Program):
    def __init__(self, instructions: list[Instruction]):
        self.instructions = instructions
