import torch

from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER

class AddiProgram(Program):
    """
    A simple addi program with a branch and a matmul.
    """

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'addi.S')

    memory_regions: list[tuple[int, torch.Tensor]] = []
