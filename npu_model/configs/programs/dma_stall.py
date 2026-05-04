import torch

from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER

class DMAStallProgram(Program):
    """
    A simple program demonstrating DMA loads, stalling logic, and matrix multiplication
    updated for the latest npu_model ISA.
    """

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'dma_stall.S')

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (0x80000000, torch.eye(32, 32, dtype=torch.float8_e4m3fn)),
        (0x80000400, torch.eye(32, 32, dtype=torch.float8_e4m3fn)),
    ]
