import torch
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER

# Constants for memory layout
DRAM_ACTIVATION_BASE = 0x80000000
DRAM_WEIGHT_BASE = 0x80000400
DRAM_OUTPUT_BASE = 0x80000800
VMEM_ACTIVATION_BASE = 0x20002000
VMEM_WEIGHT_BASE = 0x20002400
VMEM_OUTPUT_BASE = 0x20002800

# Mock data for matmul verification
ACTIVATION_DATA = torch.eye(32, 32, dtype=torch.float8_e4m3fn)
WEIGHT_DATA = (2 * torch.eye(32, 32, dtype=torch.float32)).to(torch.float8_e4m3fn)
MATMUL_RESULT = (ACTIVATION_DATA.to(torch.float32) @ WEIGHT_DATA.to(torch.float32)).to(
    torch.bfloat16
)


class MatmulProgram(Program):
    """
    Rewritten Matmul test using structured Args dataclasses.
    """

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'matmul.S')

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_ACTIVATION_BASE, ACTIVATION_DATA),
        (DRAM_WEIGHT_BASE, WEIGHT_DATA),
    ]

    golden_result: list[tuple[int, torch.Tensor]] = [(
        DRAM_OUTPUT_BASE,
        torch.cat((MATMUL_RESULT[:, :16], MATMUL_RESULT[:, 16:]), dim=0),
    )]