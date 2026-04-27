import torch
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER

# Memory layout (DRAM is program-loaded; VMEM is scratchpad accessed by vload/vstore)
DRAM_INPUT_BASE = 0x0000
DRAM_OUTPUT_BASE = 0x0800
VMEM_INPUT_BASE = 0x2000
VMEM_OUTPUT_BASE = 0x2800

# Keep the cube input range modest so BF16 stays numerically well-behaved.
INPUT = torch.linspace(-4.0, 4.0, steps=32 * 32, dtype=torch.bfloat16).reshape(32, 32)


def _bf16_arithmetic_reference(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.bfloat16)
    identity = ((x + x).to(torch.bfloat16) - x).to(torch.bfloat16)
    square = (identity * identity).to(torch.bfloat16)
    cube = (x * x * x).to(torch.bfloat16)
    return (square * cube).to(torch.bfloat16)


class VectorArithmeticProgram(Program):
    """
    Basic arithmetic correctness
    """

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'vpu_tests.S')

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_INPUT_BASE, INPUT),
    ]

    golden_result: list[tuple[int, torch.Tensor]] = [(
        DRAM_OUTPUT_BASE,
        _bf16_arithmetic_reference(INPUT),
    )]
