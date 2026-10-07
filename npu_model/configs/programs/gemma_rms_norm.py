import torch
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER
from npu_model.workload.gemma_blocks import gemma_rms_norm_forward


INPUT_DATA = 10 * torch.randn(32, 32, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(0))
EPS = 1e-6

# DRAM layout
DRAM_INPUT_BASE = 0x0000
DRAM_EPS_BASE = 0x0800
DRAM_OUTPUT_BASE = 0x1000

# VMEM layout
VMEM_INPUT_BASE = 0x2000
VMEM_EPS_BASE = 0x2800
VMEM_OUTPUT_BASE = 0x3000


def _column_halves(tile: torch.Tensor) -> torch.Tensor:
    """DMA image of a BF16 pair: m[vd] holds columns 0-15, m[vd + 1] 16-31."""
    return torch.cat((tile[:, :16], tile[:, 16:]))


class GemmaRmsNormProgram(Program):
    """
    Gemma RMS norm program.
    RMS norm: x * rsqrt(mean(x^2) + eps).
    Row sums via vredsum.row.bf16 over the (m0, m1) pair.
    """
    instructions: list[Instruction] = load_asm(ASM_FOLDER / "gemma_rms_norm.S")

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_INPUT_BASE, _column_halves(INPUT_DATA)),
        (DRAM_EPS_BASE, torch.full(INPUT_DATA.shape, EPS, dtype=torch.bfloat16)),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUTPUT_BASE,
        _column_halves(gemma_rms_norm_forward(INPUT_DATA, EPS).to(torch.bfloat16)),
    )
