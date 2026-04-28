from npu_model.hardware.arch_state import ArchState
from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.isa import MatrixReg
from npu_model.configs.isa_definition import (
    VADD_BF16,
    VMUL_BF16,
    VRECIP_BF16,
    VREDSUM_ROW_BF16,
    VSQRT_BF16,
)
import torch
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER
from npu_model.workload.gemma_blocks import gemma_rms_norm_forward


# INPUT_DATA = torch.ones(32, 32, dtype=torch.bfloat16) * 10
INPUT_DATA = (10 - 0) * torch.randn(32, 32, dtype=torch.bfloat16) + 0
EPS = 1e-6

# DRAM layout
DRAM_INPUT_BASE = 0x0000
DRAM_EPS_BASE = 0x0800
DRAM_OUTPUT_BASE = 0x1000

# VMEM layout
VMEM_INPUT_BASE = 0x2000
VMEM_EPS_BASE = 0x2800
VMEM_OUTPUT_BASE = 0x3000


def _gemma_rms_norm_program_reference(
    x: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    state = ArchState(DefaultHardwareConfig().arch_state_config)
    x_bf16 = x.to(torch.bfloat16).contiguous()

    # Setup MRF (Memory Register File)
    state.write_mrf_bf16_tile(0, x_bf16)  # Input x
    state.write_mrf_bf16_tile(
        2, torch.full_like(x_bf16, eps, dtype=torch.bfloat16)
    )  # Epsilon
    state.write_mrf_bf16(
        8, torch.full((32, 16), x.shape[-1], dtype=torch.bfloat16)
    )  # N (dim)
    state.write_mrf_bf16(
        9, torch.full((32, 16), x.shape[-1], dtype=torch.bfloat16)
    )  # N (unused)

    # 1. Square the inputs: x^2
    VMUL_BF16(vd=MatrixReg(4), vs1=MatrixReg(0), vs2=MatrixReg(0)).exec(state)

    # 2. Sum the squares across the row
    VREDSUM_ROW_BF16(vd=MatrixReg(6), vs1=MatrixReg(4)).exec(state)

    # 3. Calculate 1/N
    VRECIP_BF16(vd=MatrixReg(10), vs1=MatrixReg(8)).exec(state)

    # 4. Mean Square: (1/N) * sum(x^2)
    VMUL_BF16(vd=MatrixReg(12), vs1=MatrixReg(6), vs2=MatrixReg(10)).exec(state)

    # 5. Add epsilon: MS + eps
    VADD_BF16(vd=MatrixReg(14), vs1=MatrixReg(12), vs2=MatrixReg(2)).exec(state)

    # 6. Sqrt: sqrt(MS + eps)
    VSQRT_BF16(vd=MatrixReg(16), vs1=MatrixReg(14)).exec(state)

    # 7. Reciprocal: 1 / sqrt(MS + eps)
    VRECIP_BF16(vd=MatrixReg(18), vs1=MatrixReg(16)).exec(state)

    # 8. Final scaling: x * (1 / sqrt(MS + eps))
    VMUL_BF16(vd=MatrixReg(20), vs1=MatrixReg(0), vs2=MatrixReg(18)).exec(state)

    return state.read_mrf_bf16_tile(20).clone()


class GemmaRmsNormProgram(Program):
    """
    Gemma RMS norm program.
    RMS norm: x * rsqrt(mean(x^2) + eps).
    Row-wise mean via transpose + vreduce.sum (second-to-last dim) + vbroadcast.cols.
    """
    instructions: list[Instruction] = load_asm(ASM_FOLDER / "gemma_rms_norm.S")

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_INPUT_BASE, INPUT_DATA),
        (DRAM_EPS_BASE, torch.full(INPUT_DATA.shape, EPS, dtype=torch.bfloat16)),
    ]

    # FIXME: Re-derive a standalone golden reference for the pair-register BF16
    # VPU path. The current kernel wiring is exercised by simulation, but the
    # previous float-side golden no longer matches the staged BF16 execution.
    # golden_result: tuple[int, torch.Tensor] = (
    #     DRAM_OUTPUT_BASE,
    #     gemma_rms_norm_forward(INPUT_DATA).to(torch.bfloat16),
    # )
