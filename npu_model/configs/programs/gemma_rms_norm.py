import torch
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER
from npu_model.workload.gemma_blocks import gemma_rms_norm_forward

INPUT_DATA = torch.randn(32, 32, dtype=torch.bfloat16)
EPS = 1e-6

# DRAM layout
DRAM_INPUT_BASE = 0x0000
DRAM_EPS_BASE = 0x0800
DRAM_OUTPUT_BASE = 0x1000

# VMEM layout
VMEM_INPUT_BASE = 0x2000
VMEM_EPS_BASE = 0x2800
VMEM_OUTPUT_BASE = 0x3000


# def _gemma_rms_norm_program_reference(
#     x: torch.Tensor, eps: float = EPS
# ) -> torch.Tensor:
#     state = ArchState(DefaultHardwareConfig().arch_state_config)
#     x_bf16 = x.to(torch.bfloat16).contiguous()
#     state.write_mrf_bf16_tile(0, x_bf16)
#     state.write_mrf_bf16_tile(2, torch.full_like(x_bf16, eps, dtype=torch.bfloat16))
#     state.write_mrf_bf16(8, torch.full((32, 16), x.shape[-1], dtype=torch.bfloat16))
#     state.write_mrf_bf16(9, torch.full((32, 16), x.shape[-1], dtype=torch.bfloat16))
#     vmul_bf16(state, VectorArgs(vd=4, vs1=0, vs2=0))
#     vredsum_row_bf16(state, VectorArgs(vd=6, vs1=4))
#     vrecip_bf16(state, VectorArgs(vd=10, vs1=8))
#     vmul_bf16(state, VectorArgs(vd=12, vs1=6, vs2=10))
#     vadd_bf16(state, VectorArgs(vd=14, vs1=12, vs2=2))
#     vsqrt_bf16(state, VectorArgs(vd=16, vs1=14))
#     vrecip_bf16(state, VectorArgs(vd=18, vs1=16))
#     vmul_bf16(state, VectorArgs(vd=20, vs1=0, vs2=18))
#     return state.read_mrf_bf16_tile(20).clone()



class GemmaRmsNormProgram(Program):
    """
    Gemma RMS norm program.
    RMS norm: x * rsqrt(mean(x^2) + eps).
    Row-wise mean via transpose + vreduce.sum (second-to-last dim) + vbroadcast.cols.
    """

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'gemma_rms_norm.S')

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_INPUT_BASE, INPUT_DATA),
        (DRAM_EPS_BASE, torch.full(INPUT_DATA.shape, EPS, dtype=torch.bfloat16)),
    ]

    # FIXME: Re-derive a standalone golden reference for the pair-register BF16
    # VPU path. The current kernel wiring is exercised by simulation, but the
    # previous float-side golden no longer matches the staged BF16 execution.
    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUTPUT_BASE,
        gemma_rms_norm_forward(INPUT_DATA).to(torch.bfloat16),
    )
