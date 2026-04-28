import torch
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER
from npu_model.workload.gemma_blocks import gemma_mlp_gate_up_forward

GATE_PROJ_WEIGHT_DATA = torch.ones((32, 32), dtype=torch.float8_e4m3fn)
UP_PROJ_WEIGHT_DATA = torch.ones((32, 32), dtype=torch.float8_e4m3fn)
ACTIVATION_DATA = torch.ones((32, 32), dtype=torch.float8_e4m3fn)

# DRAM memory layout (program-loaded)
DRAM_GATE_WEIGHT_BASE = 0x0000
DRAM_UP_WEIGHT_BASE = 0x0400  # 1024 bytes after gate
DRAM_ACTIVATION_BASE = 0x0800
DRAM_OUTPUT_BASE = 0x0C00

# VMEM scratchpad layout
VMEM_GATE_WEIGHT_BASE = 0x2000
VMEM_UP_WEIGHT_BASE = 0x2400
VMEM_ACTIVATION_BASE = 0x2800
VMEM_OUTPUT_BASE = 0x2C00


class GemmaMlpProgram(Program):
    """
    Gemma MLP kernel program (simplified).
    Gate and up projections, then elementwise gate*up (simplified GeGLU).
    """

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'gemma_mlp.S')

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_GATE_WEIGHT_BASE, GATE_PROJ_WEIGHT_DATA),
        (DRAM_UP_WEIGHT_BASE, UP_PROJ_WEIGHT_DATA),
        (DRAM_ACTIVATION_BASE, ACTIVATION_DATA),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUTPUT_BASE,
        gemma_mlp_gate_up_forward(
            ACTIVATION_DATA,
            GATE_PROJ_WEIGHT_DATA,
            UP_PROJ_WEIGHT_DATA,
            use_gelu=False,  # matches NPU: gate * up
        ).to(torch.bfloat16).contiguous(),
    )
