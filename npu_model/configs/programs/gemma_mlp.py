from typing import List, Tuple
from ...software import (
    Instruction,
    Program,
)
import torch

from npu_model.workload.gemma_blocks import gemma_mlp_gate_up_forward
from npu_model.configs.isa_definition import ScalarArgs, DmaArgs, MatrixArgs, VectorArgs


GATE_PROJ_WEIGHT_DATA = torch.ones((32, 16), dtype=torch.float8_e4m3fn)
UP_PROJ_WEIGHT_DATA = torch.ones((32, 16), dtype=torch.float8_e4m3fn)
ACTIVATION_DATA = torch.ones((64, 32), dtype=torch.float8_e4m3fn)

# Memory layout: gate weights, up weights, activation, output
GATE_WEIGHT_BASE = 0x0000
UP_WEIGHT_BASE = 0x0200  # 512 bytes after gate
ACTIVATION_DATA_BASE = 0x2000
OUTPUT_DATA_BASE = 0x3000


class GemmaMlpProgram(Program):
    """
    Gemma MLP kernel program (simplified).
    Gate and up projections, then elementwise gate*up (simplified GeGLU).
    """

    instructions: List[Instruction] = [
        Instruction(
            mnemonic="dma.load.mxu0",
            args=DmaArgs(
                rd=0,
                base=GATE_WEIGHT_BASE,
                size=GATE_PROJ_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                flag=0,
            ),
        ),
        Instruction(
            mnemonic="dma.load.mxu0",
            args=DmaArgs(
                rd=1,
                base=UP_WEIGHT_BASE,
                size=UP_PROJ_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                flag=1,
            ),
        ),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=0)),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=1)),
        Instruction(
            mnemonic="dma.load",
            args=DmaArgs(
                rd=0,
                base=ACTIVATION_DATA_BASE,
                size=ACTIVATION_DATA.numel() * torch.float8_e4m3fn.itemsize,
                flag=0,
            ),
        ),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=0)),
        Instruction(mnemonic="matmul.mxu0", args=MatrixArgs(mrd=1, mrs1=0, mrs2=0)),
        Instruction(mnemonic="matmul.mxu0", args=MatrixArgs(mrd=2, mrs1=0, mrs2=1)),
        Instruction(mnemonic="vmul", args=VectorArgs(vrd=6, vrs1=1, vrs2=2)),
        Instruction(
            mnemonic="dma.store",
            args=DmaArgs(
                rs1=6,
                base=OUTPUT_DATA_BASE,
                size=64 * 16 * torch.bfloat16.itemsize,
                flag=2,
            ),
        ),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=2)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (GATE_WEIGHT_BASE, GATE_PROJ_WEIGHT_DATA),
        (UP_WEIGHT_BASE, UP_PROJ_WEIGHT_DATA),
        (ACTIVATION_DATA_BASE, ACTIVATION_DATA),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        OUTPUT_DATA_BASE,
        gemma_mlp_gate_up_forward(
            ACTIVATION_DATA,
            GATE_PROJ_WEIGHT_DATA,
            UP_PROJ_WEIGHT_DATA,
            use_gelu=False,  # matches NPU: gate * up
        ).to(torch.bfloat16),
    )
