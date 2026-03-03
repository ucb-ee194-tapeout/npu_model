from typing import List, Tuple
from ...software import (
    Instruction,
    Program,
)
import torch

from model_npu.workload.gemma_blocks import gemma_mlp_gate_up_forward


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
        # Load gate weight to WB mxu1 (matmul.mxu0 reads from mxu1)
        Instruction(
            mnemonic="dma.load.mxu1",
            args={
                "rd": 0,
                "base": GATE_WEIGHT_BASE,
                "size": GATE_PROJ_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 0,
            },
        ),
        Instruction(
            mnemonic="dma.load.mxu1",
            args={
                "rd": 1,
                "base": UP_WEIGHT_BASE,
                "size": UP_PROJ_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 1,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),
        # Load activation to MRF
        Instruction(
            mnemonic="dma.load",
            args={
                "rd": 0,
                "base": ACTIVATION_DATA_BASE,
                "size": ACTIVATION_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 0,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        # Gate projection: activation @ gate_weight -> MRF 1
        Instruction(mnemonic="matmul.mxu0", args={"rd": 1, "rs1": 0, "rs2": 0}),
        # Up projection: activation @ up_weight -> MRF 2
        Instruction(mnemonic="matmul.mxu0", args={"rd": 2, "rs1": 0, "rs2": 1}),
        # Simplified output: gate * up (GeGLU uses gate * silu(up))
        Instruction(mnemonic="vmul", args={"vrd": 6, "vs1": 1, "vs2": 2}),
        # Store result
        Instruction(
            mnemonic="dma.store",
            args={
                "rs1": 6,
                "base": OUTPUT_DATA_BASE,
                "size": 64 * 16 * torch.bfloat16.itemsize,
                "flag": 2,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 2}),
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
