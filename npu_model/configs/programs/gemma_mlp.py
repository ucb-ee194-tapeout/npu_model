from typing import List, Tuple
from ...software import (
    Instruction,
    Program,
)
import torch

from npu_model.workload.gemma_blocks import gemma_mlp_gate_up_forward


GATE_PROJ_WEIGHT_DATA = torch.ones((32, 32), dtype=torch.float8_e4m3fn)
UP_PROJ_WEIGHT_DATA = torch.ones((32, 32), dtype=torch.float8_e4m3fn)
ACTIVATION_DATA = torch.ones((32, 32), dtype=torch.float8_e4m3fn)

# Memory layout: gate weights, up weights, activation, output
GATE_WEIGHT_BASE = 0x0000
UP_WEIGHT_BASE = 0x0400  # 1024 bytes after gate
ACTIVATION_DATA_BASE = 0x2000
OUTPUT_DATA_BASE = 0x3000


class GemmaMlpProgram(Program):
    """
    Gemma MLP kernel program (simplified).
    Gate and up projections, then elementwise gate*up (simplified GeGLU).
    """

    instructions: List[Instruction] = [
        # Load gate and up weights into MXU0's weight buffer (matmul.mxu0 uses mxu0's WB)
        Instruction(
            mnemonic="dma.load.mxu0",
            args={
                "rd": 0,
                "base": GATE_WEIGHT_BASE,
                "size": GATE_PROJ_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 0,
            },
        ),
        Instruction(
            mnemonic="dma.load.mxu0",
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
        # Gate projection: activation @ gate_weight -> MRF {2, 3}
        Instruction(mnemonic="matmul.mxu0", args={"rd": 2, "rs1": 0, "rs2": 0}),
        # Up projection: activation @ up_weight -> MRF {4, 5}
        Instruction(mnemonic="matmul.mxu0", args={"rd": 4, "rs1": 0, "rs2": 1}),
        Instruction(mnemonic="delay", args={"imm": 0}, delay=32),
        # Simplified output: gate * up (GeGLU uses gate * silu(up)), half-tile at a time.
        Instruction(mnemonic="vmul", args={"vrd": 6, "vs1": 2, "vs2": 4}),
        Instruction(mnemonic="vmul", args={"vrd": 7, "vs1": 3, "vs2": 5}),
        Instruction(
            mnemonic="dma.store",
            args={
                "rs1": 6,
                "base": OUTPUT_DATA_BASE,
                "size": 32 * 16 * torch.bfloat16.itemsize,
                "flag": 0,
            },
        ),
        Instruction(
            mnemonic="dma.store",
            args={
                "rs1": 7,
                "base": OUTPUT_DATA_BASE + 32 * 16 * torch.bfloat16.itemsize,
                "size": 32 * 16 * torch.bfloat16.itemsize,
                "flag": 1,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),
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
