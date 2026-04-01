from typing import List, Tuple
from ...software import (
    Instruction,
    Program,
)
import torch
from npu_model.isa import DmaArgs, MatrixArgs, VectorArgs, ScalarArgs
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
        # --- PHASE 1: Load Weights into MXU0 Weight Buffer ---
        Instruction(
            mnemonic="dma.load.mxu0",
            args=DmaArgs(
                rd=0,
                base=GATE_WEIGHT_BASE,
                size=GATE_PROJ_WEIGHT_DATA.numel() * 1,  # fp8 = 1 byte
                flag=0,
            ),
        ),
        Instruction(
            mnemonic="dma.load.mxu0",
            args=DmaArgs(
                rd=1, base=UP_WEIGHT_BASE, size=UP_PROJ_WEIGHT_DATA.numel() * 1, flag=1
            ),
        ),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=0)),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=1)),
        # --- PHASE 2: Load Activation to MRF ---
        Instruction(
            mnemonic="dma.load",
            args=DmaArgs(
                rd=0,
                base=ACTIVATION_DATA_BASE,
                size=ACTIVATION_DATA.numel() * 1,
                flag=0,
            ),
        ),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=0)),
        # --- PHASE 3: Matrix Multiplications ---
        # Gate projection: activation @ gate_weight -> Acc/MRF
        # Note: Using MatrixArgs for matmul
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=0, vs2=0)),
        Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=2, vs1=0)),
        # Up projection: activation @ up_weight -> Acc/MRF
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=0, vs2=1)),
        Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=4, vs1=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=0), delay=32),
        # --- PHASE 4: Element-wise Multiplication (GeGLU Simplified) ---
        # Using VectorArgs: corrected 'vrd' to 'vd'
        Instruction(mnemonic="vmul", args=VectorArgs(vd=6, vs1=2, vs2=4)),
        Instruction(mnemonic="vmul", args=VectorArgs(vd=7, vs1=3, vs2=5)),
        # --- PHASE 5: Store Results ---
        Instruction(
            mnemonic="dma.store",
            args=DmaArgs(
                rs1=6, base=OUTPUT_DATA_BASE, size=32 * 16 * 2, flag=0  # bf16 = 2 bytes
            ),
        ),
        Instruction(
            mnemonic="dma.store",
            args=DmaArgs(
                rs1=7, base=OUTPUT_DATA_BASE + (32 * 16 * 2), size=32 * 16 * 2, flag=1
            ),
        ),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=0)),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=1)),
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
