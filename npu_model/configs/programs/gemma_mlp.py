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

    instructions: List[Instruction] = [
        # x1..x4: VMEM bases (use LUI+ADDI so immediates stay 12-bit clean)
        # 0x2000
        Instruction(mnemonic="lui", args=ScalarArgs(rd=1, imm=0x2)),
        # 0x2400
        Instruction(mnemonic="lui", args=ScalarArgs(rd=2, imm=0x2)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=2, rs1=2, imm=0x400)),
        # 0x2800 = 0x3000 - 0x800
        Instruction(mnemonic="lui", args=ScalarArgs(rd=3, imm=0x3)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=3, rs1=3, imm=-2048)),
        # 0x2C00 = 0x3000 - 0x400
        Instruction(mnemonic="lui", args=ScalarArgs(rd=4, imm=0x3)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=4, rs1=4, imm=-1024)),
        # x5..x8: DRAM bases
        Instruction(
            mnemonic="addi", args=ScalarArgs(rd=5, rs1=0, imm=DRAM_GATE_WEIGHT_BASE)
        ),
        Instruction(
            mnemonic="addi", args=ScalarArgs(rd=6, rs1=0, imm=DRAM_UP_WEIGHT_BASE)
        ),
        # DRAM_ACTIVATION_BASE = 0x0800 = 0x1000 - 0x800
        Instruction(mnemonic="lui", args=ScalarArgs(rd=7, imm=0x1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=7, rs1=7, imm=-2048)),
        # DRAM_OUTPUT_BASE = 0x0C00 does not fit in signed 12-bit addi; use LUI/ADDI.
        Instruction(mnemonic="lui", args=ScalarArgs(rd=8, imm=0x1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=8, rs1=8, imm=-1024)),
        # x9: byte length for fp8 tile (32*32*1 = 1024)
        Instruction(mnemonic="addi", args=ScalarArgs(rd=9, rs1=0, imm=1024)),
        # x10: byte length for bf16 tile (32*32*2 = 2048)
        Instruction(mnemonic="lui", args=ScalarArgs(rd=10, imm=0x1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=10, rs1=10, imm=-2048)),
        # DRAM -> VMEM
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=1, rs1=5, rs2=9, channel=0)
        ),
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=2, rs1=6, rs2=9, channel=1)
        ),
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=3, rs1=7, rs2=9, channel=2)
        ),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=2)),
        # VMEM -> MRF (weights + activation)
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),  # gate W (fp8)
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=1, rs1=2, imm12=0)),  # up W (fp8)
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=2, rs1=3, imm12=0)),  # act (fp8)
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),

        # Push weights to MXU0 WB slots 0 and 1
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=0, vs1=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=32)),
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=1, vs1=1)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=32)),
        # --- PHASE 3: Matrix Multiplications ---
        # Gate projection: activation @ gate_weight -> Acc/MRF
        # Note: Using MatrixArgs for matmul
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=2, vs2=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=32)),
        Instruction(
            mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=4, vs1=0)
        ),  # gate -> mrf4+5
        Instruction(mnemonic="delay", args=ScalarArgs(imm=32)),
        # Up projection: activation @ up_weight -> Acc/MRF
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=2, vs2=1)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=32)),
        Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=6, vs1=0)),  # up -> mrf6+7
        Instruction(mnemonic="delay", args=ScalarArgs(imm=32)),
        # --- PHASE 4: Element-wise Multiplication (GeGLU Simplified) ---
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=8, vs1=4, vs2=6)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=66)),
        # --- PHASE 5: Store Results ---
        Instruction(mnemonic="vstore", args=VectorArgs(vd=8, rs1=4, imm12=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=9, rs1=4, imm12=32)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=34)),
        # VMEM -> DRAM
        Instruction(
            mnemonic="dma.store.ch<N>", args=DmaArgs(rd=8, rs1=4, rs2=10, channel=0)
        ),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
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
