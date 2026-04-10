from typing import List, Tuple, Any
from ...software import Instruction, Program
import torch
from ...workload.gemma_blocks import gemma_rms_norm_forward
from npu_model.isa import DmaArgs, MatrixArgs, VectorArgs, ScalarArgs


# Input shape matches one BF16 tensor register: 32 rows x 16 columns.
INPUT_DATA = torch.randn(32, 16, dtype=torch.bfloat16)
ROW_SIZE = INPUT_DATA.shape[-1]
EPS = 1e-6
# DRAM layout
DRAM_INPUT_BASE = 0x0000
DRAM_EPS_BASE = 0x0400
DRAM_OUTPUT_BASE = 0x0800

# VMEM layout
VMEM_INPUT_BASE = 0x2000
VMEM_EPS_BASE = 0x2400
VMEM_OUTPUT_BASE = 0x2800


class GemmaRmsNormProgram(Program):
    """
    Gemma RMS norm program.
    RMS norm: x * rsqrt(mean(x^2) + eps).
    Row-wise mean via transpose + vreduce.sum (second-to-last dim) + vbroadcast.cols.
    """

    instructions: List[Instruction[Any]] = [
        # VMEM bases (use LUI+ADDI so immediates stay 12-bit clean)
        # 0x2000
        Instruction(mnemonic="lui", args=ScalarArgs(rd=1, imm=0x2)),
        # 0x2400
        Instruction(mnemonic="lui", args=ScalarArgs(rd=2, imm=0x2)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=2, rs1=2, imm=0x400)),
        # 0x2800 = 0x3000 - 0x800
        Instruction(mnemonic="lui", args=ScalarArgs(rd=3, imm=0x3)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=3, rs1=3, imm=-2048)),
        # DRAM bases
        Instruction(mnemonic="addi", args=ScalarArgs(rd=4, rs1=0, imm=DRAM_INPUT_BASE)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=5, rs1=0, imm=DRAM_EPS_BASE)),
        # DRAM_OUTPUT_BASE = 0x0800 = 0x1000 - 0x800
        Instruction(mnemonic="lui", args=ScalarArgs(rd=6, imm=0x1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=6, rs1=6, imm=-2048)),
        # byte length for bf16 tile
        Instruction(mnemonic="addi", args=ScalarArgs(rd=7, rs1=0, imm=1024)),
        # DRAM -> VMEM
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=1, rs1=4, rs2=7, channel=0)
        ),
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=2, rs1=5, rs2=7, channel=1)
        ),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
        # VMEM -> MRF
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),  # x
        Instruction(mnemonic="vload", args=VectorArgs(vd=1, rs1=2, imm12=0)),  # eps
        Instruction(mnemonic="delay", args=ScalarArgs(imm=16)),
        # x_sq = x * x
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=2, vs1=0, vs2=0)),
        # sum_sq over columns, broadcast back across each row
        Instruction(mnemonic="vredsum.row.bf16", args=VectorArgs(vd=3, vs1=2)),
        # mean_sq = sum_sq * (1/ROW_SIZE)
        Instruction(mnemonic="vli.all", args=VectorArgs(vd=4, imm=ROW_SIZE)),
        Instruction(mnemonic="vrecip.bf16", args=VectorArgs(vd=5, vs1=4)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=6, vs1=3, vs2=5)),
        # var_eps = var + eps
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=7, vs1=6, vs2=1)),
        # rsqrt = 1/sqrt(var_eps)
        Instruction(mnemonic="vsqrt.bf16", args=VectorArgs(vd=8, vs1=7)),
        Instruction(mnemonic="vrecip.bf16", args=VectorArgs(vd=9, vs1=8)),
        # output = x * rsqrt
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=10, vs1=0, vs2=9)),
        # MRF -> VMEM -> DRAM
        Instruction(mnemonic="vstore", args=VectorArgs(vd=10, rs1=3, imm12=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=16)),
        Instruction(
            mnemonic="dma.store.ch<N>", args=DmaArgs(rd=6, rs1=3, rs2=7, channel=0)
        ),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_INPUT_BASE, INPUT_DATA),
        (DRAM_EPS_BASE, torch.full(INPUT_DATA.shape, EPS, dtype=torch.bfloat16)),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUTPUT_BASE,
        gemma_rms_norm_forward(INPUT_DATA).to(torch.bfloat16),
    )
