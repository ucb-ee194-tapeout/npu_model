from typing import List, Tuple
from ...software import Instruction, Program
import torch

from ...workload.gemma_blocks import gemma_rms_norm_forward
from npu_model.isa import ScalarArgs, DmaArgs, MatrixArgs, VectorArgs


# Input shape matches one BF16 tensor register: 32 rows x 16 columns.
INPUT_DATA = torch.randn(32, 16, dtype=torch.bfloat16)
ROW_SIZE = INPUT_DATA.shape[-1]
EPS = 1e-6
INPUT_BASE = 0x0000
EPS_BASE = 0x0400  # 1024 bytes after input
DIVISOR_BASE = 0x0800  # 1024 bytes after eps
OUTPUT_BASE = 0x0C00  # 1024 bytes after divisor


class GemmaRmsNormProgram(Program):
    """
    Gemma RMS norm program.
    RMS norm: x * rsqrt(mean(x^2) + eps).
    Row-wise mean via transpose + vreduce.sum (second-to-last dim) + vbroadcast.cols.
    """

    instructions: List[Instruction] = [
        Instruction(
            mnemonic="dma.load",
            args=DmaArgs(
                rd=0,
                base=INPUT_BASE,
                size=INPUT_DATA.numel() * torch.bfloat16.itemsize,
                flag=0,
            ),
        ),
        Instruction(
            mnemonic="dma.load",
            args=DmaArgs(
                rd=2,
                base=EPS_BASE,
                size=INPUT_DATA.numel() * torch.bfloat16.itemsize,
                flag=1,
            ),
        ),
        Instruction(
            mnemonic="dma.load",
            args=DmaArgs(
                rd=8,
                base=DIVISOR_BASE,
                size=INPUT_DATA.numel() * torch.bfloat16.itemsize,
                flag=2,
            ),
        ),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=0)),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=1)),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=2)),
        Instruction(mnemonic="vmul", args=VectorArgs(vrd=3, vrs1=0, vrs2=0)),
        Instruction(mnemonic="vrot.reduce.sum", args=VectorArgs(vrd=10, vrs1=3)),
        Instruction(mnemonic="vrcp", args=VectorArgs(vrd=9, vrs1=8)),
        Instruction(mnemonic="vmul", args=VectorArgs(vrd=4, vrs1=10, vrs2=9)),
        Instruction(mnemonic="vadd", args=VectorArgs(vrd=5, vrs1=4, vrs2=2)),
        Instruction(mnemonic="vsqrt", args=VectorArgs(vrd=6, vrs1=5)),
        Instruction(mnemonic="vrcp", args=VectorArgs(vrd=7, vrs1=6)),
        Instruction(mnemonic="vmul", args=VectorArgs(vrd=1, vrs1=0, vrs2=7)),
        Instruction(
            mnemonic="dma.store",
            args=DmaArgs(
                rs1=1,
                base=OUTPUT_BASE,
                size=INPUT_DATA.numel() * torch.bfloat16.itemsize,
                flag=0,
            ),
        ),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (INPUT_BASE, INPUT_DATA),
        (EPS_BASE, torch.full(INPUT_DATA.shape, EPS, dtype=torch.bfloat16)),
        (
            DIVISOR_BASE,
            torch.full(INPUT_DATA.shape, float(ROW_SIZE), dtype=torch.bfloat16),
        ),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        OUTPUT_BASE,
        gemma_rms_norm_forward(INPUT_DATA).to(torch.bfloat16),
    )
