from typing import List, Tuple
from ...software import Instruction, Program
import torch

from ...workload.gemma_blocks import gemma_rms_norm_forward


# Input shape matches MRF: 64 rows, 16 bf16 per row
INPUT_DATA = torch.randn(64, 16, dtype=torch.bfloat16)
ROW_SIZE = INPUT_DATA.shape[-1]
EPS = 1e-6
INPUT_BASE = 0x0000
EPS_BASE = 0x0800  # 2048 bytes after input
DIVISOR_BASE = 0x1000  # 2048 bytes after eps
OUTPUT_BASE = 0x1800  # 2048 bytes after divisor


class GemmaRmsNormProgram(Program):
    """
    Gemma RMS norm program.
    RMS norm: x * rsqrt(mean(x^2) + eps).
    Row-wise mean via transpose + vreduce.sum (second-to-last dim) + vbroadcast.cols.
    """

    instructions: List[Instruction] = [
        Instruction(
            mnemonic="dma.load.ch0",
            args={
                "rd": 0,
                "base": INPUT_BASE,
                "size": INPUT_DATA.numel() * torch.bfloat16.itemsize,
            },
        ),
        Instruction(
            mnemonic="dma.load.ch1",
            args={
                "rd": 2,
                "base": EPS_BASE,
                "size": INPUT_DATA.numel() * torch.bfloat16.itemsize,
            },
        ),
        Instruction(
            mnemonic="dma.load.ch2",
            args={
                "rd": 8,
                "base": DIVISOR_BASE,
                "size": INPUT_DATA.numel() * torch.bfloat16.itemsize,
            },
        ),
        Instruction(mnemonic="dma.wait.ch0", args={}),
        Instruction(mnemonic="dma.wait.ch1", args={}),
        Instruction(mnemonic="dma.wait.ch2", args={}),

        # x_sq = x * x
        Instruction(mnemonic="vmul", args={"vrd": 3, "vs1": 0, "vs2": 0}),
        # Row-wise sum via vrot.reduce.sum -> full (16,64), then reduce + broadcast
        Instruction(mnemonic="vrot.reduce.sum", args={"vrd": 10, "vs1": 3}),
        # var = sum_sq / row_size = sum_sq * (1/row_size)
        Instruction(mnemonic="vrcp", args={"vrd": 9, "vs1": 8}),  # 1/row_size
        Instruction(mnemonic="vmul", args={"vrd": 4, "vs1": 10, "vs2": 9}),
        # var_eps = var + eps
        Instruction(mnemonic="vadd", args={"vrd": 5, "vs1": 4, "vs2": 2}),
        # rsqrt = 1/sqrt(var_eps)
        Instruction(mnemonic="vsqrt", args={"vrd": 6, "vs1": 5}),
        Instruction(mnemonic="vrcp", args={"vrd": 7, "vs1": 6}),
        # output = x * rsqrt
        Instruction(mnemonic="vmul", args={"vrd": 1, "vs1": 0, "vs2": 7}),

        Instruction(
            mnemonic="dma.store.ch0",
            args={
                "rs1": 1,
                "base": OUTPUT_BASE,
                "size": INPUT_DATA.numel() * torch.bfloat16.itemsize,
            },
        ),
        Instruction(mnemonic="dma.wait.ch0", args={}),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (INPUT_BASE, INPUT_DATA),
        (EPS_BASE, torch.full(INPUT_DATA.shape, EPS, dtype=torch.bfloat16)),
        (DIVISOR_BASE, torch.full(INPUT_DATA.shape, float(ROW_SIZE), dtype=torch.bfloat16)),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        OUTPUT_BASE,
        gemma_rms_norm_forward(INPUT_DATA).to(torch.bfloat16),
    )
