import math
import torch
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER

SEQ_LEN = 32
HEAD_DIM = 32

QUERY_DATA = torch.ones((SEQ_LEN, HEAD_DIM), dtype=torch.float8_e4m3fn)
KEY_DATA = torch.ones((SEQ_LEN, HEAD_DIM), dtype=torch.float8_e4m3fn)

# Scaling matrix: every entry is 1 / sqrt(HEAD_DIM), in bf16 for vector ops
SCALE_VALUE = 1.0 / math.sqrt(float(HEAD_DIM))
SCALE_DATA = torch.full((SEQ_LEN, HEAD_DIM), SCALE_VALUE, dtype=torch.bfloat16)

# DRAM layout (program-loaded)
DRAM_QUERY_BASE = 0x0000
DRAM_KEY_BASE = 0x0400
DRAM_SCALE_BASE = 0x0800
DRAM_OUTPUT_BASE = 0x1000


class GemmaAttentionProgram(Program):
    """
    Gemma attention kernel program (simplified, single-head).

    This program demonstrates a scaled dot-product attention block using
    the NPU ISA:
      - `matmul.mxu0` for Q @ K and softmax(QK^T) @ V
      - `vexp`, `vreduce.sum`, `vrcp`, and `vmul` to implement softmax.
    """

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'gemma_attention.S')

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_QUERY_BASE, QUERY_DATA),
        (DRAM_KEY_BASE, KEY_DATA),
        (DRAM_SCALE_BASE, SCALE_DATA),
    ]

    # Golden result: softmax(scores_scaled) (no max subtraction), pure torch
    golden_result: list[tuple[int, torch.Tensor]] = [(
        DRAM_OUTPUT_BASE,
        torch.softmax(
            (QUERY_DATA.to(torch.float32) @ KEY_DATA.to(torch.float32)) * SCALE_VALUE,
            dim=1,
        ).to(torch.bfloat16),
    )]
