from typing import List, Tuple
import math
import torch
from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, VectorArgs, ScalarArgs


SEQ_LEN = 64
HEAD_DIM = 16

# Data tensors for Q, K, V in fp8 (to match MXU matmul expectations)
QUERY_DATA = torch.ones((SEQ_LEN, HEAD_DIM), dtype=torch.float8_e4m3fn)
KEY_DATA = torch.ones((HEAD_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)
VALUE_DATA = torch.ones((HEAD_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)

# Scaling matrix: every entry is 1 / sqrt(HEAD_DIM), in bf16 for vector ops
SCALE_VALUE = 1.0 / math.sqrt(float(HEAD_DIM))
SCALE_DATA = torch.full(
    (SEQ_LEN, HEAD_DIM),
    SCALE_VALUE,
    dtype=torch.bfloat16,
)

QUERY_BASE = 0x0000
KEY_BASE = 0x2000
VALUE_BASE = 0x3000
SCALE_BASE = 0x4000
OUTPUT_BASE = 0x5000


class GemmaAttentionProgram(Program):
    """
    Gemma attention kernel program (simplified, single-head).

    This program demonstrates a scaled dot-product attention block using
    the NPU ISA:
      - `matmul.mxu0` for Q @ K and softmax(QK^T) @ V
      - `vexp`, `vreduce.sum`, `vrcp`, and `vmul` to implement softmax.
    """

    instructions: List[Instruction] = [
        # Load K and V into MXU1 weight buffer (indices 0 and 1)
        Instruction(
            mnemonic="dma.load.mxu1",
            args=DmaArgs(
                rd=0,
                base=KEY_BASE,
                size=KEY_DATA.numel() * torch.float8_e4m3fn.itemsize,
                flag=0,
            ),
        ),
        Instruction(
            mnemonic="dma.load.mxu1",
            args=DmaArgs(
                rd=1,
                base=VALUE_BASE,
                size=VALUE_DATA.numel() * torch.float8_e4m3fn.itemsize,
                flag=1,
            ),
        ),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=0)),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=1)),
        # Load Q into MRF 0 and the scaling matrix into MRF 2
        Instruction(
            mnemonic="dma.load",
            args=DmaArgs(
                rd=0,
                base=QUERY_BASE,
                size=QUERY_DATA.numel() * torch.float8_e4m3fn.itemsize,
                flag=2,
            ),
        ),
        Instruction(
            mnemonic="dma.load",
            args=DmaArgs(
                rd=2,
                base=SCALE_BASE,
                size=SCALE_DATA.numel() * torch.bfloat16.itemsize,
                flag=3,
            ),
        ),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=2)),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=3)),
        # scores = Q @ K   -> MRF 3
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=0, vs2=0)),
        Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=3, vs1=0)),
        # scores_scaled = scores * scale
        Instruction(mnemonic="vmul", args=VectorArgs(vrd=4, vs1=3, vs2=2)),
        # exp_scores = exp(scores_scaled)
        Instruction(mnemonic="vexp", args=VectorArgs(vrd=5, vs1=4)),
        # row_sum = sum(exp_scores) (broadcast row-wise)
        Instruction(mnemonic="vreduce.sum", args=VectorArgs(vrd=6, vs1=5)),
        # inv_row_sum = 1 / row_sum
        Instruction(mnemonic="vrcp", args=VectorArgs(vrd=7, vs1=6)),
        # softmax_scores = exp_scores * inv_row_sum
        Instruction(mnemonic="vmul", args=VectorArgs(vrd=8, vs1=5, vs2=7)),
        # attn_output = softmax_scores @ V  -> MRF 9
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=8, vs2=1)),
        Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=9, vs1=0)),
        # Store result
        Instruction(
            mnemonic="dma.store",
            args=DmaArgs(
                rs1=9,
                base=OUTPUT_BASE,
                size=SEQ_LEN * HEAD_DIM * torch.bfloat16.itemsize,
                flag=4,
            ),
        ),
        Instruction(mnemonic="dma.wait", args=DmaArgs(flag=4)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (QUERY_BASE, QUERY_DATA),
        (KEY_BASE, KEY_DATA),
        (VALUE_BASE, VALUE_DATA),
        (SCALE_BASE, SCALE_DATA),
    ]

    # Golden result: scaled dot-product attention in eager PyTorch
    golden_result: tuple[int, torch.Tensor] = (
        OUTPUT_BASE,
        (
            (QUERY_DATA.to(torch.float32) @ KEY_DATA.to(torch.float32)) * SCALE_VALUE
        ).softmax(dim=-1)
        @ VALUE_DATA.to(torch.float32),
    )[0].to(torch.bfloat16)
