from typing import List, Tuple

import math

import torch

from ...software import Instruction, Program


SEQ_LEN = 64
MODEL_DIM = 32
HEAD_DIM = 16
ROW_SIZE = HEAD_DIM
EPS = 1e-6
SCALE_VALUE = 1.0 / math.sqrt(float(HEAD_DIM))

Q_WEIGHT_DATA    = torch.ones((MODEL_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)
K_WEIGHT_DATA    = torch.ones((MODEL_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)
V_WEIGHT_DATA    = torch.ones((MODEL_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)
UP_WEIGHT_DATA   = torch.ones((MODEL_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)
DOWN_WEIGHT_DATA = torch.ones((MODEL_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)

INPUT_DATA = torch.ones((SEQ_LEN, MODEL_DIM), dtype=torch.float8_e4m3fn)

SCALE_DATA   = torch.full((SEQ_LEN, HEAD_DIM), SCALE_VALUE,     dtype=torch.bfloat16)
RMS_EPS_DATA = torch.full((SEQ_LEN, HEAD_DIM), EPS,             dtype=torch.bfloat16)
RMS_DIV_DATA = torch.full((SEQ_LEN, HEAD_DIM), float(ROW_SIZE), dtype=torch.bfloat16)
HALF_DATA    = torch.full((SEQ_LEN, HEAD_DIM), 0.5,             dtype=torch.bfloat16)

# Memory layout: Q_weight, K_weight, V_weight, up_weight, down_weight
Q_WEIGHT_BASE = 0x0000
K_WEIGHT_BASE = 0x0200
V_WEIGHT_BASE = 0x0400
UP_WEIGHT_BASE = 0x0600
DOWN_WEIGHT_BASE = 0x0800
INPUT_BASE       = 0x0A00
SCALE_BASE       = 0x1200
RMS_EPS_BASE     = 0x1A00
RMS_DIV_BASE     = 0x2200
K_INTER_BASE     = 0x2A00
V_INTER_BASE     = 0x2C00
HALF_BASE        = 0x2E00
OUTPUT_BASE      = 0x3600

# WB slot size in bytes — used for K/V spill size
_WB_BYTES = 512


class GemmaSingleHeadLayerProgram(Program):
    """
    Single-headed transformer layer: QKV projection → RMS norm → attention →
    up project → ReLU → down project → RMS norm.

    """

    instructions: List[Instruction] = [
        # Load Q_weight and K_weight into MXU1 weight buffer (slots 0 and 1)
        Instruction(
            mnemonic="dma.load.mxu1",
            args={
                "rd": 0,
                "base": Q_WEIGHT_BASE,
                "size": Q_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 0,
            },
        ),
        Instruction(
            mnemonic="dma.load.mxu1",
            args={
                "rd": 1,
                "base": K_WEIGHT_BASE,
                "size": K_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 1,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),
        # Load input into MRF 0
        Instruction(
            mnemonic="dma.load",
            args={
                "rd": 0,
                "base": INPUT_BASE,
                "size": INPUT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 0,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        # Q = input @ Q_weight -> MRF 1
        Instruction(mnemonic="matmul.mxu0", args={"rd": 1, "rs1": 0, "rs2": 0}),
        # K = input @ K_weight -> MRF 2
        Instruction(mnemonic="matmul.mxu0", args={"rd": 2, "rs1": 0, "rs2": 1}),
        # Reload slot 0 with V_weight
        Instruction(
            mnemonic="dma.load.mxu1",
            args={
                "rd": 0,
                "base": V_WEIGHT_BASE,
                "size": V_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 0,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        # V = input @ V_weight -> MRF 3
        Instruction(mnemonic="matmul.mxu0", args={"rd": 3, "rs1": 0, "rs2": 0}),
        # Transpose K (MRF 2): vtrpose.h upper half -> MRF 4, vtrpose.l lower half -> MRF 5, combine -> MRF 2
        Instruction(mnemonic="vtrpose.h", args={"vrd": 4, "vs1": 2}),
        Instruction(mnemonic="vtrpose.l", args={"vrd": 5, "vs1": 2}),
        Instruction(mnemonic="vadd", args={"vrd": 2, "vs1": 4, "vs2": 5}),
        # Load scale into MRF 11, eps into MRF 12, row_size into MRF 13
        Instruction(
            mnemonic="dma.load",
            args={
                "rd": 11,
                "base": SCALE_BASE,
                "size": SCALE_DATA.numel() * torch.bfloat16.itemsize,
                "flag": 0,
            },
        ),
        Instruction(
            mnemonic="dma.load",
            args={
                "rd": 12,
                "base": RMS_EPS_BASE,
                "size": RMS_EPS_DATA.numel() * torch.bfloat16.itemsize,
                "flag": 1,
            },
        ),
        Instruction(
            mnemonic="dma.load",
            args={
                "rd": 13,
                "base": RMS_DIV_BASE,
                "size": RMS_DIV_DATA.numel() * torch.bfloat16.itemsize,
                "flag": 2,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),
        Instruction(mnemonic="dma.wait", args={"flag": 2}),
        # Load half (0.5) constant into MRF 14
        Instruction(
            mnemonic="dma.load",
            args={
                "rd": 14,
                "base": HALF_BASE,
                "size": HALF_DATA.numel() * torch.bfloat16.itemsize,
                "flag": 0,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        # RMS norm on Q: Q_norm = Q / sqrt(mean(Q^2) + eps) -> MRF 1
        Instruction(mnemonic="vmul", args={"vrd": 5, "vs1": 1, "vs2": 1}),
        Instruction(mnemonic="vrot.reduce.sum", args={"vrd": 6, "vs1": 5}),
        Instruction(mnemonic="vrcp", args={"vrd": 7, "vs1": 13}),
        Instruction(mnemonic="vmul", args={"vrd": 8, "vs1": 6, "vs2": 7}),
        Instruction(mnemonic="vadd", args={"vrd": 9, "vs1": 8, "vs2": 12}),
        Instruction(mnemonic="vsqrt", args={"vrd": 10, "vs1": 9}),
        Instruction(mnemonic="vrcp", args={"vrd": 5, "vs1": 10}),
        Instruction(mnemonic="vmul", args={"vrd": 1, "vs1": 1, "vs2": 5}),
        # Spill K^T (MRF 2) and V (MRF 3) to SRAM, reload into MXU1 weight buffer
        Instruction(
            mnemonic="dma.store",
            args={
                "rs1": 2,
                "base": K_INTER_BASE,
                "size": _WB_BYTES,
                "flag": 0,
            },
        ),
        Instruction(
            mnemonic="dma.store",
            args={
                "rs1": 3,
                "base": V_INTER_BASE,
                "size": _WB_BYTES,
                "flag": 1,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),
        Instruction(
            mnemonic="dma.load.mxu1",
            args={
                "rd": 0,
                "base": K_INTER_BASE,
                "size": _WB_BYTES,
                "flag": 0,
            },
        ),
        Instruction(
            mnemonic="dma.load.mxu1",
            args={
                "rd": 1,
                "base": V_INTER_BASE,
                "size": _WB_BYTES,
                "flag": 1,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),
        # scores = Q_norm @ K^T -> MRF 3
        Instruction(mnemonic="matmul.mxu0", args={"rd": 3, "rs1": 1, "rs2": 0}),
        # scores_scaled = scores * scale
        Instruction(mnemonic="vmul", args={"vrd": 4, "vs1": 3, "vs2": 11}),
        # exp_scores = exp(scores_scaled)
        Instruction(mnemonic="vexp", args={"vrd": 5, "vs1": 4}),
        # col_sum = sum(exp_scores) (broadcast column-wise)
        Instruction(mnemonic="vreduce.sum", args={"vrd": 6, "vs1": 5}),
        # inv_col_sum = 1 / col_sum
        Instruction(mnemonic="vrcp", args={"vrd": 7, "vs1": 6}),
        # softmax_scores = exp_scores * inv_col_sum
        Instruction(mnemonic="vmul", args={"vrd": 8, "vs1": 5, "vs2": 7}),
        # attn_out = softmax_scores @ V -> MRF 9
        Instruction(mnemonic="matmul.mxu0", args={"rd": 9, "rs1": 8, "rs2": 1}),
        # Load up_weight and down_weight into MXU1 weight buffer (slots 0 and 1)
        Instruction(
            mnemonic="dma.load.mxu1",
            args={
                "rd": 0,
                "base": UP_WEIGHT_BASE,
                "size": UP_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 0,
            },
        ),
        Instruction(
            mnemonic="dma.load.mxu1",
            args={
                "rd": 1,
                "base": DOWN_WEIGHT_BASE,
                "size": DOWN_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 1,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),
        # up_out = attn_out @ up_weight -> MRF 10
        Instruction(mnemonic="matmul.mxu0", args={"rd": 10, "rs1": 9, "rs2": 0}),
        # relu(x) = 0.5 * (x + |x|), where |x| = sqrt(x^2)
        Instruction(mnemonic="vmul", args={"vrd": 4, "vs1": 10, "vs2": 10}),
        Instruction(mnemonic="vsqrt", args={"vrd": 4, "vs1": 4}),
        Instruction(mnemonic="vadd", args={"vrd": 10, "vs1": 10, "vs2": 4}),
        Instruction(mnemonic="vmul", args={"vrd": 10, "vs1": 10, "vs2": 14}),
        # down_out = relu_out @ down_weight -> MRF 11
        Instruction(mnemonic="matmul.mxu0", args={"rd": 11, "rs1": 10, "rs2": 1}),
        # RMS norm on down_out -> MRF 0
        Instruction(mnemonic="vmul", args={"vrd": 5, "vs1": 11, "vs2": 11}),
        Instruction(mnemonic="vrot.reduce.sum", args={"vrd": 6, "vs1": 5}),
        Instruction(mnemonic="vrcp", args={"vrd": 7, "vs1": 13}),
        Instruction(mnemonic="vmul", args={"vrd": 8, "vs1": 6, "vs2": 7}),
        Instruction(mnemonic="vadd", args={"vrd": 9, "vs1": 8, "vs2": 12}),
        Instruction(mnemonic="vsqrt", args={"vrd": 10, "vs1": 9}),
        Instruction(mnemonic="vrcp", args={"vrd": 5, "vs1": 10}),
        Instruction(mnemonic="vmul", args={"vrd": 0, "vs1": 11, "vs2": 5}),
        # Store result
        Instruction(
            mnemonic="dma.store",
            args={
                "rs1": 0,
                "base": OUTPUT_BASE,
                "size": SEQ_LEN * HEAD_DIM * torch.bfloat16.itemsize,
                "flag": 0,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (Q_WEIGHT_BASE,    Q_WEIGHT_DATA),
        (K_WEIGHT_BASE,    K_WEIGHT_DATA),
        (V_WEIGHT_BASE,    V_WEIGHT_DATA),
        (UP_WEIGHT_BASE,   UP_WEIGHT_DATA),
        (DOWN_WEIGHT_BASE, DOWN_WEIGHT_DATA),
        (INPUT_BASE,       INPUT_DATA),
        (SCALE_BASE,       SCALE_DATA),
        (RMS_EPS_BASE,     RMS_EPS_DATA),
        (RMS_DIV_BASE,     RMS_DIV_DATA),
        (HALF_BASE,        HALF_DATA),
    ]

    # Golden result: eager PyTorch reference matching the NPU instruction sequence
    golden_result: tuple[int, torch.Tensor]


def _rms_norm(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / (x.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()


def _mrf_as_fp8(bf16_tile: torch.Tensor) -> torch.Tensor:
    """Simulate read_mrf_fp8: reinterpret (64,16) bf16 bytes as (64,32) fp8."""
    return bf16_tile.flatten().view(torch.uint8).view(torch.float8_e4m3fn).reshape(64, 32)


def _wb_as_fp8(mrf_bf16: torch.Tensor) -> torch.Tensor:
    """Simulate WB spill: take first _WB_BYTES bytes of (64,16) bf16 MRF slot,
    then read them back as fp8 (32,16) — matching dma.store(size=512)+dma.load.mxu1."""
    raw = mrf_bf16.flatten().view(torch.uint8)[:_WB_BYTES]
    return raw.view(torch.float8_e4m3fn).reshape(32, 16)


def _hw_matmul(act: torch.Tensor, weight_fp8: torch.Tensor,
               acc: torch.Tensor | None = None) -> torch.Tensor:
    """Simulate matmul.mxu0: reads MRF as fp8 (64,32) and WB as fp8 (32,16)."""
    if act.dtype == torch.float8_e4m3fn:
        act_fp8 = act
    else:
        act_fp8 = _mrf_as_fp8(act)
    product = act_fp8.to(torch.float16) @ weight_fp8.to(torch.float16)
    if acc is not None:
        product = product + acc.to(torch.float16)
    return product.to(torch.bfloat16)


def _compute_golden() -> torch.Tensor:
    f32 = torch.float32

    Q = _hw_matmul(INPUT_DATA, Q_WEIGHT_DATA)
    K = _hw_matmul(INPUT_DATA, K_WEIGHT_DATA)
    V = _hw_matmul(INPUT_DATA, V_WEIGHT_DATA)

    Q_norm = _rms_norm(Q)

    K_wb = _wb_as_fp8(K.T.contiguous())
    V_wb = _wb_as_fp8(V)

    scores     = _hw_matmul(Q_norm, K_wb)
    scores_sc  = (scores * SCALE_VALUE).to(torch.bfloat16)
    exp_s      = scores_sc.to(f32).exp().to(torch.bfloat16)
    col_sum    = exp_s.sum(dim=0, keepdim=True).expand_as(exp_s).to(torch.bfloat16)
    softmax    = (exp_s.to(f32) / col_sum.to(f32)).to(torch.bfloat16)
    attn_out   = _hw_matmul(softmax, V_wb)

    up_out   = _hw_matmul(attn_out, UP_WEIGHT_DATA)
    relu_out = torch.relu(up_out.to(f32)).to(torch.bfloat16)
    down_out = _hw_matmul(relu_out, DOWN_WEIGHT_DATA)

    return _rms_norm(down_out).to(torch.bfloat16)


GemmaSingleHeadLayerProgram.golden_result = (OUTPUT_BASE, _compute_golden())
