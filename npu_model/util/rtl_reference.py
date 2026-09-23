"""Untimed kernel references using the RTL's numerical contract.

These compose arithmetic primitives without execution units, instruction issue,
port scheduling or memory accesses. RTL trace tests separately validate those
primitives and the cycle model. Ideal ML references can differ due to flushing,
lookup approximations, truncation and rounding after every SA MAC.
"""
import torch
from npu_model.hardware.rtl_math import unary, sa_fma, ipt_row
from npu_model.hardware.vpu import pack_row, _truncated_bf16


def quantize(value: torch.Tensor) -> torch.Tensor:
    return pack_row(value.contiguous().reshape(-1), 127).view(torch.float8_e4m3fn).reshape(value.shape)


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _truncated_bf16(a.float() + b.float())


def sa_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Default custom-FMA array rounds each inner-dimension MAC to BF16."""
    rows, inner = a.shape
    cols = b.shape[1]
    out = torch.zeros((rows, cols), dtype=torch.bfloat16)
    for k in range(inner):
        av = a.view(torch.uint8)[:, k:k+1].expand(rows, cols).contiguous().view(torch.float8_e4m3fn)
        bv = b.view(torch.uint8)[k:k+1, :].expand(rows, cols).contiguous().view(torch.float8_e4m3fn)
        out = sa_fma(av.flatten(), bv.flatten(), out.flatten()).reshape(rows, cols)
    return out


def row_sum(value: torch.Tensor) -> torch.Tensor:
    values = value.float()
    while values.shape[-1] > 1:
        values = values[..., ::2] + values[..., 1::2]
    return values.to(torch.bfloat16)


def ipt_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.bfloat16)
    for k in range(0, a.shape[1], 32):
        for col in range(0, b.shape[1], 32):
            weight = b.view(torch.uint8)[k:k+32, col:col+32].T.contiguous().view(torch.float8_e4m3fn)
            for row in range(a.shape[0]):
                out[row,col:col+32] = ipt_row(a[row,k:k+32], weight, out[row,col:col+32])
    return out
