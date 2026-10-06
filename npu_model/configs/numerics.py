"""Arithmetic behind Instruction.exec, selected by ArchStateConfig.numerics.

Both backends implement one ISA: which elements an instruction combines and
where its results land are fixed by isa_definition. They differ only in
rounding, special values and function approximation. Tensors are BF16 unless
noted; ``scale`` is a raw ERF value, the power of two 2**(scale - 127).
"""
import torch

from .rtl_math import (unary, minmax, extremum, pack_row, unpack_row, _truncated_bf16,
                       sa_fma, ipt_row)


def _exponent(scale: int) -> int:
    return min(127, max(-128, int(scale) - 127))


class RtlNumerics:
    """Bit-exact with the RTL datapaths."""

    @staticmethod
    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _truncated_bf16(a.float() + b.float())

    @staticmethod
    def sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _truncated_bf16(a.float() - b.float())

    @staticmethod
    def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b

    @staticmethod
    def minimum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return minmax(a, b, False)

    @staticmethod
    def maximum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return minmax(a, b, True)

    @staticmethod
    def unary(name: str, x: torch.Tensor) -> torch.Tensor:
        """Exhaustive lane-box tables."""
        return unary("log" if name == "log2" else name, x)

    @staticmethod
    def row_sum(x: torch.Tensor) -> torch.Tensor:
        """ReduSumRec: balanced FP32 tree along the last dim, rounded to BF16."""
        values = x.float()
        while values.shape[-1] > 1:
            values = values[..., ::2] + values[..., 1::2]
        return values.to(torch.bfloat16)

    @staticmethod
    def column_sum(x: torch.Tensor) -> torch.Tensor:
        """ColAddVec: FP32 accumulation down dim 0 in row order, truncated to BF16."""
        total = x[0].float()
        for row in x[1:]:
            total = total + row.float()
        return _truncated_bf16(total).unsqueeze(0)

    @staticmethod
    def amin(x: torch.Tensor, dim: int) -> torch.Tensor:
        return extremum(x, dim, False)

    @staticmethod
    def amax(x: torch.Tensor, dim: int) -> torch.Tensor:
        return extremum(x, dim, True)

    @staticmethod
    def to_fp8(x: torch.Tensor, scale: int) -> torch.Tensor:
        """FP8Pack: saturates to 448 and flushes subnormal results to +0."""
        return pack_row(x.flatten(), scale).view(torch.float8_e4m3fn).reshape(x.shape)

    @staticmethod
    def from_fp8(x: torch.Tensor, scale: int) -> torch.Tensor:
        """FP8Unpack: flushes subnormal and NaN inputs to signed zero."""
        return unpack_row(x.flatten().view(torch.uint8), scale).reshape(x.shape)

    @staticmethod
    def acc_to_fp8(x: torch.Tensor, scale: int) -> torch.Tensor:
        """BF16ScaleToE4M3: like FP8Pack, but a value rounding to 480 encodes 0x7f."""
        return pack_row(x.flatten(), scale, mxu=True).view(torch.float8_e4m3fn).reshape(x.shape)

    @staticmethod
    def systolic_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Custom-FMA array: c + a @ b, rounding every MAC to BF16 in inner order."""
        rows, columns = a.shape[0], b.shape[1]
        a8, b8 = a.view(torch.uint8).contiguous(), b.view(torch.uint8).contiguous()
        out = c.flatten()
        for k in range(a.shape[1]):
            av = a8[:, k:k + 1].expand(rows, columns).reshape(-1).view(torch.float8_e4m3fn)
            bv = b8[k:k + 1, :].expand(rows, columns).reshape(-1).view(torch.float8_e4m3fn)
            out = sa_fma(av, bv, out)
        return out.reshape(rows, columns)

    @staticmethod
    def inner_product_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """AnchorAccumulationTree: c + a @ b, rounding each output once."""
        weights = b.view(torch.uint8).T.contiguous().view(torch.float8_e4m3fn)
        return torch.stack([ipt_row(a[row], weights, c[row]) for row in range(a.shape[0])])


_TORCH_UNARY = {
    "rcp": torch.reciprocal, "exp": torch.exp, "exp2": torch.exp2,
    "relu": torch.relu, "sin": torch.sin, "cos": torch.cos, "tanh": torch.tanh,
    "log2": torch.log2, "sqrt": torch.sqrt,
    "square": lambda x: x * x, "cube": lambda x: x * x * x,
}


class TorchNumerics:
    """PyTorch's arithmetic, as torch evaluates each operation on BF16 tensors."""

    @staticmethod
    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    @staticmethod
    def sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a - b

    @staticmethod
    def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b

    @staticmethod
    def minimum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.minimum(a, b)

    @staticmethod
    def maximum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.maximum(a, b)

    @staticmethod
    def unary(name: str, x: torch.Tensor) -> torch.Tensor:
        return _TORCH_UNARY[name](x)

    @staticmethod
    def row_sum(x: torch.Tensor) -> torch.Tensor:
        return x.sum(-1, keepdim=True)

    @staticmethod
    def column_sum(x: torch.Tensor) -> torch.Tensor:
        return x.sum(0, keepdim=True)

    @staticmethod
    def amin(x: torch.Tensor, dim: int) -> torch.Tensor:
        return x.amin(dim, keepdim=True)

    @staticmethod
    def amax(x: torch.Tensor, dim: int) -> torch.Tensor:
        return x.amax(dim, keepdim=True)

    @staticmethod
    def to_fp8(x: torch.Tensor, scale: int) -> torch.Tensor:
        """Saturates to 448, keeping subnormals."""
        scaled = x.float() * 2.0 ** -_exponent(scale)
        return scaled.clamp(-448, 448).to(torch.float8_e4m3fn)

    @staticmethod
    def from_fp8(x: torch.Tensor, scale: int) -> torch.Tensor:
        return (x.float() * 2.0 ** _exponent(scale)).to(torch.bfloat16)

    @staticmethod
    def acc_to_fp8(x: torch.Tensor, scale: int) -> torch.Tensor:
        scaled = x.float() * 2.0 ** _exponent(scale)
        return scaled.clamp(-448, 448).to(torch.float8_e4m3fn)

    @staticmethod
    def systolic_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return (a.float() @ b.float() + c.float()).to(torch.bfloat16)

    # Both arrays compute the same product; only RTL rounding tells them apart.
    inner_product_matmul = systolic_matmul


Numerics = RtlNumerics | TorchNumerics
NUMERICS: dict[str, Numerics] = {"rtl": RtlNumerics(), "pytorch": TorchNumerics()}
