"""Arithmetic behind Instruction.exec, selected by ArchStateConfig.numerics.

Both backends implement one ISA: which elements an instruction combines and
where its results land are fixed by isa_definition. They differ only in
rounding, special values and function approximation. Tensors are BF16 unless
noted (mxu1 accumulators are FP32); ``scale`` is a raw ERF value, the power of two
2**(scale - 127).
"""
import numpy as np
import torch

from . import cvt, fp8_mma
from .rtl_math import unary, minmax, extremum, _truncated_bf16, canonical_nan


class RtlNumerics:
    """Bit-exact with the RTL datapaths."""

    @staticmethod
    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """AddSubSumVec: FP32 sum, then BF16 nearest even; together BF16_RNE of the exact sum."""
        return canonical_nan((a.float() + b.float()).to(torch.bfloat16))

    @staticmethod
    def sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return canonical_nan((a.float() - b.float()).to(torch.bfloat16))

    @staticmethod
    def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return canonical_nan(a * b)

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
        return canonical_nan(values.to(torch.bfloat16))

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
        """FP8Pack: E4M3_RN_SATFINITE(x * 2**(scale - 127)), as cvt.py."""
        bits = x.contiguous().view(torch.int16).numpy().astype(np.uint16).astype(np.uint32) << 16
        return torch.from_numpy(cvt.e4m3_rn_satfinite(bits, scale)).view(torch.float8_e4m3fn)

    @staticmethod
    def from_fp8(x: torch.Tensor, scale: int) -> torch.Tensor:
        """FP8Unpack: BF16_RNE(x * 2**(scale - 127)), as cvt.py."""
        codes = x.contiguous().view(torch.uint8).numpy()
        return torch.from_numpy(cvt.bf16_rne(cvt.e4m3_to_f32(codes), scale).view(np.int16)).view(torch.bfloat16)

    @staticmethod
    def f32_from_fp8(x: torch.Tensor) -> torch.Tensor:
        """mxu1 push: E4M3 to FP32, exact (cvt.py)."""
        codes = x.contiguous().view(torch.uint8).numpy()
        return torch.from_numpy(cvt.e4m3_to_f32(codes).view(np.float32))

    @staticmethod
    def f32_to_fp8(x: torch.Tensor, scale: int) -> torch.Tensor:
        """mxu1 pop: E4M3_RN_SATFINITE(x * 2**(scale - 127)) (cvt.py)."""
        bits = x.contiguous().view(torch.int32).numpy().view(np.uint32)
        return torch.from_numpy(cvt.e4m3_rn_satfinite(bits, scale)).view(torch.float8_e4m3fn)

    @staticmethod
    def f32_to_bf16(x: torch.Tensor, scale: int = 127) -> torch.Tensor:
        """mxu1 pops: BF16_RNE(x * 2**(scale - 127)) (cvt.py)."""
        bits = x.contiguous().view(torch.int32).numpy().view(np.uint32)
        return torch.from_numpy(cvt.bf16_rne(bits, scale).view(np.int16)).view(torch.bfloat16)

    @staticmethod
    def inner_product_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """AnchorAccumulationTree: FP32 c + a @ b, a Blackwell FP8 MMA per output (fp8_mma.py)."""
        a8 = a.contiguous().view(torch.uint8).numpy()
        w8 = b.contiguous().view(torch.uint8).T.numpy()
        bits = c.contiguous().view(torch.int32).numpy().view(np.uint32)
        return torch.from_numpy(fp8_mma.matmul(a8, w8, bits).view(np.float32))


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
        scaled = x.double() * 2.0 ** (int(scale) - 127)
        return scaled.clamp(-448, 448).to(torch.float8_e4m3fn)

    @staticmethod
    def from_fp8(x: torch.Tensor, scale: int) -> torch.Tensor:
        return (x.double() * 2.0 ** (int(scale) - 127)).to(torch.bfloat16)

    @staticmethod
    def f32_from_fp8(x: torch.Tensor) -> torch.Tensor:
        return x.float()

    @staticmethod
    def f32_to_fp8(x: torch.Tensor, scale: int) -> torch.Tensor:
        scaled = x.double() * 2.0 ** (int(scale) - 127)
        return scaled.clamp(-448, 448).to(torch.float8_e4m3fn)

    @staticmethod
    def f32_to_bf16(x: torch.Tensor, scale: int = 127) -> torch.Tensor:
        """BF16_RNE(x * 2**(scale - 127)), rounded once as cvt.bf16_rne; NaN gives 0x7FFF."""
        y = x.double() * 2.0 ** (int(scale) - 127)
        # Round in FP64, where y is exact, to the BF16 quantum of y; the cast is then exact.
        quantum = torch.exp2((torch.frexp(y).exponent.clamp(min=-125) - 8).double())
        bits = (torch.round(y / quantum) * quantum).to(torch.bfloat16).view(torch.int16)
        return torch.where(y.isnan(), 0x7FFF, bits).to(torch.int16).view(torch.bfloat16)

    @staticmethod
    def inner_product_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return a.float() @ b.float() + c


Numerics = RtlNumerics | TorchNumerics
NUMERICS: dict[str, Numerics] = {"rtl": RtlNumerics(), "pytorch": TorchNumerics()}
