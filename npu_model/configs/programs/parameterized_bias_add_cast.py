"""Parameterized bias-add-cast kernel: fp8(x + bias) for 32×32.

Fuses an elementwise bf16 addition (x + bias) with an fp8 quantization
(unit-scale bf16 → fp8_e4m3fn cast via vpack.bf16.fp8).

DRAM layout (per _make_program):
  [dram_x   ]  2 × 1024 B  — col-blocked bf16 x   (H0 then H1)
  [dram_bias ]  2 × 1024 B  — col-blocked bf16 bias (H0 then H1)
  [dram_out  ]      1024 B  — fp8 output tile (32×32 × 1 B)

VMEM slots:
  0x2000  VMEM_X      2 KB  — x tile (H0 at 0x2000, H1 at 0x2400 via imm12=32)
  0x2800  VMEM_BIAS   2 KB  — bias tile (H0 at 0x2800, H1 at 0x2C00 via imm12=32)
  0x3000  VMEM_OUT    1 KB  — fp8 output

MRF layout per tile:
  (v0, v1) = x bf16 LMUL=2 pair
  (v2, v3) = bias bf16 LMUL=2 pair
  (v4, v5) = x + bias  (vadd.bf16)
  v6       = fp8(x + bias)  (vpack.bf16.fp8)

Constraint: single 32×32 tile only (M=N=32).
"""

from typing import List, Tuple

import torch

from npu_model.software.program import Program, ASM_FOLDER
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction

TILE = 32
BF16_BYTES = 2
FP8_BYTES = 1
HALF_BYTES = TILE * (TILE // 2) * BF16_BYTES  # 1024 B (32×16 bf16)
FP8_TILE_BYTES = TILE * TILE * FP8_BYTES       # 1024 B (32×32 fp8)
BF16_TILE_BYTES = TILE * TILE * BF16_BYTES     # 2048 B (two halves)

VMEM_X = 0x2000
VMEM_BIAS = 0x2800
VMEM_OUT = 0x3000


def _colblock_bf16(mat: torch.Tensor) -> torch.Tensor:
    """Pack 32×32 bf16 into col-blocked layout: H0 (32×16) then H1 (32×16)."""
    assert mat.shape == (TILE, TILE)
    h0 = mat[:, : TILE // 2].contiguous()
    h1 = mat[:, TILE // 2 :].contiguous()
    return torch.cat([h0, h1], dim=0)  # (64, 16) bf16 = 2048 B


def _tile_fp8(mat: torch.Tensor) -> torch.Tensor:
    """Flatten 32×32 fp8 tile to 1-D (row-major)."""
    assert mat.shape == (TILE, TILE)
    return mat.reshape(-1)


def bias_add_cast_reference(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """fp8(x + bias) with unit scale — matches vpack.bf16.fp8 seli=1 path."""
    return (x.float() + bias.float()).to(torch.float8_e4m3fn)


def _make_program(seed: int):
    dram_x = 0x0000
    dram_bias = dram_x + BF16_TILE_BYTES
    dram_out = dram_bias + BF16_TILE_BYTES

    torch.manual_seed(seed)
    # Keep values in fp8 range to avoid saturation in reference
    x = torch.randn(TILE, TILE, dtype=torch.bfloat16) * 0.4
    bias = torch.randn(TILE, TILE, dtype=torch.bfloat16) * 0.1
    expected = bias_add_cast_reference(x, bias)

    regions = [
        (dram_x, _colblock_bf16(x)),
        (dram_bias, _colblock_bf16(bias)),
    ]
    golden = (dram_out, _tile_fp8(expected))
    return regions, golden


_regions, _golden = _make_program(seed=210)


class ParameterizedBiasAddCast32x32Program(Program):
    """bias_add_cast: fp8(x + bias) on a single 32×32 bf16 tile."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_bias_add_cast32x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _regions
    golden_result: tuple[int, torch.Tensor] = _golden
    kernel_tolerance: tuple[float, float] = (1e-1, 1e-1)
