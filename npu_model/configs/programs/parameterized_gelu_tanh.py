"""Parameterized GELU (tanh approximation) kernel for M×32 bf16 tensors.

Computes:  y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

N is fixed at 32.  M must be a multiple of 32.  Uses col-blocked layout.
Three polynomial constants (k_coeff, k_sqrt2pi, k_half) are loaded from
DRAM once; k_one = 1.0 is materialised with vli.all.

DRAM layout (per _make_program):
  [dram_x        ]  M_groups × 2 × 1024 B  — col-blocked input
  [dram_k_coeff  ]  1024 B  — 0.044715 broadcast tile (32×16 bf16)
  [dram_k_sqrt2pi]  1024 B  — sqrt(2/π) ≈ 0.7979 broadcast tile
  [dram_k_half   ]  1024 B  — 0.5 broadcast tile
  [dram_out      ]  M_groups × 2 × 1024 B  — col-blocked output

VMEM slots:
  0x2000  VMEM_X        2 KB — current input pair
  0x2800  VMEM_K_COEFF  2 KB — 0.044715 pair (both halves same data)
  0x3000  VMEM_K_SQRT   2 KB — sqrt(2/π) pair
  0x3800  VMEM_K_HALF   2 KB — 0.5 pair
  0x4000  VMEM_OUT      2 KB — current output pair

MRF layout per group:
  (v0,  v1 ) = X
  (v2,  v3 ) = X²              via vsquare.bf16
  (v12, v13) = X³              via vmul(v2, v0)
  (v12, v13) = k_coeff * X³   via vmul(v12, v4)
  (v12, v13) = X + k_coeff*X³ via vadd(v0, v12)
  (v12, v13) *= k_sqrt2pi      via vmul(v12, v6)
  (v12, v13) = tanh(...)       via vtanh.bf16
  (v10, v11) = 1.0             via vli.all
  (v12, v13) = 1 + tanh(...)   via vadd(v10, v12)
  (v12, v13) *= k_half         via vmul(v12, v8)
  (v14, v15) = Y = x*(...)     via vmul(v0, v12)
"""

import math
from typing import List, Tuple

import torch

from npu_model.software.program import Program, ASM_FOLDER
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction

TILE = 32
BF16_BYTES = 2
HALF_BYTES = TILE * (TILE // 2) * BF16_BYTES  # 1024

VMEM_X = 0x2000
VMEM_K_COEFF = 0x2800
VMEM_K_SQRT = 0x3000
VMEM_K_HALF = 0x3800
VMEM_OUT = 0x4000

_K_COEFF = 0.044715
_K_SQRT2PI = math.sqrt(2.0 / math.pi)  # ≈ 0.7979
_K_HALF = 0.5


def _colblock_bf16(mat: torch.Tensor, M: int) -> torch.Tensor:
    """Arrange M×32 into col-blocked format: for each 32-row group,
    H0 (cols 0–15) then H1 (cols 16–31)."""
    parts = []
    for g in range(M // TILE):
        group = mat[g * TILE : (g + 1) * TILE, :]
        parts.append(group[:, : TILE // 2].contiguous())
        parts.append(group[:, TILE // 2 :].contiguous())
    return torch.cat(parts, dim=0)


def gelu_tanh_reference(x: torch.Tensor) -> torch.Tensor:
    """GELU tanh approximation matching the bf16 ISA sequence."""
    xf = x.float()
    x3 = xf.pow(3.0)
    inner = _K_SQRT2PI * (xf + _K_COEFF * x3)
    return (xf * 0.5 * (1.0 + torch.tanh(inner))).to(x.dtype)


def _make_program(M: int, seed: int):
    M_groups = M // TILE
    dram_x = 0x0000
    dram_k_coeff = dram_x + M_groups * 2 * HALF_BYTES
    dram_k_sqrt2pi = dram_k_coeff + HALF_BYTES
    dram_k_half = dram_k_sqrt2pi + HALF_BYTES
    dram_out = dram_k_half + HALF_BYTES

    torch.manual_seed(seed)
    x = torch.randn(M, TILE, dtype=torch.bfloat16) * 0.5
    expected = gelu_tanh_reference(x)

    k_coeff_tile = torch.full((TILE, TILE // 2), _K_COEFF, dtype=torch.bfloat16)
    k_sqrt_tile = torch.full((TILE, TILE // 2), _K_SQRT2PI, dtype=torch.bfloat16)
    k_half_tile = torch.full((TILE, TILE // 2), _K_HALF, dtype=torch.bfloat16)

    regions = [
        (dram_x, _colblock_bf16(x, M)),
        (dram_k_coeff, k_coeff_tile),
        (dram_k_sqrt2pi, k_sqrt_tile),
        (dram_k_half, k_half_tile),
    ]
    golden = (dram_out, _colblock_bf16(expected, M))
    return regions, golden


_32_regions, _32_golden = _make_program(32, seed=140)


class ParameterizedGeluTanh32x32Program(Program):
    """GELU (tanh approximation) on a single 32×32 bf16 tile."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_gelu_tanh32x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_regions, _64_golden = _make_program(64, seed=141)


class ParameterizedGeluTanh64x32Program(Program):
    """GELU (tanh approximation) on a 64×32 bf16 tensor (2 groups)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_gelu_tanh64x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_96_regions, _96_golden = _make_program(96, seed=142)


class ParameterizedGeluTanh96x32Program(Program):
    """GELU (tanh approximation) on a 96×32 bf16 tensor (3 groups)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_gelu_tanh96x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _96_regions
    golden_result: tuple[int, torch.Tensor] = _96_golden
