"""Parameterized RMS-norm kernel: y = x * rsqrt(mean(x^2) + eps) for M×32.

N is fixed at 32. M must be a multiple of 32. Uses a hardware counted
loop over groups of 32 rows (M_groups iterations).

DRAM layout:
  dram_x:      M_groups * 2 * 1024 B  col-blocked input
  dram_inv_dim:             1024 B  1/32 broadcast constant
  dram_eps:                 1024 B  1e-6 broadcast constant
  dram_out:    M_groups * 2 * 1024 B  col-blocked output

VMEM slots:
  0x2000  VMEM_X_H0   1 KB
  0x2400  VMEM_X_H1   1 KB
  0x3000  VMEM_INVM   2 KB (H0 at 0x3000, H1 at 0x3400)
  0x4000  VMEM_EPS    2 KB (H0 at 0x4000, H1 at 0x4400)
  0x5000  VMEM_OUT    2 KB (H0 at 0x5000, H1 at 0x5400)

Scalar register map:
    x1  VMEM_X_H0    x2  VMEM_INV_DIM    x3  VMEM_EPS    x4  VMEM_OUT
    x5  HALF_BYTES (1024)    x6  VMEM_X_H1    x7  VMEM_INV_DIM_H1
    x8  VMEM_EPS_H1          x9  VMEM_OUT_H1
    x10  dram_x ptr (H0, advances by 2048 per group)
    x11  dram_out ptr (H0, advances by 2048 per group)
    x12  group counter    x13  M_groups (limit)    x14  2048 (group stride)
    x15  H1 input addr (computed per iter)    x16  H1 output addr (computed per iter)
"""

from typing import List, Tuple

import torch

from npu_model.software.program import Program, ASM_FOLDER
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction

TILE = 32
BF16_BYTES = 2
HALF_BYTES = TILE * (TILE // 2) * BF16_BYTES  # 1024

VMEM_X = 0x2000
VMEM_INV_DIM = 0x3000
VMEM_EPS = 0x4000
VMEM_OUT = 0x5000


def _colblock_bf16(mat: torch.Tensor, M: int) -> torch.Tensor:
    parts = []
    for g in range(M // TILE):
        group = mat[g * TILE : (g + 1) * TILE, :]
        parts.append(group[:, : TILE // 2].contiguous())
        parts.append(group[:, TILE // 2 :].contiguous())
    return torch.cat(parts, dim=0)


def rms_norm_reference(
    x: torch.Tensor,
    inv_dim: float = 1.0 / 32.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    xb = x.to(torch.bfloat16)
    sq = (xb * xb).to(torch.bfloat16)
    row_sum = sq.sum(dim=-1, keepdim=True).to(torch.bfloat16)
    inv_dim_t = torch.full_like(row_sum, inv_dim, dtype=torch.bfloat16)
    eps_t = torch.full_like(row_sum, eps, dtype=torch.bfloat16)
    mean = (row_sum * inv_dim_t).to(torch.bfloat16)
    denom = (mean + eps_t).to(torch.bfloat16)
    root = torch.sqrt(denom.float()).to(torch.bfloat16)
    inv = (1.0 / root.float()).to(torch.bfloat16)
    return (xb * inv).to(x.dtype)


def _make_program(M: int, seed: int):
    M_groups = M // TILE
    dram_x = 0x0000
    dram_inv_dim = dram_x + M_groups * 2 * HALF_BYTES
    dram_eps = dram_inv_dim + HALF_BYTES
    dram_out = dram_eps + HALF_BYTES

    torch.manual_seed(seed)
    x = torch.randn(M, TILE, dtype=torch.bfloat16)
    expected = rms_norm_reference(x)

    inv_dim_tile = torch.full((TILE, TILE // 2), 1.0 / TILE, dtype=torch.bfloat16)
    eps_tile = torch.full((TILE, TILE // 2), 1e-6, dtype=torch.bfloat16)

    regions = [
        (dram_x, _colblock_bf16(x, M)),
        (dram_inv_dim, inv_dim_tile),
        (dram_eps, eps_tile),
    ]
    golden = (dram_out, _colblock_bf16(expected, M))
    return regions, golden


_32_regions, _32_golden = _make_program(32, seed=90)


class ParameterizedRmsNorm32x32Program(Program):
    """RMS-norm on a single 32x32 bf16 tile."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_rms_norm32x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_regions, _64_golden = _make_program(64, seed=91)


class ParameterizedRmsNorm64x32Program(Program):
    """RMS-norm on a 64x32 bf16 tensor (2 groups)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_rms_norm64x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_96_regions, _96_golden = _make_program(96, seed=92)


class ParameterizedRmsNorm96x32Program(Program):
    """RMS-norm on a 96x32 bf16 tensor (3 groups)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_rms_norm96x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _96_regions
    golden_result: tuple[int, torch.Tensor] = _96_golden
