"""Parameterized row-wise reduction-sum kernel for M×32 bf16 tensors.

N is fixed at 32. M must be a multiple of 32. Uses a hardware counted
loop over groups of 32 rows (M_groups iterations).

DRAM layout:
  dram_x:   M_groups * 2 * 1024 B  col-blocked input (H0 then H1 per group)
  dram_out: M_groups     * 1024 B  output row-sums (32x16 per group, H0 only)

VMEM slots:
  0x2000  VMEM_X   2 KB (H0 at +0, H1 at +1024)
  0x3000  VMEM_OUT 1 KB (v4 only, 32x16)

Scalar register map:
    x1  VMEM_X    x2  VMEM_OUT    x3  HALF_BYTES (1024)
    x4  VMEM_X + 1024 (H1 dest)
    x5  dram_x ptr (H0, advances by 2048 per group)
    x6  dram_out ptr (advances by 1024 per group)
    x7  group counter    x8  M_groups (limit)    x9  2048 (input group stride)
    x10  H1 input addr (computed per iter)
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
VMEM_OUT = 0x3000


def _colblock_bf16(mat: torch.Tensor, M: int) -> torch.Tensor:
    parts = []
    for g in range(M // TILE):
        group = mat[g * TILE : (g + 1) * TILE, :]
        parts.append(group[:, : TILE // 2].contiguous())
        parts.append(group[:, TILE // 2 :].contiguous())
    return torch.cat(parts, dim=0)


def reduction_sum_reference(x: torch.Tensor) -> torch.Tensor:
    row_sum = x.sum(dim=-1, keepdim=True).to(torch.bfloat16)
    return row_sum.expand(-1, TILE // 2).contiguous().to(x.dtype)


def _make_program(M: int, seed: int):
    M_groups = M // TILE
    dram_x = 0x0000
    dram_out = dram_x + M_groups * 2 * HALF_BYTES

    torch.manual_seed(seed)
    x = torch.randn(M, TILE, dtype=torch.bfloat16)
    expected = reduction_sum_reference(x)

    regions = [(dram_x, _colblock_bf16(x, M))]
    golden = (dram_out, expected)
    return regions, golden


_32_regions, _32_golden = _make_program(32, seed=110)


class ParameterizedReductionSum32x32Program(Program):
    """Row-wise reduction-sum on a single 32x32 bf16 tile -> (32, 16) output."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_reduction_sum32x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_regions, _64_golden = _make_program(64, seed=111)


class ParameterizedReductionSum64x32Program(Program):
    """Row-wise reduction-sum on a 64x32 bf16 tensor (2 groups) -> (64, 16) output."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_reduction_sum64x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_96_regions, _96_golden = _make_program(96, seed=112)


class ParameterizedReductionSum96x32Program(Program):
    """Row-wise reduction-sum on a 96x32 bf16 tensor (3 groups) -> (96, 16) output."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_reduction_sum96x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _96_regions
    golden_result: tuple[int, torch.Tensor] = _96_golden
