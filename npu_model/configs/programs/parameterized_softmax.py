"""Parameterized row-wise stable softmax kernel for M×32 bf16 tensors.

N is fixed at 32. M must be a multiple of 32. Uses a hardware counted
loop over groups of 32 rows (M_groups iterations).

Col-blocked layout (H0 = cols 0-15, H1 = cols 16-31 per group):
  dram_x:   M_groups * 2 * 1024 B input
  dram_out: M_groups * 2 * 1024 B output

VMEM slots:
  0x2000  VMEM_X   2 KB (H0 at +0, H1 at +1024)
  0x3000  VMEM_OUT 2 KB (H0 at +0, H1 at +1024)

Scalar register map:
    x1  VMEM_X    x2  VMEM_OUT    x3  HALF_BYTES (1024)
    x4  VMEM_X + 1024 (H1 dest)  x5  VMEM_OUT + 1024 (H1 source for DMA)
    x6  dram_x ptr (H0, advances by 2048 per group)
    x7  dram_out ptr (H0, advances by 2048 per group)
    x8  group counter    x9  M_groups (limit)    x10  2048 (group stride)
    x11  H1 input addr (computed per iter)    x12  H1 output addr (computed per iter)
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


def softmax_reference(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    xm = xf - xf.max(dim=-1, keepdim=True).values
    ex = xm.exp()
    return (ex / ex.sum(dim=-1, keepdim=True)).to(x.dtype)


def _make_program(M: int, seed: int):
    M_groups = M // TILE
    dram_x = 0x0000
    dram_out = dram_x + M_groups * 2 * HALF_BYTES

    torch.manual_seed(seed)
    x = torch.randn(M, TILE, dtype=torch.bfloat16) * 2.0
    expected = softmax_reference(x)

    regions = [(dram_x, _colblock_bf16(x, M))]
    golden = (dram_out, _colblock_bf16(expected, M))
    return regions, golden


_32_regions, _32_golden = _make_program(32, seed=100)


class ParameterizedSoftmax32x32Program(Program):
    """Row-wise softmax on a single 32x32 bf16 tile."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_softmax32x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_regions, _64_golden = _make_program(64, seed=101)


class ParameterizedSoftmax64x32Program(Program):
    """Row-wise softmax on a 64x32 bf16 tensor (2 groups)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_softmax64x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_96_regions, _96_golden = _make_program(96, seed=102)


class ParameterizedSoftmax96x32Program(Program):
    """Row-wise softmax on a 96x32 bf16 tensor (3 groups)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_softmax96x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _96_regions
    golden_result: tuple[int, torch.Tensor] = _96_golden
