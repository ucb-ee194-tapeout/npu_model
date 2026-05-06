"""Parameterized elementwise divide kernel: C = A / B for arbitrary M×N.

Uses vrecip.bf16 + vmul.bf16 (no dedicated divide instruction). Uses a
hardware counted loop over all tiles. Each tile is 2048 B bf16.

Constraints:
    - M and N must be multiples of 32.

VMEM slots: VMEM_A=0x2000, VMEM_B=0x2800, VMEM_C=0x3000 (each 2 KB).

Scalar register map:
    x1  VMEM_A    x2  VMEM_B    x3  VMEM_C
    x4  TILE_BYTES_BF16 (2048, also used as pointer stride)
    x5  dram_a pointer    x6  dram_b pointer    x7  dram_c pointer
    x8  loop counter      x9  total_tiles (loop limit)
"""

from typing import List, Tuple

import torch

from npu_model.software.program import Program, ASM_FOLDER
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction

VMEM_A = 0x2000
VMEM_B = 0x2800
VMEM_C = 0x3000

TILE = 32
BF16_BYTES = 2
TILE_BYTES_BF16 = TILE * TILE * BF16_BYTES


def _tile_matrix_bf16(mat: torch.Tensor, M: int, N: int) -> torch.Tensor:
    M_tiles = M // TILE
    N_tiles = N // TILE
    parts = []
    for r in range(M_tiles):
        for c in range(N_tiles):
            tile = mat[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE].contiguous()
            parts.append(tile.reshape(-1))
    return torch.cat(parts)


def elementwise_div_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Kernel computes via vrecip + vmul in bf16; mirror the rounding."""
    inv_b = (1.0 / b.float()).to(torch.bfloat16)
    return (a * inv_b).to(a.dtype)


def _make_program(M: int, N: int, seed: int):
    dram_a = 0x0000
    dram_b = dram_a + M * N * BF16_BYTES
    dram_c = dram_b + M * N * BF16_BYTES

    torch.manual_seed(seed)
    a = torch.randn(M, N, dtype=torch.bfloat16)
    b = torch.randn(M, N, dtype=torch.bfloat16)
    b = torch.where(b.abs() < 0.25, torch.full_like(b, 0.5), b)
    expected = elementwise_div_reference(a, b)

    regions = [
        (dram_a, _tile_matrix_bf16(a, M, N)),
        (dram_b, _tile_matrix_bf16(b, M, N)),
    ]
    golden = (dram_c, _tile_matrix_bf16(expected, M, N))
    return regions, golden


_32_regions, _32_golden = _make_program(32, 32, seed=40)


class ParameterizedElementwiseDiv32x32Program(Program):
    """Elementwise divide on a single 32x32 bf16 tile."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_elementwise_div32x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_regions, _64_golden = _make_program(64, 64, seed=41)


class ParameterizedElementwiseDiv64x64Program(Program):
    """Elementwise divide on a 64x64 bf16 tensor (2x2 tiles)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_elementwise_div64x64.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_64x32_regions, _64x32_golden = _make_program(64, 32, seed=42)


class ParameterizedElementwiseDiv64x32Program(Program):
    """Elementwise divide on a 64x32 bf16 tensor (2x1 tiles)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_elementwise_div64x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _64x32_regions
    golden_result: tuple[int, torch.Tensor] = _64x32_golden
