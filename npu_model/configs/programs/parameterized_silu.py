"""Parameterized SiLU/Swish activation kernel for arbitrary M×N.

Computes SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
Uses a hardware counted loop over all 32x32 bf16 tiles.

Constraints:
    - M and N must be multiples of 32.

VMEM slots:
    VMEM_X   = 0x2000   2 KB
    VMEM_OUT = 0x2800   2 KB

Scalar register map:
    x1  VMEM_X    x2  VMEM_OUT    x3  TILE_BYTES_BF16 (2048, also stride)
    x4  dram_x pointer    x5  dram_out pointer
    x6  loop counter      x7  total_tiles (loop limit)
"""

from typing import List, Tuple

import torch

from npu_model.software.program import Program, ASM_FOLDER
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction

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


def silu_reference(x: torch.Tensor) -> torch.Tensor:
    return (x.float() * torch.sigmoid(x.float())).to(x.dtype)


def _make_program(M: int, N: int, seed: int):
    dram_x = 0x0000
    dram_out = dram_x + M * N * BF16_BYTES

    torch.manual_seed(seed)
    x = torch.randn(M, N, dtype=torch.bfloat16)
    expected = silu_reference(x)

    regions = [(dram_x, _tile_matrix_bf16(x, M, N))]
    golden = (dram_out, _tile_matrix_bf16(expected, M, N))
    return regions, golden


_32_regions, _32_golden = _make_program(32, 32, seed=50)


class ParameterizedSilu32x32Program(Program):
    """SiLU on a single 32x32 bf16 tile."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_silu32x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_regions, _64_golden = _make_program(64, 64, seed=51)


class ParameterizedSilu64x64Program(Program):
    """SiLU on a 64x64 bf16 tensor (2x2 tiles)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_silu64x64.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_64x32_regions, _64x32_golden = _make_program(64, 32, seed=52)


class ParameterizedSilu64x32Program(Program):
    """SiLU on a 64x32 bf16 tensor (2x1 tiles)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_silu64x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _64x32_regions
    golden_result: tuple[int, torch.Tensor] = _64x32_golden
