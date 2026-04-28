"""Parameterized fused_norm_scale kernel for arbitrary M×N.

Computes out[i,j] = matrix[i,j] * rsqrt(variance[i,j]).
Uses a hardware counted loop over all 32x32 bf16 tiles.

Constraints:
    - M and N must be multiples of 32.
    - Variance values must be positive.

VMEM slots:
    VMEM_VAR = 0x2000   2 KB
    VMEM_MAT = 0x2800   2 KB
    VMEM_OUT = 0x3000   2 KB

Scalar register map:
    x1  VMEM_VAR    x2  VMEM_MAT    x3  VMEM_OUT
    x4  TILE_BYTES_BF16 (2048, also stride)
    x5  dram_var pointer    x6  dram_mat pointer    x7  dram_out pointer
    x8  loop counter        x9  total_tiles (loop limit)
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


def fused_norm_scale_reference(
    variance: torch.Tensor, matrix: torch.Tensor
) -> torch.Tensor:
    sqrt_v = torch.sqrt(variance.float()).to(torch.bfloat16)
    rsqrt_v = (1.0 / sqrt_v.float()).to(torch.bfloat16)
    return (matrix.float() * rsqrt_v.float()).to(matrix.dtype)


def _make_program(M: int, N: int, seed: int):
    dram_var = 0x0000
    dram_mat = dram_var + M * N * BF16_BYTES
    dram_out = dram_mat + M * N * BF16_BYTES

    torch.manual_seed(seed)
    variance = (torch.randn(M, N).abs() + 0.1).to(torch.bfloat16)
    matrix = torch.randn(M, N, dtype=torch.bfloat16)
    expected = fused_norm_scale_reference(variance, matrix)

    regions = [
        (dram_var, _tile_matrix_bf16(variance, M, N)),
        (dram_mat, _tile_matrix_bf16(matrix, M, N)),
    ]
    golden = (dram_out, _tile_matrix_bf16(expected, M, N))
    return regions, golden


_32_regions, _32_golden = _make_program(32, 32, seed=80)


class ParameterizedFusedNormScale32x32Program(Program):
    """fused_norm_scale on a single 32x32 bf16 tile."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_fused_norm_scale32x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_regions, _64_golden = _make_program(64, 64, seed=81)


class ParameterizedFusedNormScale64x64Program(Program):
    """fused_norm_scale on a 64x64 bf16 tensor (2x2 tiles)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_fused_norm_scale64x64.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_64x32_regions, _64x32_golden = _make_program(64, 32, seed=82)


class ParameterizedFusedNormScale64x32Program(Program):
    """fused_norm_scale on a 64x32 bf16 tensor (2x1 tiles)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_fused_norm_scale64x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _64x32_regions
    golden_result: tuple[int, torch.Tensor] = _64x32_golden
