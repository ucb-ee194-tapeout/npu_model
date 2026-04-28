"""Parameterized bf16→fp8 requantization kernel for arbitrary M×N.

Reads col-blocked bf16 input tiles (32×16 halves) and packs them into
contiguous 32×32 fp8 output tiles via vpack.bf16.fp8 with unit scale.

Constraints:
    - M and N must be multiples of 32.

DRAM layout (per _make_program):
  [dram_x  ]  M_tiles × N_tiles × 2 × 1024 B  — col-blocked bf16 input
  [dram_out ]  M_tiles × N_tiles     × 1024 B  — fp8 output (32×32 × 1 B)

VMEM slots:
  0x2000  VMEM_X_H0  1 KB — bf16 H0 (cols 0–15)
  0x2400  VMEM_X_H1  1 KB — bf16 H1 (cols 16–31)
  0x3000  VMEM_OUT   1 KB — fp8 tile (32×32 × 1 B)
"""

from typing import List, Tuple

import torch

from npu_model.software.program import Program, ASM_FOLDER
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction

TILE = 32
BF16_BYTES = 2
FP8_BYTES = 1
HALF_BYTES = TILE * (TILE // 2) * BF16_BYTES  # 1024  (32×16 bf16)
FP8_TILE_BYTES = TILE * TILE * FP8_BYTES       # 1024  (32×32 fp8)

VMEM_X_H0 = 0x2000
VMEM_X_H1 = 0x2400
VMEM_OUT = 0x3000


def _tile_fp8(mat: torch.Tensor, M: int, N: int) -> torch.Tensor:
    """Arrange M×N fp8 tensor into row-major tile order (matching DRAM output)."""
    M_tiles = M // TILE
    N_tiles = N // TILE
    parts = []
    for mt in range(M_tiles):
        for nt in range(N_tiles):
            tile = mat[mt * TILE : (mt + 1) * TILE, nt * TILE : (nt + 1) * TILE].contiguous()
            parts.append(tile.reshape(-1))
    return torch.cat(parts)


def _colblock_bf16(mat: torch.Tensor, M: int, N: int) -> torch.Tensor:
    """Arrange M×N into tiled col-blocked format.

    For tile (m, n): H0 (cols n*32 to n*32+15) then H1 (cols n*32+16 to n*32+31),
    each as a contiguous (32, 16) bf16 slice.  Tile order: row-major over tiles.
    """
    M_tiles = M // TILE
    N_tiles = N // TILE
    parts = []
    for mt in range(M_tiles):
        for nt in range(N_tiles):
            rows = mat[mt * TILE : (mt + 1) * TILE, nt * TILE : (nt + 1) * TILE]
            parts.append(rows[:, : TILE // 2].contiguous())
            parts.append(rows[:, TILE // 2 :].contiguous())
    return torch.cat(parts, dim=0)


def requant_reference(x: torch.Tensor) -> torch.Tensor:
    """bf16 → fp8_e4m3fn unit-scale cast.  Matches seli imm=1 path."""
    return x.to(torch.float8_e4m3fn)


def _make_program(M: int, N: int, seed: int):
    M_tiles = M // TILE
    N_tiles = N // TILE
    n_tiles = M_tiles * N_tiles

    dram_x = 0x0000
    dram_out = dram_x + n_tiles * 2 * HALF_BYTES

    torch.manual_seed(seed)
    # Keep values in fp8_e4m3fn range
    x = torch.randn(M, N, dtype=torch.bfloat16) * 0.5
    expected = requant_reference(x)

    regions = [(dram_x, _colblock_bf16(x, M, N))]
    golden = (dram_out, _tile_fp8(expected, M, N))
    return regions, golden


_32_regions, _32_golden = _make_program(32, 32, seed=120)


class ParameterizedRequant32x32Program(Program):
    """bf16→fp8 requant on a single 32×32 tile."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_requant32x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_regions, _64_golden = _make_program(64, 64, seed=121)


class ParameterizedRequant64x64Program(Program):
    """bf16→fp8 requant on a 64×64 tensor (2×2 tiles)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_requant64x64.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_64x32_regions, _64x32_golden = _make_program(64, 32, seed=122)


class ParameterizedRequant64x32Program(Program):
    """bf16→fp8 requant on a 64×32 tensor (2×1 tiles)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_requant64x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _64x32_regions
    golden_result: tuple[int, torch.Tensor] = _64x32_golden
