"""Parameterized fused matmul+bias kernel: (A_fp8 @ B_fp8)_bf16 + bias_bf16.

Tiles the computation over M×N output tiles with K=32 fixed.
A: M×32 fp8, B: 32×N fp8, bias: M×N bf16, output: M×N bf16.
M and N must be multiples of 32.

Bias and output use col-blocked bf16 layout (32×16 halves per tile).
A and B tiles are contiguous 32×32 fp8 tiles (1024 B each).

DRAM layout (per _make_program):
  [dram_a   ]  M_tiles × 1024 B           — fp8 A row-blocks (one per M tile)
  [dram_b   ]  N_tiles × 1024 B           — fp8 B col-blocks (one per N tile)
  [dram_bias ]  M_tiles × N_tiles × 2048 B — col-blocked bf16 bias
  [dram_out  ]  M_tiles × N_tiles × 2048 B — col-blocked bf16 output

VMEM slots:
  0x2000  VMEM_A     1 KB — fp8 A tile
  0x2400  VMEM_B     1 KB — fp8 B tile
  0x2800  VMEM_BIAS  2 KB — bf16 bias pair (H0 at +0, H1 at +1024)
  0x3000  VMEM_OUT   2 KB — bf16 output pair (H0 at +0, H1 at +1024)

MRF layout per output tile:
  v0         = A fp8 (32×32, single MRF register)
  v2         = B fp8 (32×32)
  (v4,  v5 ) = bias bf16 pair
  (v6,  v7 ) = matmul result bf16  (from vmatpop.bf16.acc.mxu0 vd=6)
  (v8,  v9 ) = output = result + bias
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
BF16_TILE_BYTES = TILE * TILE * BF16_BYTES     # 2048  (32×32 bf16)

VMEM_A = 0x2000
VMEM_B = 0x2400
VMEM_BIAS = 0x2800
VMEM_OUT = 0x3000


def _colblock_bf16(mat: torch.Tensor, M: int, N: int) -> torch.Tensor:
    """Arrange M×N into tiled col-blocked format (H0 then H1 per tile)."""
    M_tiles = M // TILE
    N_tiles = N // TILE
    parts = []
    for mt in range(M_tiles):
        for nt in range(N_tiles):
            rows = mat[mt * TILE : (mt + 1) * TILE, nt * TILE : (nt + 1) * TILE]
            parts.append(rows[:, : TILE // 2].contiguous())
            parts.append(rows[:, TILE // 2 :].contiguous())
    return torch.cat(parts, dim=0)


def fused_matmul_bias_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """(A_fp8 @ B_fp8) → bf16, then + bias.  Mirrors MXU semantics."""
    mm = (a.to(torch.float32) @ b.to(torch.float32)).to(torch.bfloat16)
    return (mm.float() + bias.float()).to(torch.bfloat16)


def _make_program(M: int, N: int, seed: int):
    M_tiles = M // TILE
    N_tiles = N // TILE
    n_tiles = M_tiles * N_tiles

    dram_a = 0x0000
    dram_b = dram_a + M_tiles * FP8_TILE_BYTES
    dram_bias = dram_b + N_tiles * FP8_TILE_BYTES
    dram_out = dram_bias + n_tiles * BF16_TILE_BYTES

    torch.manual_seed(seed)
    a = torch.randint(-8, 8, (M, TILE), dtype=torch.int8).to(torch.float8_e4m3fn)
    b = torch.randint(-8, 8, (TILE, N), dtype=torch.int8).to(torch.float8_e4m3fn)
    bias = torch.randn(M, N, dtype=torch.bfloat16)
    expected = fused_matmul_bias_reference(a, b, bias)

    a_tiled = torch.cat([
        a[mt * TILE : (mt + 1) * TILE, :].contiguous()
        for mt in range(M_tiles)
    ])
    b_tiled = torch.cat([
        b[:, nt * TILE : (nt + 1) * TILE].contiguous()
        for nt in range(N_tiles)
    ])

    regions = [
        (dram_a, a_tiled),
        (dram_b, b_tiled),
        (dram_bias, _colblock_bf16(bias, M, N)),
    ]
    golden = (dram_out, _colblock_bf16(expected, M, N))
    return regions, golden


_32_regions, _32_golden = _make_program(32, 32, seed=130)


class ParameterizedFusedMatmulBias32x32Program(Program):
    """fused_matmul_bias on a single 32×32 tile (K=32)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_fused_matmul_bias32x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_regions, _64_golden = _make_program(64, 64, seed=131)


class ParameterizedFusedMatmulBias64x64Program(Program):
    """fused_matmul_bias on a 64×64 bf16 output tensor (K=32, 2×2 output tiles)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_fused_matmul_bias64x64.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_64x32_regions, _64x32_golden = _make_program(64, 32, seed=132)


class ParameterizedFusedMatmulBias64x32Program(Program):
    """fused_matmul_bias on a 64×32 bf16 output tensor (K=32, 2×1 output tiles)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_fused_matmul_bias64x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _64x32_regions
    golden_result: tuple[int, torch.Tensor] = _64x32_golden
