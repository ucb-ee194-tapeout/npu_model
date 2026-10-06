"""Parameterized batch matmul: C[b] = A[b] @ B[b] for B batches of M×K×N fp8.

Each batch element is an independent fp8 tiled matmul. Uses hardware
branch loops over B, M, N, and K dimensions so the instruction count
is O(1) in the problem size.

DRAM layout (per _make_program):
  [dram_a]  B × M_tiles × K_tiles × 1024 B  — fp8 A tiles, batch-major
  [dram_b]  B × K_tiles × N_tiles × 1024 B  — fp8 B tiles, batch-major
  [dram_c]  B × M_tiles × N_tiles × 2048 B  — col-blocked bf16 C tiles

VMEM slots (fixed, reused per tile):
  0x2000  VMEM_A   1 KB — current fp8 A tile
  0x2400  VMEM_B   1 KB — current fp8 B tile
  0x2800  VMEM_C0  1 KB — C low  half (cols  0-15)
  0x2C00  VMEM_C1  1 KB — C high half (cols 16-31)

Constraints:
    - B, M, K, N must satisfy M, K, N % 32 == 0.
"""

from typing import List, Tuple

import torch

from npu_model.software.program import Program, ASM_FOLDER
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction

VMEM_A = 0x2000
VMEM_B = 0x2400
VMEM_C0 = 0x2800
VMEM_C1 = 0x2C00

TILE = 32
FP8_BYTES = 1
BF16_BYTES = 2
TILE_BYTES_FP8 = TILE * TILE * FP8_BYTES    # 1024 B
TILE_BYTES_BF16 = TILE * TILE * BF16_BYTES  # 2048 B (two col-blocked halves)


def _tile_matrix(mat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Flatten (rows, cols) into tiled row-major tile order (each tile contiguous)."""
    row_tiles = rows // TILE
    col_tiles = cols // TILE
    parts = []
    for r in range(row_tiles):
        for c in range(col_tiles):
            parts.append(
                mat[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE].contiguous()
            )
    return torch.cat([p.reshape(-1) for p in parts])


def _colblock_bf16(mat: torch.Tensor, M: int, N: int) -> torch.Tensor:
    """Col-blocked layout: for each tile, H0 (cols 0-15) then H1 (cols 16-31)."""
    M_tiles = M // TILE
    N_tiles = N // TILE
    parts = []
    for r in range(M_tiles):
        for c in range(N_tiles):
            tile = mat[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE]
            parts.append(tile[:, : TILE // 2].contiguous())
            parts.append(tile[:, TILE // 2 :].contiguous())
    return torch.cat(parts, dim=0)


def batch_matmul_reference(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Hardware-faithful batch matmul: fp8 inputs, bf16 accumulation per K-tile.

    a: (B, M, K) fp8 — activations
    b: (B, K, N) fp8 — weights
    Returns: (B, M, N) bf16
    """
    B, M, K = a.shape
    N = b.shape[2]
    K_tiles = K // TILE
    results = []
    for bi in range(B):
        acc = None
        for k in range(K_tiles):
            a_k = a[bi, :, k * TILE : (k + 1) * TILE].to(torch.float16)
            b_k = b[bi, k * TILE : (k + 1) * TILE, :].to(torch.float16)
            if acc is None:
                acc = a_k @ b_k
            else:
                acc = acc.to(torch.bfloat16).to(torch.float16) + (a_k @ b_k)
        results.append(acc.to(torch.bfloat16))
    return torch.stack(results)


def _batch_expected_stacked(result: torch.Tensor, B: int, M: int, N: int) -> torch.Tensor:
    """Reorder (B, M, N) expected into the flat tiled col-blocked DRAM layout."""
    parts = []
    for bi in range(B):
        parts.append(_colblock_bf16(result[bi], M, N))
    return torch.cat(parts, dim=0)


def _make_program(B: int, M: int, K: int, N: int, seed: int):
    dram_a = 0x0000
    dram_b = dram_a + B * M * K * FP8_BYTES
    dram_c = dram_b + B * K * N * FP8_BYTES

    torch.manual_seed(seed)
    a = torch.randint(-8, 8, (B, M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
    b = torch.randint(-8, 8, (B, K, N), dtype=torch.int8).to(torch.float8_e4m3fn)
    expected = batch_matmul_reference(a, b)

    a_tiled_parts = [_tile_matrix(a[bi], M, K) for bi in range(B)]
    b_tiled_parts = [_tile_matrix(b[bi], K, N) for bi in range(B)]
    a_dram = torch.cat(a_tiled_parts)
    b_dram = torch.cat(b_tiled_parts)

    regions = [(dram_a, a_dram), (dram_b, b_dram)]
    golden = (dram_c, _batch_expected_stacked(expected, B, M, N))
    return regions, golden


# 2×32×32×32: 2 batches, 1×1 output tiles, 1 K-tile each

_2x32_regions, _2x32_golden = _make_program(2, 32, 32, 32, seed=300)


class ParameterizedBatchMatmul2x32x32x32Program(Program):
    """Batch matmul (B=2, M=K=N=32): 2 independent 32×32×32 fp8 matmuls."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_batch_matmul2x32x32x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _2x32_regions
    golden_result: tuple[int, torch.Tensor] = _2x32_golden


# 4×32×64×32: 4 batches, 1×1 output tiles, 2 K-tiles each

_4x32_regions, _4x32_golden = _make_program(4, 32, 64, 32, seed=301)


class ParameterizedBatchMatmul4x32x64x32Program(Program):
    """Batch matmul (B=4, M=32, K=64, N=32): 4 batches, K-accumulation path."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_batch_matmul4x32x64x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _4x32_regions
    golden_result: tuple[int, torch.Tensor] = _4x32_golden


# 2×64×32×64: 2 batches, 2×2 output tiles, 1 K-tile each

_2x64_regions, _2x64_golden = _make_program(2, 64, 32, 64, seed=302)


class ParameterizedBatchMatmul2x64x32x64Program(Program):
    """Batch matmul (B=2, M=64, K=32, N=64): 2 batches, multi-tile output."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_batch_matmul2x64x32x64.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _2x64_regions
    golden_result: tuple[int, torch.Tensor] = _2x64_golden
