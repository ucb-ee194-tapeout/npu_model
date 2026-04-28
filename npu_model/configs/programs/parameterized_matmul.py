"""Parameterized matmul kernel: C = A @ B for arbitrary M×K×N.

Tiles A (M×K fp8) and B (K×N fp8) into 32×32 chunks and accumulates
across the K dimension using the MXU0 accumulator chaining pattern:

    vmatmul.mxu0       — first K-tile  (resets accumulator)
    vmatmul.acc.mxu0   — subsequent K-tiles (adds to accumulator)

Constraints:
    - M, K, N must be multiples of 32
    - A is stored in DRAM in tiled layout: (M_tiles × K_tiles) contiguous 32×32 fp8 blocks
    - B is stored in DRAM in tiled layout: (K_tiles × N_tiles) contiguous 32×32 fp8 blocks
    - C is written to DRAM in tiled layout: (M_tiles × N_tiles) contiguous blocks,
      each block stored as two 32×16 bf16 halves (cols 0-15, then cols 16-31)

The tiled layout means tile (m, k) of A starts at byte:
    (m * K_tiles + k) * TILE_BYTES_FP8

This differs from row-major, where tile rows are interleaved with adjacent-column
data. Tiled storage makes each DMA transfer a single contiguous 1 KB load.

VMEM uses four fixed slots (reused across all tiles):
    VMEM_A   = 0x2000   1 KB — current A tile
    VMEM_B   = 0x2400   1 KB — current B tile
    VMEM_C0  = 0x2800   1 KB — C tile low  half  (cols  0-15)
    VMEM_C1  = 0x2C00   1 KB — C tile high half  (cols 16-31)
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
TILE_BYTES_BF16 = TILE * TILE * BF16_BYTES  # 2048 B — both halves


def _tile_matrix(mat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Rearrange a (rows, cols) matrix into a flat tiled layout.

    Extracts each (TILE, TILE) block in (m, k) or (k, n) row-major tile order
    and concatenates them so each tile is contiguous in memory — required for
    a flat DMA load to land the right data in VMEM.
    """
    row_tiles = rows // TILE
    col_tiles = cols // TILE
    parts = []
    for r in range(row_tiles):
        for c in range(col_tiles):
            parts.append(
                mat[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE].contiguous()
            )
    return torch.cat([p.reshape(-1) for p in parts])


def matmul_reference(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Hardware-faithful reference: fp8 inputs, bf16 accumulation per K-tile."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    K_tiles = K // TILE

    acc = torch.zeros(M, N, dtype=torch.float16)
    for k in range(K_tiles):
        a_k = a[:, k * TILE : (k + 1) * TILE].to(torch.float16)
        b_k = b[k * TILE : (k + 1) * TILE, :].to(torch.float16)
        if k == 0:
            acc = a_k @ b_k
        else:
            acc = (acc.to(torch.bfloat16).to(torch.float16)) + (a_k @ b_k)
    return acc.to(torch.bfloat16)


def _expected_stacked(result: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Reorder expected into the DRAM tiled layout the kernel writes."""
    row_tiles = rows // TILE
    col_tiles = cols // TILE
    parts = []
    for m in range(row_tiles):
        for n in range(col_tiles):
            tile = result[m * TILE:(m + 1) * TILE, n * TILE:(n + 1) * TILE]
            parts.append(tile[:, :TILE // 2])   # cols 0-15
            parts.append(tile[:, TILE // 2:])   # cols 16-31
    return torch.cat(parts, dim=0)


# 64×64×64: 2×2 output tiles, 2 K-tiles each

M, K, N = 64, 64, 64

DRAM_A = 0x0000
DRAM_B = DRAM_A + M * K * FP8_BYTES    # 0x1000
DRAM_C = DRAM_B + K * N * FP8_BYTES    # 0x2000

torch.manual_seed(7)
INPUT_A = torch.randint(-8, 8, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
INPUT_B = torch.randint(-8, 8, (K, N), dtype=torch.int8).to(torch.float8_e4m3fn)
EXPECTED = matmul_reference(INPUT_A, INPUT_B)

INPUT_A_TILED = _tile_matrix(INPUT_A, M, K)
INPUT_B_TILED = _tile_matrix(INPUT_B, K, N)
EXPECTED_DRAM = _expected_stacked(EXPECTED, M, N)


def _make_program(M: int, K: int, N: int, seed: int):
    """Build the (memory_regions, golden_result) pair for an M×K×N matmul."""
    dram_a = 0x0000
    dram_b = dram_a + M * K * FP8_BYTES
    dram_c = dram_b + K * N * FP8_BYTES

    torch.manual_seed(seed)
    a = torch.randint(-8, 8, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
    b = torch.randint(-8, 8, (K, N), dtype=torch.int8).to(torch.float8_e4m3fn)
    expected = matmul_reference(a, b)

    regions = [(dram_a, _tile_matrix(a, M, K)), (dram_b, _tile_matrix(b, K, N))]
    golden = (dram_c, _expected_stacked(expected, M, N))
    return regions, golden


class ParameterizedMatmulProgram(Program):
    """64×64×64 fp8 matmul — 2×2 output tiles, 2 K-tiles each."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_matmul.S')

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_A, INPUT_A_TILED),
        (DRAM_B, INPUT_B_TILED),
    ]

    golden_result: tuple[int, torch.Tensor] = (DRAM_C, EXPECTED_DRAM)


# 32×32×32: single tile, no K accumulation

_32_regions, _32_golden = _make_program(32, 32, 32, seed=1)


class ParameterizedMatmul32x32x32Program(Program):
    """32×32×32 fp8 matmul — 1×1 output tile, 1 K-tile."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_matmul32x32x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


# 32×64×32: single output tile, 2 K-tiles (K accumulation path)

_kchain_regions, _kchain_golden = _make_program(32, 64, 32, seed=2)


class ParameterizedMatmul32x64x32Program(Program):
    """32×64×32 fp8 matmul — 1×1 output tile, 2 K-tiles."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_matmul32x64x32.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _kchain_regions
    golden_result: tuple[int, torch.Tensor] = _kchain_golden


# 64×32×96: 6 output tiles (2×3), single K-tile

_multi_regions, _multi_golden = _make_program(64, 32, 96, seed=3)


class ParameterizedMatmul64x32x96Program(Program):
    """64×32×96 fp8 matmul — 2×3 output tiles, 1 K-tile."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_matmul64x32x96.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _multi_regions
    golden_result: tuple[int, torch.Tensor] = _multi_golden
