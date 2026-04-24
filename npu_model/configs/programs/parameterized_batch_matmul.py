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

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs

VMEM_A = 0x2000
VMEM_B = 0x2400
VMEM_C0 = 0x2800
VMEM_C1 = 0x2C00

TILE = 32
FP8_BYTES = 1
BF16_BYTES = 2
TILE_BYTES_FP8 = TILE * TILE * FP8_BYTES    # 1024 B
TILE_BYTES_BF16 = TILE * TILE * BF16_BYTES  # 2048 B (two col-blocked halves)


def _emit_load_imm32(rd: int, value: int, out: list[Instruction]) -> None:
    if value == 0:
        out.append(Instruction("addi", ScalarArgs(rd=rd, rs1=0, imm=0)))
        return
    upper = (value + 0x800) >> 12
    lower = value - (upper << 12)
    if upper:
        out.append(Instruction("lui", ScalarArgs(rd=rd, imm=upper)))
        if lower:
            out.append(Instruction("addi", ScalarArgs(rd=rd, rs1=rd, imm=lower)))
    else:
        out.append(Instruction("addi", ScalarArgs(rd=rd, rs1=0, imm=lower)))


def _emit_load_vmem_addr(rd: int, vmem_addr: int, out: list[Instruction]) -> None:
    _emit_load_imm32(rd, vmem_addr, out)


def _a_tile_offset(b: int, m: int, k: int, M: int, K: int) -> int:
    M_tiles = M // TILE
    K_tiles = K // TILE
    return (b * M_tiles * K_tiles + m * K_tiles + k) * TILE_BYTES_FP8


def _b_tile_offset(b: int, k: int, n: int, K: int, N: int) -> int:
    K_tiles = K // TILE
    N_tiles = N // TILE
    return (b * K_tiles * N_tiles + k * N_tiles + n) * TILE_BYTES_FP8


def _c_tile_offset(b: int, m: int, n: int, M: int, N: int) -> int:
    M_tiles = M // TILE
    N_tiles = N // TILE
    return (b * M_tiles * N_tiles + m * N_tiles + n) * TILE_BYTES_BF16


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


def make_batch_matmul_instructions(
    B: int,
    M: int,
    K: int,
    N: int,
    dram_a: int,
    dram_b: int,
    dram_c: int,
) -> list[Instruction]:
    """Generate instructions for a B-batched M×K×N fp8 tiled matmul.

    Uses hardware branch loops over B, M, N, and K dimensions. The first
    K-tile is peeled (vmatmul.mxu0 to reset acc); subsequent K-tiles loop
    with vmatmul.acc.mxu0. The batch loop resets the m×n×k loop pointers.

    Scalar register map:
        x1  VMEM_A      x2  VMEM_B      x3  VMEM_C0     x4  VMEM_C1
        x5  1024(fp8)   x6  2048(bf16)
        x7  stride_B_k = N_tiles*1024   x8  stride_A_m = K_tiles*1024
        x9  stride_C_m = N_tiles*2048
        x10 M_tiles     x11 N_tiles     x12 K_tiles (if K_tiles>1)
        x13 B_n_base    x14 C_m_base    x15 A_m_base
        x16 B_n_ptr     x17 C_out_ptr   x18 A_k_ptr     x19 B_k_ptr
        x20 m counter   x21 n counter   x22 k counter (if K_tiles>1)
        x23 B_batches   x24 b counter
        x25 stride_A_b  x26 stride_B_b  x27 stride_C_b
        x28 A_batch_base x29 B_batch_base x30 C_batch_base
    """
    assert M % TILE == 0 and K % TILE == 0 and N % TILE == 0
    M_tiles = M // TILE
    K_tiles = K // TILE
    N_tiles = N // TILE

    nop = Instruction("addi", ScalarArgs(rd=0, rs1=0, imm=0))
    insns: list[Instruction] = []

    _emit_load_vmem_addr(1, VMEM_A, insns)
    _emit_load_vmem_addr(2, VMEM_B, insns)
    _emit_load_vmem_addr(3, VMEM_C0, insns)
    _emit_load_vmem_addr(4, VMEM_C1, insns)
    insns.append(Instruction("addi", ScalarArgs(rd=5, rs1=0, imm=TILE_BYTES_FP8)))
    _emit_load_imm32(6, TILE_BYTES_BF16, insns)
    _emit_load_imm32(7, N_tiles * TILE_BYTES_FP8, insns)          # stride_B_k
    _emit_load_imm32(8, K_tiles * TILE_BYTES_FP8, insns)          # stride_A_m
    _emit_load_imm32(9, N_tiles * TILE_BYTES_BF16, insns)         # stride_C_m
    _emit_load_imm32(10, M_tiles, insns)
    _emit_load_imm32(11, N_tiles, insns)
    if K_tiles > 1:
        _emit_load_imm32(12, K_tiles, insns)
    _emit_load_imm32(23, B, insns)
    _emit_load_imm32(25, M_tiles * K_tiles * TILE_BYTES_FP8, insns)  # stride_A_b
    _emit_load_imm32(26, K_tiles * N_tiles * TILE_BYTES_FP8, insns)  # stride_B_b
    _emit_load_imm32(27, M_tiles * N_tiles * TILE_BYTES_BF16, insns) # stride_C_b
    _emit_load_imm32(28, dram_a, insns)  # A_batch_base
    _emit_load_imm32(29, dram_b, insns)  # B_batch_base
    _emit_load_imm32(30, dram_c, insns)  # C_batch_base
    insns.append(Instruction("addi", ScalarArgs(rd=24, rs1=0, imm=0)))  # b = 0

    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    # b-loop
    b_loop_start = len(insns)
    # Reset m×n×k pointers for this batch element
    insns.append(Instruction("addi", ScalarArgs(rd=15, rs1=28, imm=0)))  # A_m_base = A_batch_base
    insns.append(Instruction("addi", ScalarArgs(rd=14, rs1=30, imm=0)))  # C_m_base = C_batch_base
    insns.append(Instruction("addi", ScalarArgs(rd=13, rs1=29, imm=0)))  # B_n_base_src = B_batch_base
    insns.append(Instruction("addi", ScalarArgs(rd=20, rs1=0, imm=0)))   # m = 0

    # m-loop
    m_loop_start = len(insns)
    insns.append(Instruction("addi", ScalarArgs(rd=16, rs1=13, imm=0)))  # B_n_ptr = B base
    insns.append(Instruction("addi", ScalarArgs(rd=17, rs1=14, imm=0)))  # C_out_ptr = C_m_base
    insns.append(Instruction("addi", ScalarArgs(rd=21, rs1=0, imm=0)))   # n = 0

    # n-loop
    n_loop_start = len(insns)
    insns.append(Instruction("addi", ScalarArgs(rd=18, rs1=15, imm=0)))  # A_k_ptr = A_m_base
    insns.append(Instruction("addi", ScalarArgs(rd=19, rs1=16, imm=0)))  # B_k_ptr = B_n_ptr

    # Peeled k=0
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=18, rs2=5, channel=0)))
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=19, rs2=5, channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("vload", VectorArgs(vd=0, rs1=1)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("vload", VectorArgs(vd=1, rs1=2)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("vmatpush.weight.mxu0", VectorArgs(vs1=1)))
    insns.append(Instruction("delay", ScalarArgs(imm=32)))
    insns.append(Instruction("vmatmul.mxu0", MatrixArgs(vs1=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=96)))
    insns.append(Instruction("addi", ScalarArgs(rd=18, rs1=18, imm=TILE_BYTES_FP8)))
    insns.append(Instruction("add", ScalarArgs(rd=19, rs1=19, rs2=7)))

    # k-loop for k=1..K_tiles-1
    if K_tiles > 1:
        insns.append(Instruction("addi", ScalarArgs(rd=22, rs1=0, imm=1)))
        k_loop_start = len(insns)
        insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=18, rs2=5, channel=0)))
        insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=19, rs2=5, channel=1)))
        insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
        insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))
        insns.append(Instruction("vload", VectorArgs(vd=0, rs1=1)))
        insns.append(Instruction("delay", ScalarArgs(imm=34)))
        insns.append(Instruction("vload", VectorArgs(vd=1, rs1=2)))
        insns.append(Instruction("delay", ScalarArgs(imm=34)))
        insns.append(Instruction("vmatpush.weight.mxu0", VectorArgs(vs1=1)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))
        insns.append(Instruction("vmatmul.acc.mxu0", MatrixArgs(vs1=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=96)))
        insns.append(Instruction("addi", ScalarArgs(rd=18, rs1=18, imm=TILE_BYTES_FP8)))
        insns.append(Instruction("add", ScalarArgs(rd=19, rs1=19, rs2=7)))
        insns.append(Instruction("addi", ScalarArgs(rd=22, rs1=22, imm=1)))
        blt_idx = len(insns)
        insns.append(Instruction("blt", ScalarArgs(rs1=22, rs2=12, imm=k_loop_start - blt_idx)))
        insns.append(nop)
        insns.append(nop)

    # Pop accumulator, store output tile
    insns.append(Instruction("vmatpop.bf16.acc.mxu0", VectorArgs(vd=2)))
    insns.append(Instruction("delay", ScalarArgs(imm=32)))
    insns.append(Instruction("vstore", VectorArgs(vd=2, rs1=3)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("vstore", VectorArgs(vd=3, rs1=4)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=17, rs1=3, rs2=6, channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

    # Advance n
    insns.append(Instruction("addi", ScalarArgs(rd=16, rs1=16, imm=TILE_BYTES_FP8)))
    insns.append(Instruction("add", ScalarArgs(rd=17, rs1=17, rs2=6)))
    insns.append(Instruction("addi", ScalarArgs(rd=21, rs1=21, imm=1)))
    n_blt_idx = len(insns)
    insns.append(Instruction("blt", ScalarArgs(rs1=21, rs2=11, imm=n_loop_start - n_blt_idx)))
    insns.append(nop)
    insns.append(nop)

    # Advance m
    insns.append(Instruction("add", ScalarArgs(rd=15, rs1=15, rs2=8)))
    insns.append(Instruction("add", ScalarArgs(rd=14, rs1=14, rs2=9)))
    insns.append(Instruction("addi", ScalarArgs(rd=20, rs1=20, imm=1)))
    m_blt_idx = len(insns)
    insns.append(Instruction("blt", ScalarArgs(rs1=20, rs2=10, imm=m_loop_start - m_blt_idx)))
    insns.append(nop)
    insns.append(nop)

    # Advance b
    insns.append(Instruction("add", ScalarArgs(rd=28, rs1=28, rs2=25)))
    insns.append(Instruction("add", ScalarArgs(rd=29, rs1=29, rs2=26)))
    insns.append(Instruction("add", ScalarArgs(rd=30, rs1=30, rs2=27)))
    insns.append(Instruction("addi", ScalarArgs(rd=24, rs1=24, imm=1)))
    b_blt_idx = len(insns)
    insns.append(Instruction("blt", ScalarArgs(rs1=24, rs2=23, imm=b_loop_start - b_blt_idx)))
    insns.append(nop)
    insns.append(nop)

    return insns


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

    # Tile each batch element's A and B
    a_tiled_parts = [_tile_matrix(a[bi], M, K) for bi in range(B)]
    b_tiled_parts = [_tile_matrix(b[bi], K, N) for bi in range(B)]
    a_dram = torch.cat(a_tiled_parts)
    b_dram = torch.cat(b_tiled_parts)

    insns = make_batch_matmul_instructions(
        B=B, M=M, K=K, N=N,
        dram_a=dram_a, dram_b=dram_b, dram_c=dram_c,
    )
    regions = [(dram_a, a_dram), (dram_b, b_dram)]
    golden = (dram_c, _batch_expected_stacked(expected, B, M, N))
    return insns, regions, golden


# ── 2×32×32×32: 2 batches, 1×1 output tiles, 1 K-tile each ──────────────────

_2x32_insns, _2x32_regions, _2x32_golden = _make_program(2, 32, 32, 32, seed=300)


class ParameterizedBatchMatmul2x32x32x32Program(Program):
    """Batch matmul (B=2, M=K=N=32): 2 independent 32×32×32 fp8 matmuls."""

    instructions: List[Instruction[Any]] = _2x32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _2x32_regions
    golden_result: tuple[int, torch.Tensor] = _2x32_golden


# ── 4×32×64×32: 4 batches, 1×1 output tiles, 2 K-tiles each ─────────────────

_4x32_insns, _4x32_regions, _4x32_golden = _make_program(4, 32, 64, 32, seed=301)


class ParameterizedBatchMatmul4x32x64x32Program(Program):
    """Batch matmul (B=4, M=32, K=64, N=32): 4 batches, K-accumulation path."""

    instructions: List[Instruction[Any]] = _4x32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _4x32_regions
    golden_result: tuple[int, torch.Tensor] = _4x32_golden


# ── 2×64×32×64: 2 batches, 2×2 output tiles, 1 K-tile each ──────────────────

_2x64_insns, _2x64_regions, _2x64_golden = _make_program(2, 64, 32, 64, seed=302)


class ParameterizedBatchMatmul2x64x32x64Program(Program):
    """Batch matmul (B=2, M=64, K=32, N=64): 2 batches, multi-tile output."""

    instructions: List[Instruction[Any]] = _2x64_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _2x64_regions
    golden_result: tuple[int, torch.Tensor] = _2x64_golden
