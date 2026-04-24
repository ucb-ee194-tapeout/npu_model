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

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs

# ── fixed VMEM addresses ──────────────────────────────────────────────────
VMEM_A = 0x2000
VMEM_B = 0x2400
VMEM_C0 = 0x2800
VMEM_C1 = 0x2C00

TILE = 32        # hardware tile size (rows and cols)
FP8_BYTES = 1    # bytes per fp8 element
BF16_BYTES = 2   # bytes per bf16 element
TILE_BYTES_FP8 = TILE * TILE * FP8_BYTES    # 1024 B
TILE_BYTES_BF16 = TILE * TILE * BF16_BYTES  # 2048 B — both halves


# ── helpers ───────────────────────────────────────────────────────────────

def _emit_load_imm32(rd: int, value: int, out: list[Instruction]) -> None:
    """Emit lui + addi to materialise an arbitrary 32-bit signed value in rd.

    Uses RISC-V two-instruction sequence; handles the sign-extension
    correction for values with bit 11 set.
    """
    if value == 0:
        out.append(Instruction("addi", ScalarArgs(rd=rd, rs1=0, imm=0)))
        return
    upper = (value + 0x800) >> 12   # pages, rounded so addi sign-extends correctly
    lower = value - (upper << 12)   # 12-bit signed remainder
    if upper:
        out.append(Instruction("lui", ScalarArgs(rd=rd, imm=upper)))
        if lower:
            out.append(Instruction("addi", ScalarArgs(rd=rd, rs1=rd, imm=lower)))
    else:
        out.append(Instruction("addi", ScalarArgs(rd=rd, rs1=0, imm=lower)))


def _emit_load_vmem_addr(rd: int, vmem_addr: int, out: list[Instruction]) -> None:
    """Materialise a fixed VMEM address into rd."""
    _emit_load_imm32(rd, vmem_addr, out)


def _a_offset(m_tile: int, k_tile: int, K: int) -> int:
    """Byte offset of A tile (m_tile, k_tile) in tiled DRAM layout.

    Tiles are stored sequentially in (m, k) order — tile (m, k) is at
    index (m * K_tiles + k), each occupying TILE_BYTES_FP8 contiguous bytes.
    """
    k_tiles = K // TILE
    return (m_tile * k_tiles + k_tile) * TILE_BYTES_FP8


def _b_offset(k_tile: int, n_tile: int, N: int) -> int:
    """Byte offset of B tile (k_tile, n_tile) in tiled DRAM layout.

    Tiles are stored sequentially in (k, n) order — tile (k, n) is at
    index (k * N_tiles + n), each occupying TILE_BYTES_FP8 contiguous bytes.
    """
    n_tiles = N // TILE
    return (k_tile * n_tiles + n_tile) * TILE_BYTES_FP8


def _c_offset(m_tile: int, n_tile: int, N: int) -> int:
    """Byte offset of tile (m_tile, n_tile) in the tiled DRAM output layout.

    Output tiles are stored sequentially in (m, n) order: tile (m, n)
    occupies a contiguous TILE_BYTES_BF16-byte slot at index (m * N_tiles + n).
    Within each slot the two 32×16 bf16 halves are back-to-back:
    half-0 (cols 0-15) followed immediately by half-1 (cols 16-31).
    This matches the order written by vstore(C0) then vstore(C1) and the
    layout produced by _expected_stacked.
    """
    n_tiles = N // TILE
    return (m_tile * n_tiles + n_tile) * TILE_BYTES_BF16


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


# ── instruction sequence generator ───────────────────────────────────────

def make_matmul_instructions(
    M: int,
    K: int,
    N: int,
    dram_a: int,
    dram_b: int,
    dram_c: int,
) -> list[Instruction]:
    """Generate the full instruction list for an M×K×N tiled matmul.

    Uses hardware branch loops over M, N, and K dimensions instead of
    Python-level unrolling. The first K-tile is peeled (uses vmatmul.mxu0
    to reset the accumulator); subsequent K-tiles run in a hardware loop
    using vmatmul.acc.mxu0.

    Scalar register map:
        x1   VMEM_A         x2   VMEM_B          x3  VMEM_C0       x4  VMEM_C1
        x5   1024 (fp8)     x6   2048 (bf16)
        x7   stride_B_k = N_tiles*1024            x8  stride_A_m = K_tiles*1024
        x9   stride_C_m = N_tiles*2048
        x10  M_tiles        x11  N_tiles          x12 K_tiles (if K_tiles>1)
        x13  dram_b base    x14  C_m_base         x15 A_m_base
        x16  B_n_base       x17  C_out_ptr        x18 A_k_ptr      x19 B_k_ptr
        x20  m counter      x21  n counter        x22 k counter (if K_tiles>1)
    """
    assert M % TILE == 0 and K % TILE == 0 and N % TILE == 0, (
        f"M={M}, K={K}, N={N} must all be multiples of {TILE}"
    )
    M_tiles = M // TILE
    K_tiles = K // TILE
    N_tiles = N // TILE

    nop = Instruction("addi", ScalarArgs(rd=0, rs1=0, imm=0))
    insns: list[Instruction] = []

    # Prologue: fixed VMEM addresses and tile strides
    _emit_load_vmem_addr(1, VMEM_A, insns)
    _emit_load_vmem_addr(2, VMEM_B, insns)
    _emit_load_vmem_addr(3, VMEM_C0, insns)
    _emit_load_vmem_addr(4, VMEM_C1, insns)
    insns.append(Instruction("addi", ScalarArgs(rd=5, rs1=0, imm=TILE_BYTES_FP8)))
    _emit_load_imm32(6, TILE_BYTES_BF16, insns)
    _emit_load_imm32(7, N_tiles * TILE_BYTES_FP8, insns)   # stride_B_k
    _emit_load_imm32(8, K_tiles * TILE_BYTES_FP8, insns)   # stride_A_m
    _emit_load_imm32(9, N_tiles * TILE_BYTES_BF16, insns)  # stride_C_m
    _emit_load_imm32(10, M_tiles, insns)
    _emit_load_imm32(11, N_tiles, insns)
    if K_tiles > 1:
        _emit_load_imm32(12, K_tiles, insns)
    _emit_load_imm32(13, dram_b, insns)
    _emit_load_imm32(14, dram_c, insns)   # C_m_base (initial)
    _emit_load_imm32(15, dram_a, insns)   # A_m_base (initial)
    insns.append(Instruction("addi", ScalarArgs(rd=20, rs1=0, imm=0)))  # m = 0

    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    # m-loop
    m_loop_start = len(insns)
    insns.append(Instruction("addi", ScalarArgs(rd=16, rs1=13, imm=0)))  # B_n_base = dram_b
    insns.append(Instruction("addi", ScalarArgs(rd=17, rs1=14, imm=0)))  # C_out_ptr = C_m_base
    insns.append(Instruction("addi", ScalarArgs(rd=21, rs1=0, imm=0)))   # n = 0

    # n-loop
    n_loop_start = len(insns)
    insns.append(Instruction("addi", ScalarArgs(rd=18, rs1=15, imm=0)))  # A_k_ptr = A_m_base
    insns.append(Instruction("addi", ScalarArgs(rd=19, rs1=16, imm=0)))  # B_k_ptr = B_n_base

    # Peeled k=0: reset accumulator
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
    insns.append(Instruction("addi", ScalarArgs(rd=18, rs1=18, imm=TILE_BYTES_FP8)))  # A_k_ptr++
    insns.append(Instruction("add", ScalarArgs(rd=19, rs1=19, rs2=7)))               # B_k_ptr++

    # k-loop for k=1..K_tiles-1 (skipped when K_tiles==1)
    if K_tiles > 1:
        insns.append(Instruction("addi", ScalarArgs(rd=22, rs1=0, imm=1)))  # k = 1
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
    insns.append(Instruction("addi", ScalarArgs(rd=16, rs1=16, imm=TILE_BYTES_FP8)))  # B_n_base += 1024
    insns.append(Instruction("add", ScalarArgs(rd=17, rs1=17, rs2=6)))                # C_out_ptr += 2048
    insns.append(Instruction("addi", ScalarArgs(rd=21, rs1=21, imm=1)))
    n_blt_idx = len(insns)
    insns.append(Instruction("blt", ScalarArgs(rs1=21, rs2=11, imm=n_loop_start - n_blt_idx)))
    insns.append(nop)
    insns.append(nop)

    # Advance m
    insns.append(Instruction("add", ScalarArgs(rd=15, rs1=15, rs2=8)))   # A_m_base += stride_A_m
    insns.append(Instruction("add", ScalarArgs(rd=14, rs1=14, rs2=9)))   # C_m_base += stride_C_m
    insns.append(Instruction("addi", ScalarArgs(rd=20, rs1=20, imm=1)))
    m_blt_idx = len(insns)
    insns.append(Instruction("blt", ScalarArgs(rs1=20, rs2=10, imm=m_loop_start - m_blt_idx)))
    insns.append(nop)
    insns.append(nop)

    return insns


# ── reference implementation ──────────────────────────────────────────────

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


# ── concrete 64×64×64 program (2×2 output tiles, 2 K-tiles each) ──────────

M, K, N = 64, 64, 64
M_TILES, K_TILES, N_TILES = M // TILE, K // TILE, N // TILE

DRAM_A = 0x0000
DRAM_B = DRAM_A + M * K * FP8_BYTES    # 0x1000  (M_tiles*K_tiles tiles of 1 KB each)
DRAM_C = DRAM_B + K * N * FP8_BYTES    # 0x2000

torch.manual_seed(7)
INPUT_A = torch.randint(-8, 8, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
INPUT_B = torch.randint(-8, 8, (K, N), dtype=torch.int8).to(torch.float8_e4m3fn)
EXPECTED = matmul_reference(INPUT_A, INPUT_B)

# A and B are stored in DRAM in tiled layout: each 32×32 tile is contiguous.
INPUT_A_TILED = _tile_matrix(INPUT_A, M, K)
INPUT_B_TILED = _tile_matrix(INPUT_B, K, N)

# Output in DRAM is tiled: (M_tiles × N_tiles) blocks, each block stored as
# [low_half(32×16 bf16), high_half(32×16 bf16)] back-to-back.
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


EXPECTED_DRAM = _expected_stacked(EXPECTED, M, N)


def _make_program(M: int, K: int, N: int, seed: int):
    """Build the (instructions, memory_regions, golden_result) triple for an M×K×N matmul."""
    dram_a = 0x0000
    dram_b = dram_a + M * K * FP8_BYTES
    dram_c = dram_b + K * N * FP8_BYTES

    torch.manual_seed(seed)
    a = torch.randint(-8, 8, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
    b = torch.randint(-8, 8, (K, N), dtype=torch.int8).to(torch.float8_e4m3fn)
    expected = matmul_reference(a, b)

    insns = make_matmul_instructions(M=M, K=K, N=N, dram_a=dram_a, dram_b=dram_b, dram_c=dram_c)
    regions = [(dram_a, _tile_matrix(a, M, K)), (dram_b, _tile_matrix(b, K, N))]
    golden = (dram_c, _expected_stacked(expected, M, N))
    return insns, regions, golden


class ParameterizedMatmulProgram(Program):
    """64×64×64 fp8 matmul — 2×2 output tiles, 2 K-tiles each.

    Demonstrates make_matmul_instructions() for any (M, K, N) that are
    multiples of 32. Inputs stored in tiled DRAM layout (see _tile_matrix).
    """

    instructions: List[Instruction[Any]] = make_matmul_instructions(
        M=M, K=K, N=N,
        dram_a=DRAM_A,
        dram_b=DRAM_B,
        dram_c=DRAM_C,
    )

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_A, INPUT_A_TILED),
        (DRAM_B, INPUT_B_TILED),
    ]

    golden_result: tuple[int, torch.Tensor] = (DRAM_C, EXPECTED_DRAM)


# ── 32×32×32: single tile, no K accumulation (matches SmolVLA matmul) ────────

_32_insns, _32_regions, _32_golden = _make_program(32, 32, 32, seed=1)


class ParameterizedMatmul32x32x32Program(Program):
    """32×32×32 fp8 matmul — 1×1 output tile, 1 K-tile.

    Minimal single-tile case equivalent to SmolVLAMatmulProgram but driven
    through make_matmul_instructions().  K=32 means the accumulator is reset
    once and popped immediately with no vmatmul.acc.mxu0.
    """

    instructions: List[Instruction[Any]] = _32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


# ── 32×64×32: single output tile, 2 K-tiles (K accumulation path) ────────────

_kchain_insns, _kchain_regions, _kchain_golden = _make_program(32, 64, 32, seed=2)


class ParameterizedMatmul32x64x32Program(Program):
    """32×64×32 fp8 matmul — 1×1 output tile, 2 K-tiles.

    Exercises the accumulator chaining path: first K-tile uses vmatmul.mxu0
    (reset), second uses vmatmul.acc.mxu0 (accumulate).  Equivalent to
    SmolVLAMatmulKChainProgram but generated rather than hand-written.
    """

    instructions: List[Instruction[Any]] = _kchain_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _kchain_regions
    golden_result: tuple[int, torch.Tensor] = _kchain_golden


# ── 64×32×96: 6 output tiles (2×3), single K-tile ────────────────────────────

_multi_insns, _multi_regions, _multi_golden = _make_program(64, 32, 96, seed=3)


class ParameterizedMatmul64x32x96Program(Program):
    """64×32×96 fp8 matmul — 2×3 output tiles, 1 K-tile.

    Exercises multi-tile output with no K accumulation: the generator must
    correctly compute DRAM addresses for all 6 output tiles and issue the
    right sequence of DMA stores without overlap.
    """

    instructions: List[Instruction[Any]] = _multi_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _multi_regions
    golden_result: tuple[int, torch.Tensor] = _multi_golden
