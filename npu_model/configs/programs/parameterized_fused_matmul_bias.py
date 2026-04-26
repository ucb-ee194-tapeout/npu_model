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

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs

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


def make_fused_matmul_bias_instructions(
    M: int,
    N: int,
    dram_a: int,
    dram_b: int,
    dram_bias: int,
    dram_out: int,
) -> list[Instruction]:
    """Generate instructions for fused M×32×N matmul+bias (K=32 fixed).

    Uses hardware branch loops over M and N tile dimensions. K=32 means
    a single vmatmul.mxu0 per tile (no K accumulation loop).

    Scalar register map:
        x1  VMEM_A      x2  VMEM_B      x3  VMEM_BIAS     x4  VMEM_OUT
        x5  1024 (fp8)  x6  2048 (bf16)
        x7  VMEM_BIAS+1024              x8  VMEM_OUT+1024
        x9  M_tiles     x10 N_tiles
        x11 dram_b base x12 dram_bias base  x13 dram_out base
        x14 A_m_ptr     x15 B_n_ptr    x16 bias_ptr       x17 out_ptr
        x18 m counter   x19 n counter
        x20 stride_bias_m = N_tiles*2048   x21 stride_out_m = N_tiles*2048
    """
    assert M % TILE == 0 and N % TILE == 0
    M_tiles = M // TILE
    N_tiles = N // TILE

    nop = Instruction("addi", ScalarArgs(rd=0, rs1=0, imm=0))
    insns: list[Instruction] = []

    _emit_load_vmem_addr(1, VMEM_A, insns)
    _emit_load_vmem_addr(2, VMEM_B, insns)
    _emit_load_vmem_addr(3, VMEM_BIAS, insns)
    _emit_load_vmem_addr(4, VMEM_OUT, insns)
    _emit_load_imm32(5, FP8_TILE_BYTES, insns)
    _emit_load_imm32(6, BF16_TILE_BYTES, insns)
    _emit_load_imm32(7, VMEM_BIAS + HALF_BYTES, insns)
    _emit_load_imm32(8, VMEM_OUT + HALF_BYTES, insns)
    _emit_load_imm32(9, M_tiles, insns)
    _emit_load_imm32(10, N_tiles, insns)
    _emit_load_imm32(11, dram_b, insns)
    _emit_load_imm32(12, dram_bias, insns)
    _emit_load_imm32(13, dram_out, insns)
    _emit_load_imm32(14, dram_a, insns)          # A_m_ptr (initial)
    _emit_load_imm32(20, N_tiles * BF16_TILE_BYTES, insns)  # stride for bias and out per m
    insns.append(Instruction("addi", ScalarArgs(rd=18, rs1=0, imm=0)))  # m = 0

    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    # m-loop
    m_loop_start = len(insns)
    # Reset n-dimension pointers to beginning of this m row
    insns.append(Instruction("addi", ScalarArgs(rd=15, rs1=11, imm=0)))  # B_n_ptr = dram_b
    insns.append(Instruction("addi", ScalarArgs(rd=16, rs1=12, imm=0)))  # bias_ptr = dram_bias base for m
    insns.append(Instruction("addi", ScalarArgs(rd=17, rs1=13, imm=0)))  # out_ptr = dram_out base for m
    insns.append(Instruction("addi", ScalarArgs(rd=19, rs1=0, imm=0)))   # n = 0

    # n-loop
    n_loop_start = len(insns)
    # Fire A (ch0) and B (ch1). After A arrives, fire bias DMA and vload A+B
    # in the ~1028cy bias window, hiding 68cy of vload serial time.
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=14, rs2=5, channel=0)))
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=15, rs2=5, channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=3, rs1=16, rs2=6, channel=0)))  # bias queues behind B
    insns.append(Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)))   # A fp8 during B+bias window
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("vload", VectorArgs(vd=2, rs1=2, imm12=0)))   # B fp8 during bias window
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("vload", VectorArgs(vd=4, rs1=3, imm12=0)))   # bias H0
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("vload", VectorArgs(vd=5, rs1=3, imm12=32)))  # bias H1
    insns.append(Instruction("delay", ScalarArgs(imm=34)))

    insns.append(Instruction("vmatpush.weight.mxu0", MatrixArgs(vd=0, vs1=2)))
    insns.append(Instruction("delay", ScalarArgs(imm=32)))
    insns.append(Instruction("vmatmul.mxu0", MatrixArgs(vd=0, vs1=0, vs2=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=96)))
    insns.append(Instruction("vmatpop.bf16.acc.mxu0", MatrixArgs(vd=6, vs1=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=32)))

    insns.append(Instruction("vadd.bf16", VectorArgs(vd=8, vs1=6, vs2=4)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))

    insns.append(Instruction("vstore", VectorArgs(vd=8, rs1=4, imm12=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("vstore", VectorArgs(vd=9, rs1=4, imm12=32)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))

    insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=17, rs1=4, rs2=6, channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

    # Advance n: B moves by FP8_TILE, bias and out move by BF16_TILE
    insns.append(Instruction("addi", ScalarArgs(rd=15, rs1=15, imm=FP8_TILE_BYTES)))  # B_n_ptr += 1024
    insns.append(Instruction("add", ScalarArgs(rd=16, rs1=16, rs2=6)))                # bias_ptr += 2048
    insns.append(Instruction("add", ScalarArgs(rd=17, rs1=17, rs2=6)))                # out_ptr += 2048
    insns.append(Instruction("addi", ScalarArgs(rd=19, rs1=19, imm=1)))
    n_blt_idx = len(insns)
    insns.append(Instruction("blt", ScalarArgs(rs1=19, rs2=10, imm=n_loop_start - n_blt_idx)))
    insns.append(nop)
    insns.append(nop)

    # Advance m: A moves by FP8_TILE (one row block), bias and out by N_tiles*BF16_TILE
    insns.append(Instruction("addi", ScalarArgs(rd=14, rs1=14, imm=FP8_TILE_BYTES)))  # A_m_ptr += 1024
    insns.append(Instruction("add", ScalarArgs(rd=12, rs1=12, rs2=20)))  # dram_bias base += stride
    insns.append(Instruction("add", ScalarArgs(rd=13, rs1=13, rs2=20)))  # dram_out base += stride
    insns.append(Instruction("addi", ScalarArgs(rd=18, rs1=18, imm=1)))
    m_blt_idx = len(insns)
    insns.append(Instruction("blt", ScalarArgs(rs1=18, rs2=9, imm=m_loop_start - m_blt_idx)))
    insns.append(nop)
    insns.append(nop)

    return insns


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

    # Tile A row-blocks: for each M tile, the corresponding 32×32 fp8 block
    a_tiled = torch.cat([
        a[mt * TILE : (mt + 1) * TILE, :].contiguous()
        for mt in range(M_tiles)
    ])
    # Tile B col-blocks: for each N tile, the corresponding 32×32 fp8 block
    b_tiled = torch.cat([
        b[:, nt * TILE : (nt + 1) * TILE].contiguous()
        for nt in range(N_tiles)
    ])

    insns = make_fused_matmul_bias_instructions(
        M=M,
        N=N,
        dram_a=dram_a,
        dram_b=dram_b,
        dram_bias=dram_bias,
        dram_out=dram_out,
    )
    regions = [
        (dram_a, a_tiled),
        (dram_b, b_tiled),
        (dram_bias, _colblock_bf16(bias, M, N)),
    ]
    golden = (dram_out, _colblock_bf16(expected, M, N))
    return insns, regions, golden


_32_insns, _32_regions, _32_golden = _make_program(32, 32, seed=130)


class ParameterizedFusedMatmulBias32x32Program(Program):
    """fused_matmul_bias on a single 32×32 tile (K=32)."""

    instructions: List[Instruction[Any]] = _32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_insns, _64_regions, _64_golden = _make_program(64, 64, seed=131)


class ParameterizedFusedMatmulBias64x64Program(Program):
    """fused_matmul_bias on a 64×64 bf16 output tensor (K=32, 2×2 output tiles)."""

    instructions: List[Instruction[Any]] = _64_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_64x32_insns, _64x32_regions, _64x32_golden = _make_program(64, 32, seed=132)


class ParameterizedFusedMatmulBias64x32Program(Program):
    """fused_matmul_bias on a 64×32 bf16 output tensor (K=32, 2×1 output tiles)."""

    instructions: List[Instruction[Any]] = _64x32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64x32_regions
    golden_result: tuple[int, torch.Tensor] = _64x32_golden
