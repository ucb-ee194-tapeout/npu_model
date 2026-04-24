"""Parameterized elementwise add kernel: C = A + B for arbitrary M×N.

Tiles both inputs and output into 32x32 bf16 blocks stored row-major in
DRAM (2048 B per tile). Uses a hardware counted loop over all tiles.

Constraints:
    - M and N must be multiples of 32.

VMEM slots (fixed, reused across all tiles):
    VMEM_A  = 0x2000   2 KB
    VMEM_B  = 0x2800   2 KB
    VMEM_C  = 0x3000   2 KB

Scalar register map:
    x1  VMEM_A    x2  VMEM_B    x3  VMEM_C
    x4  TILE_BYTES_BF16 (2048, also used as pointer stride)
    x5  dram_a pointer    x6  dram_b pointer    x7  dram_c pointer
    x8  loop counter      x9  total_tiles (loop limit)
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs

VMEM_A = 0x2000
VMEM_B = 0x2800
VMEM_C = 0x3000

TILE = 32
BF16_BYTES = 2
TILE_BYTES_BF16 = TILE * TILE * BF16_BYTES  # 2048


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


def _tile_matrix_bf16(mat: torch.Tensor, M: int, N: int) -> torch.Tensor:
    M_tiles = M // TILE
    N_tiles = N // TILE
    parts = []
    for r in range(M_tiles):
        for c in range(N_tiles):
            tile = mat[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE].contiguous()
            parts.append(tile.reshape(-1))
    return torch.cat(parts)


def make_elementwise_add_instructions(
    M: int,
    N: int,
    dram_a: int,
    dram_b: int,
    dram_c: int,
) -> list[Instruction]:
    """Generate the full instruction list for an M×N elementwise add.

    Uses a hardware counted loop: counter x8 runs 0..total_tiles-1.
    DRAM pointers x5, x6, x7 advance by TILE_BYTES_BF16 each iteration.
    """
    assert M % TILE == 0 and N % TILE == 0
    total_tiles = (M // TILE) * (N // TILE)
    nop = Instruction("addi", ScalarArgs(rd=0, rs1=0, imm=0))

    insns: list[Instruction] = []

    _emit_load_imm32(1, 0x2000, insns)
    _emit_load_imm32(2, 0x2800, insns)
    _emit_load_imm32(3, 0x3000, insns)
    _emit_load_imm32(4, TILE_BYTES_BF16, insns)
    _emit_load_imm32(5, dram_a, insns)
    _emit_load_imm32(6, dram_b, insns)
    _emit_load_imm32(7, dram_c, insns)
    insns.append(Instruction("addi", ScalarArgs(rd=8, rs1=0, imm=0)))
    _emit_load_imm32(9, total_tiles, insns)

    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    loop_start = len(insns)

    # DMA A and B in parallel
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=5, rs2=4, channel=0)))
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=6, rs2=4, channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    insns.append(Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=32)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("vload", VectorArgs(vd=2, rs1=2, imm12=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("vload", VectorArgs(vd=3, rs1=2, imm12=32)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))

    insns.append(Instruction("vadd.bf16", VectorArgs(vd=4, vs1=0, vs2=2)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))

    insns.append(Instruction("vstore", VectorArgs(vd=4, rs1=3, imm12=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("vstore", VectorArgs(vd=5, rs1=3, imm12=32)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))

    insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=7, rs1=3, rs2=4, channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

    # Advance DRAM pointers and loop counter
    insns.append(Instruction("add", ScalarArgs(rd=5, rs1=5, rs2=4)))
    insns.append(Instruction("add", ScalarArgs(rd=6, rs1=6, rs2=4)))
    insns.append(Instruction("add", ScalarArgs(rd=7, rs1=7, rs2=4)))
    insns.append(Instruction("addi", ScalarArgs(rd=8, rs1=8, imm=1)))

    blt_idx = len(insns)
    insns.append(Instruction("blt", ScalarArgs(rs1=8, rs2=9, imm=loop_start - blt_idx)))
    insns.append(nop)
    insns.append(nop)

    return insns


def elementwise_add_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() + b.float()).to(a.dtype)


def _make_program(M: int, N: int, seed: int):
    dram_a = 0x0000
    dram_b = dram_a + M * N * BF16_BYTES
    dram_c = dram_b + M * N * BF16_BYTES

    torch.manual_seed(seed)
    a = torch.randn(M, N, dtype=torch.bfloat16)
    b = torch.randn(M, N, dtype=torch.bfloat16)
    expected = elementwise_add_reference(a, b)

    insns = make_elementwise_add_instructions(M=M, N=N, dram_a=dram_a, dram_b=dram_b, dram_c=dram_c)
    regions = [
        (dram_a, _tile_matrix_bf16(a, M, N)),
        (dram_b, _tile_matrix_bf16(b, M, N)),
    ]
    golden = (dram_c, _tile_matrix_bf16(expected, M, N))
    return insns, regions, golden


_32_insns, _32_regions, _32_golden = _make_program(32, 32, seed=10)


class ParameterizedElementwiseAdd32x32Program(Program):
    """Elementwise add on a single 32x32 bf16 tile."""

    instructions: List[Instruction[Any]] = _32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_insns, _64_regions, _64_golden = _make_program(64, 64, seed=11)


class ParameterizedElementwiseAdd64x64Program(Program):
    """Elementwise add on a 64x64 bf16 tensor (2x2 tiles)."""

    instructions: List[Instruction[Any]] = _64_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_64x32_insns, _64x32_regions, _64x32_golden = _make_program(64, 32, seed=12)


class ParameterizedElementwiseAdd64x32Program(Program):
    """Elementwise add on a 64x32 bf16 tensor (2x1 tiles)."""

    instructions: List[Instruction[Any]] = _64x32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64x32_regions
    golden_result: tuple[int, torch.Tensor] = _64x32_golden
