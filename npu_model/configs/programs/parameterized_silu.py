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

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs

TILE = 32
BF16_BYTES = 2
TILE_BYTES_BF16 = TILE * TILE * BF16_BYTES


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


def make_silu_instructions(
    M: int,
    N: int,
    dram_x: int,
    dram_out: int,
) -> list[Instruction]:
    """Generate the full instruction list for an M×N SiLU activation.

    MRF layout per tile:
        (v0, v1)   = x
        (v2, v3)   = -1.0 constant
        (v4, v5)   = +1.0 constant
        (v6, v7)   = -x
        (v8, v9)   = exp(-x)
        (v10, v11) = 1 + exp(-x)
        (v12, v13) = sigmoid(x) = 1/(1+exp(-x))
        (v14, v15) = silu(x) = x * sigmoid(x)
    """
    assert M % TILE == 0 and N % TILE == 0
    total_tiles = (M // TILE) * (N // TILE)
    nop = Instruction("addi", ScalarArgs(rd=0, rs1=0, imm=0))

    insns: list[Instruction] = []

    _emit_load_imm32(1, 0x2000, insns)
    _emit_load_imm32(2, 0x2800, insns)
    _emit_load_imm32(3, TILE_BYTES_BF16, insns)
    _emit_load_imm32(4, dram_x, insns)
    _emit_load_imm32(5, dram_out, insns)
    insns.append(Instruction("addi", ScalarArgs(rd=6, rs1=0, imm=0)))
    _emit_load_imm32(7, total_tiles, insns)

    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

    # v2-v5 are loop-invariant and not overwritten by compute; initialize once.
    insns.append(Instruction("vli.all", VectorArgs(vd=2, imm=-1)))
    insns.append(Instruction("delay", ScalarArgs(imm=65)))
    insns.append(Instruction("vli.all", VectorArgs(vd=3, imm=-1)))
    insns.append(Instruction("delay", ScalarArgs(imm=65)))
    insns.append(Instruction("vli.all", VectorArgs(vd=4, imm=1)))
    insns.append(Instruction("delay", ScalarArgs(imm=65)))
    insns.append(Instruction("vli.all", VectorArgs(vd=5, imm=1)))
    insns.append(Instruction("delay", ScalarArgs(imm=65)))

    loop_start = len(insns)

    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=4, rs2=3, channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

    insns.append(Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=32)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))

    insns.append(Instruction("vmul.bf16", VectorArgs(vd=6, vs1=0, vs2=2)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))
    insns.append(Instruction("vexp.bf16", VectorArgs(vd=8, vs1=6)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))
    insns.append(Instruction("vadd.bf16", VectorArgs(vd=10, vs1=8, vs2=4)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))
    insns.append(Instruction("vrecip.bf16", VectorArgs(vd=12, vs1=10)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))
    insns.append(Instruction("vmul.bf16", VectorArgs(vd=14, vs1=0, vs2=12)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))

    insns.append(Instruction("vstore", VectorArgs(vd=14, rs1=2, imm12=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("vstore", VectorArgs(vd=15, rs1=2, imm12=32)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))

    insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=5, rs1=2, rs2=3, channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

    insns.append(Instruction("add", ScalarArgs(rd=4, rs1=4, rs2=3)))
    insns.append(Instruction("add", ScalarArgs(rd=5, rs1=5, rs2=3)))
    insns.append(Instruction("addi", ScalarArgs(rd=6, rs1=6, imm=1)))

    blt_idx = len(insns)
    insns.append(Instruction("blt", ScalarArgs(rs1=6, rs2=7, imm=loop_start - blt_idx)))
    insns.append(nop)
    insns.append(nop)

    return insns


def silu_reference(x: torch.Tensor) -> torch.Tensor:
    return (x.float() * torch.sigmoid(x.float())).to(x.dtype)


def _make_program(M: int, N: int, seed: int):
    dram_x = 0x0000
    dram_out = dram_x + M * N * BF16_BYTES

    torch.manual_seed(seed)
    x = torch.randn(M, N, dtype=torch.bfloat16)
    expected = silu_reference(x)

    insns = make_silu_instructions(M=M, N=N, dram_x=dram_x, dram_out=dram_out)
    regions = [(dram_x, _tile_matrix_bf16(x, M, N))]
    golden = (dram_out, _tile_matrix_bf16(expected, M, N))
    return insns, regions, golden


_32_insns, _32_regions, _32_golden = _make_program(32, 32, seed=50)


class ParameterizedSilu32x32Program(Program):
    """SiLU on a single 32x32 bf16 tile."""

    instructions: List[Instruction[Any]] = _32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_insns, _64_regions, _64_golden = _make_program(64, 64, seed=51)


class ParameterizedSilu64x64Program(Program):
    """SiLU on a 64x64 bf16 tensor (2x2 tiles)."""

    instructions: List[Instruction[Any]] = _64_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_64x32_insns, _64x32_regions, _64x32_golden = _make_program(64, 32, seed=52)


class ParameterizedSilu64x32Program(Program):
    """SiLU on a 64x32 bf16 tensor (2x1 tiles)."""

    instructions: List[Instruction[Any]] = _64x32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64x32_regions
    golden_result: tuple[int, torch.Tensor] = _64x32_golden
