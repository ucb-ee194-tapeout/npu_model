"""Parameterized RMS-norm kernel: y = x * rsqrt(mean(x^2) + eps) for M×32.

N is fixed at 32. M must be a multiple of 32. Uses a hardware counted
loop over groups of 32 rows (M_groups iterations).

DRAM layout:
  dram_x:      M_groups * 2 * 1024 B  col-blocked input
  dram_inv_dim:             1024 B  1/32 broadcast constant
  dram_eps:                 1024 B  1e-6 broadcast constant
  dram_out:    M_groups * 2 * 1024 B  col-blocked output

VMEM slots:
  0x2000  VMEM_X_H0   1 KB
  0x2400  VMEM_X_H1   1 KB
  0x3000  VMEM_INVM   2 KB (H0 at 0x3000, H1 at 0x3400)
  0x4000  VMEM_EPS    2 KB (H0 at 0x4000, H1 at 0x4400)
  0x5000  VMEM_OUT    2 KB (H0 at 0x5000, H1 at 0x5400)

Scalar register map:
    x1  VMEM_X_H0    x2  VMEM_INV_DIM    x3  VMEM_EPS    x4  VMEM_OUT
    x5  HALF_BYTES (1024)    x6  VMEM_X_H1    x7  VMEM_INV_DIM_H1
    x8  VMEM_EPS_H1          x9  VMEM_OUT_H1
    x10  dram_x ptr (H0, advances by 2048 per group)
    x11  dram_out ptr (H0, advances by 2048 per group)
    x12  group counter    x13  M_groups (limit)    x14  2048 (group stride)
    x15  H1 input addr (computed per iter)    x16  H1 output addr (computed per iter)
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs

TILE = 32
BF16_BYTES = 2
HALF_BYTES = TILE * (TILE // 2) * BF16_BYTES  # 1024

VMEM_X = 0x2000
VMEM_INV_DIM = 0x3000
VMEM_EPS = 0x4000
VMEM_OUT = 0x5000


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


def _colblock_bf16(mat: torch.Tensor, M: int) -> torch.Tensor:
    parts = []
    for g in range(M // TILE):
        group = mat[g * TILE : (g + 1) * TILE, :]
        parts.append(group[:, : TILE // 2].contiguous())
        parts.append(group[:, TILE // 2 :].contiguous())
    return torch.cat(parts, dim=0)


def rms_norm_reference(
    x: torch.Tensor,
    inv_dim: float = 1.0 / 32.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    xb = x.to(torch.bfloat16)
    sq = (xb * xb).to(torch.bfloat16)
    row_sum = sq.sum(dim=-1, keepdim=True).to(torch.bfloat16)
    inv_dim_t = torch.full_like(row_sum, inv_dim, dtype=torch.bfloat16)
    eps_t = torch.full_like(row_sum, eps, dtype=torch.bfloat16)
    mean = (row_sum * inv_dim_t).to(torch.bfloat16)
    denom = (mean + eps_t).to(torch.bfloat16)
    root = torch.sqrt(denom.float()).to(torch.bfloat16)
    inv = (1.0 / root.float()).to(torch.bfloat16)
    return (xb * inv).to(x.dtype)


def make_rms_norm_instructions(
    M: int,
    dram_x: int,
    dram_inv_dim: int,
    dram_eps: int,
    dram_out: int,
) -> list[Instruction]:
    """Generate instructions for an M×32 RMS-norm."""
    assert M % TILE == 0
    M_groups = M // TILE
    nop = Instruction("addi", ScalarArgs(rd=0, rs1=0, imm=0))

    insns: list[Instruction] = []

    _emit_load_imm32(1, VMEM_X, insns)
    _emit_load_imm32(2, VMEM_INV_DIM, insns)
    _emit_load_imm32(3, VMEM_EPS, insns)
    _emit_load_imm32(4, VMEM_OUT, insns)
    _emit_load_imm32(5, HALF_BYTES, insns)
    _emit_load_imm32(6, VMEM_X + HALF_BYTES, insns)
    _emit_load_imm32(7, VMEM_INV_DIM + HALF_BYTES, insns)
    _emit_load_imm32(8, VMEM_EPS + HALF_BYTES, insns)
    _emit_load_imm32(9, VMEM_OUT + HALF_BYTES, insns)
    _emit_load_imm32(10, dram_x, insns)
    _emit_load_imm32(11, dram_out, insns)
    insns.append(Instruction("addi", ScalarArgs(rd=12, rs1=0, imm=0)))
    _emit_load_imm32(13, M_groups, insns)
    _emit_load_imm32(14, 2 * HALF_BYTES, insns)

    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    # Load inv_dim and eps into VMEM, then pre-load both into MRF before the loop.
    # inv_dim gets clobbered each iteration (reused for output Y), so it must be
    # reloaded from VMEM each group. eps is never overwritten; load it once here.
    _emit_load_imm32(15, dram_inv_dim, insns)
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=15, rs2=5, channel=0)))
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=7, rs1=15, rs2=5, channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    _emit_load_imm32(15, dram_eps, insns)
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=3, rs1=15, rs2=5, channel=0)))
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=8, rs1=15, rs2=5, channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    # v8/v9 = eps: loop-invariant, not overwritten by compute. Load once here.
    insns.append(Instruction("vload", VectorArgs(vd=8, rs1=3, imm12=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("vload", VectorArgs(vd=9, rs1=3, imm12=32)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))

    loop_start = len(insns)

    # H1 addresses = H0 + HALF_BYTES
    insns.append(Instruction("add", ScalarArgs(rd=15, rs1=10, rs2=5)))
    insns.append(Instruction("add", ScalarArgs(rd=16, rs1=11, rs2=5)))

    # Fire x DMA loads (H0 ch0, H1 ch1). Vload inv_dim into MRF while DMA runs.
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=10, rs2=5, channel=0)))
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=6, rs1=15, rs2=5, channel=1)))
    # inv_dim is already in VMEM; these vloads overlap with the x DMA transfers.
    insns.append(Instruction("vload", VectorArgs(vd=6, rs1=2, imm12=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("vload", VectorArgs(vd=7, rs1=2, imm12=32)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    # vload (v0, v1) = X
    insns.append(Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=32)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))

    # (v2, v3) = X^2
    insns.append(Instruction("vsquare.bf16", VectorArgs(vd=2, vs1=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))

    # (v4, v5) = row-sum(X^2)
    insns.append(Instruction("vredsum.row.bf16", VectorArgs(vd=4, vs1=2)))
    insns.append(Instruction("delay", ScalarArgs(imm=39)))

    # (v10, v11) = mean(X^2) = sum * inv_dim
    insns.append(Instruction("vmul.bf16", VectorArgs(vd=10, vs1=4, vs2=6)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))

    # (v12, v13) = mean + eps
    insns.append(Instruction("vadd.bf16", VectorArgs(vd=12, vs1=10, vs2=8)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))

    # (v14, v15) = sqrt(mean + eps)
    insns.append(Instruction("vsqrt.bf16", VectorArgs(vd=14, vs1=12)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))

    # (v4, v5) = inv_rms = 1/sqrt (reuse v4/v5)
    insns.append(Instruction("vrecip.bf16", VectorArgs(vd=4, vs1=14)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))

    # (v6, v7) = Y = X * inv_rms (reuse v6/v7)
    insns.append(Instruction("vmul.bf16", VectorArgs(vd=6, vs1=0, vs2=4)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))

    insns.append(Instruction("vstore", VectorArgs(vd=6, rs1=4, imm12=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("vstore", VectorArgs(vd=7, rs1=4, imm12=32)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))

    # DMA store H0 and H1 (x16 = H1 output addr, x9 = VMEM_OUT_H1)
    insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=11, rs1=4, rs2=5, channel=0)))
    insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=16, rs1=9, rs2=5, channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    insns.append(Instruction("add", ScalarArgs(rd=10, rs1=10, rs2=14)))
    insns.append(Instruction("add", ScalarArgs(rd=11, rs1=11, rs2=14)))
    insns.append(Instruction("addi", ScalarArgs(rd=12, rs1=12, imm=1)))

    blt_idx = len(insns)
    insns.append(Instruction("blt", ScalarArgs(rs1=12, rs2=13, imm=loop_start - blt_idx)))
    insns.append(nop)
    insns.append(nop)

    return insns


def _make_program(M: int, seed: int):
    M_groups = M // TILE
    dram_x = 0x0000
    dram_inv_dim = dram_x + M_groups * 2 * HALF_BYTES
    dram_eps = dram_inv_dim + HALF_BYTES
    dram_out = dram_eps + HALF_BYTES

    torch.manual_seed(seed)
    x = torch.randn(M, TILE, dtype=torch.bfloat16)
    expected = rms_norm_reference(x)

    inv_dim_tile = torch.full((TILE, TILE // 2), 1.0 / TILE, dtype=torch.bfloat16)
    eps_tile = torch.full((TILE, TILE // 2), 1e-6, dtype=torch.bfloat16)

    insns = make_rms_norm_instructions(
        M=M,
        dram_x=dram_x,
        dram_inv_dim=dram_inv_dim,
        dram_eps=dram_eps,
        dram_out=dram_out,
    )
    regions = [
        (dram_x, _colblock_bf16(x, M)),
        (dram_inv_dim, inv_dim_tile),
        (dram_eps, eps_tile),
    ]
    golden = (dram_out, _colblock_bf16(expected, M))
    return insns, regions, golden


_32_insns, _32_regions, _32_golden = _make_program(32, seed=90)


class ParameterizedRmsNorm32x32Program(Program):
    """RMS-norm on a single 32x32 bf16 tile."""

    instructions: List[Instruction[Any]] = _32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_insns, _64_regions, _64_golden = _make_program(64, seed=91)


class ParameterizedRmsNorm64x32Program(Program):
    """RMS-norm on a 64x32 bf16 tensor (2 groups)."""

    instructions: List[Instruction[Any]] = _64_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_96_insns, _96_regions, _96_golden = _make_program(96, seed=92)


class ParameterizedRmsNorm96x32Program(Program):
    """RMS-norm on a 96x32 bf16 tensor (3 groups)."""

    instructions: List[Instruction[Any]] = _96_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _96_regions
    golden_result: tuple[int, torch.Tensor] = _96_golden
