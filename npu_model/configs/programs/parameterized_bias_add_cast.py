"""Parameterized bias-add-cast kernel: fp8(x + bias) for 32×32.

Fuses an elementwise bf16 addition (x + bias) with an fp8 quantization
(unit-scale bf16 → fp8_e4m3fn cast via vpack.bf16.fp8).

DRAM layout (per _make_program):
  [dram_x   ]  2 × 1024 B  — col-blocked bf16 x   (H0 then H1)
  [dram_bias ]  2 × 1024 B  — col-blocked bf16 bias (H0 then H1)
  [dram_out  ]      1024 B  — fp8 output tile (32×32 × 1 B)

VMEM slots:
  0x2000  VMEM_X      2 KB  — x tile (H0 at 0x2000, H1 at 0x2400 via imm12=32)
  0x2800  VMEM_BIAS   2 KB  — bias tile (H0 at 0x2800, H1 at 0x2C00 via imm12=32)
  0x3000  VMEM_OUT    1 KB  — fp8 output

MRF layout per tile:
  (v0, v1) = x bf16 LMUL=2 pair
  (v2, v3) = bias bf16 LMUL=2 pair
  (v4, v5) = x + bias  (vadd.bf16)
  v6       = fp8(x + bias)  (vpack.bf16.fp8)

Constraint: single 32×32 tile only (M=N=32).
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs

TILE = 32
BF16_BYTES = 2
FP8_BYTES = 1
HALF_BYTES = TILE * (TILE // 2) * BF16_BYTES  # 1024 B (32×16 bf16)
FP8_TILE_BYTES = TILE * TILE * FP8_BYTES       # 1024 B (32×32 fp8)
BF16_TILE_BYTES = TILE * TILE * BF16_BYTES     # 2048 B (two halves)

VMEM_X = 0x2000
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


def _colblock_bf16(mat: torch.Tensor) -> torch.Tensor:
    """Pack 32×32 bf16 into col-blocked layout: H0 (32×16) then H1 (32×16)."""
    assert mat.shape == (TILE, TILE)
    h0 = mat[:, : TILE // 2].contiguous()
    h1 = mat[:, TILE // 2 :].contiguous()
    return torch.cat([h0, h1], dim=0)  # (64, 16) bf16 = 2048 B


def _tile_fp8(mat: torch.Tensor) -> torch.Tensor:
    """Flatten 32×32 fp8 tile to 1-D (row-major)."""
    assert mat.shape == (TILE, TILE)
    return mat.reshape(-1)


def bias_add_cast_reference(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """fp8(x + bias) with unit scale — matches vpack.bf16.fp8 seli=1 path."""
    return (x.float() + bias.float()).to(torch.float8_e4m3fn)


def make_bias_add_cast_instructions(
    dram_x: int,
    dram_bias: int,
    dram_out: int,
) -> list[Instruction]:
    """Generate instructions for a single 32×32 bias-add-cast tile.

    Scalar register map:
        x1  VMEM_X    x2  VMEM_BIAS    x3  VMEM_OUT
        x4  BF16_TILE_BYTES (2048)     x5  HALF_BYTES (1024)
    ERF: seli rd=0, imm=1  → scale register 0 = 1.0 (unit scale)
    Scratch: x6 (dram_x addr), x7 (dram_bias addr), x8 (dram_out addr)
    """
    insns: list[Instruction] = []

    _emit_load_vmem_addr(1, VMEM_X, insns)
    _emit_load_vmem_addr(2, VMEM_BIAS, insns)
    _emit_load_vmem_addr(3, VMEM_OUT, insns)
    _emit_load_imm32(4, BF16_TILE_BYTES, insns)
    _emit_load_imm32(5, HALF_BYTES, insns)

    # Unit scale in ERF register 0
    insns.append(Instruction("seli", ScalarArgs(rd=0, imm=1)))

    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    # Fire both DMA loads; vload x halves while bias is still in flight to hide 68cy.
    _emit_load_imm32(6, dram_x, insns)
    _emit_load_imm32(7, dram_bias, insns)
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=6, rs2=4, channel=0)))
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=7, rs2=4, channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)))   # v0 = x H0 during bias wait
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=32)))  # v1 = x H1 during bias wait
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("vload", VectorArgs(vd=2, rs1=2, imm12=0)))   # v2 = bias H0
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("vload", VectorArgs(vd=3, rs1=2, imm12=32)))  # v3 = bias H1
    insns.append(Instruction("delay", ScalarArgs(imm=34)))

    # LMUL=2 add: (v4, v5) = (v0, v1) + (v2, v3)
    insns.append(Instruction("vadd.bf16", VectorArgs(vd=4, vs1=0, vs2=2)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))

    # Pack LMUL=2 pair (v4, v5) → fp8 v6 using unit scale ERF[0]=1
    insns.append(Instruction("vpack.bf16.fp8", VectorArgs(vd=6, vs1=4, es1=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))

    # vstore fp8 v6 → VMEM_OUT
    insns.append(Instruction("vstore", VectorArgs(vd=6, rs1=3, imm12=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))

    # DMA store fp8 output (1024 B) → DRAM
    _emit_load_imm32(8, dram_out, insns)
    insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=8, rs1=3, rs2=5, channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

    return insns


def _make_program(seed: int):
    dram_x = 0x0000
    dram_bias = dram_x + BF16_TILE_BYTES
    dram_out = dram_bias + BF16_TILE_BYTES

    torch.manual_seed(seed)
    # Keep values in fp8 range to avoid saturation in reference
    x = torch.randn(TILE, TILE, dtype=torch.bfloat16) * 0.4
    bias = torch.randn(TILE, TILE, dtype=torch.bfloat16) * 0.1
    expected = bias_add_cast_reference(x, bias)

    insns = make_bias_add_cast_instructions(
        dram_x=dram_x, dram_bias=dram_bias, dram_out=dram_out
    )
    regions = [
        (dram_x, _colblock_bf16(x)),
        (dram_bias, _colblock_bf16(bias)),
    ]
    golden = (dram_out, _tile_fp8(expected))
    return insns, regions, golden


_insns, _regions, _golden = _make_program(seed=210)


class ParameterizedBiasAddCast32x32Program(Program):
    """bias_add_cast: fp8(x + bias) on a single 32×32 bf16 tile."""

    instructions: List[Instruction[Any]] = _insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _regions
    golden_result: tuple[int, torch.Tensor] = _golden
    kernel_tolerance: tuple[float, float] = (1e-1, 1e-1)
