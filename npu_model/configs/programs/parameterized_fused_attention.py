"""Fully parameterized fused attention (flash SDPA) kernel.

Parameterized over:
    Q_ROWS   = any multiple of 32  (processed as Q_ROWS//32 Q-blocks in sequence)
    HEAD_DIM = any multiple of 32  (H = HEAD_DIM//32 MXU passes for Q@K^T)
    K_SEQ    = any multiple of 32  (K_TILES = K_SEQ//32 flash-attention tiles)

Flash attention per Q-block (online softmax):
    m = -inf, l = 0, O = 0
    for k in K_TILES:
        S  = Q @ K^T * scale         [32,32]
        m' = max(m, rowmax(S))
        O  = exp(m-m') * O + exp(S-m') @ V
        l  = exp(m-m') * l + rowsum(exp(S-m'))
        m  = m'
    output = O / l

MRF register layout (H = HEAD_DIM // 32, S = scratch_start):

    Persistent (survive K-tile loop):
      v0 .. v(H-1)           Q_fp8[i] = Q[:,i*32:(i+1)*32] fp8   (activations, odd OK)
      v(S0)   v(S0+1)        scale pair [32,16] broadcast  S0=_even(H)
      v(S0+2) v(S0+3)        m_prev pair
      v(S0+4) v(S0+5)        l_prev pair
      v(S0+6) .. v(S0+5+2H)  O[j] pairs for j=0..H-1

    Per-tile scratch (S = _even(S0+6+2H) = _even(3H+6)):
      v(S)   v(S+1)   KT_BF16 pair  (reused per KT segment)
      v(S+2)          KT_FP8        (even)
      v(S+4) v(S+5)   VT_BF16 pair  (reused per VT col-block)
      v(S+6)          VT_FP8        (even)
      v(S+8) v(S+9)   T_SCORES
      v(S+10) v(S+11) T_SCALED
      v(S+12) v(S+13) T_TMAX / exp_diff (reused)
      v(S+14) v(S+15) T_MNEW
      v(S+16) v(S+17) T_EXPS
      v(S+18)         T_EXPSFP8     (even)
      v(S+20) v(S+21) T_VC          (scratch per V col-block)
      v(S+22) v(S+23) T_LDECAY / inv_l
      v(S+24) v(S+25) T_LSUM

    Max regs: S+25.  H=2→37  H=4→43  H=8→55  (all < 64) ✓

DRAM layouts (bf16, column-blocked; each vload = one [32,16] block = 1024 bytes):
    Q[qb]:  [32, HEAD_DIM] col-blocked → [2H*32, 16]
    KT[k]:  K^T[HEAD_DIM, 32] col-blocked → [2H*32, 16]
    VT[k]:  V_std[32, HEAD_DIM] col-blocked → [2H*32, 16]
    SCALE:  [32, 16] single block (1024 bytes, constant)
    OUT[qb]: same layout as Q[qb]

VMEM addresses (byte offsets from 0):
    VMEM_Q     = 0
    VMEM_KT    = TILE_BYTES
    VMEM_VT    = 2 * TILE_BYTES
    VMEM_SCALE = 3 * TILE_BYTES
    VMEM_OUT   = 3 * TILE_BYTES + SCALE_BYTES

Scalar register map:
    x1=VMEM_Q  x2=VMEM_KT  x3=VMEM_VT  x4=VMEM_SCALE  x5=VMEM_OUT
    x6=DRAM_Q[qb]  x7=DRAM_SCALE  x8=DRAM_OUT[qb]
    x9=TILE_BYTES  x10=SCALE_BYTES
    x11=DRAM_KT[k]  x12=DRAM_VT[k]
"""

import math
from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs

TILE = 32
BF16 = 2
SCALE_BYTES = TILE * (TILE // 2) * BF16  # [32,16] bf16 = 1024, constant


# ── register helpers ──────────────────────────────────────────────────────────


def _even(n: int) -> int:
    return n if n % 2 == 0 else n + 1


def _reg_layout(H: int) -> dict:
    """Compute MRF register base indices for HEAD_DIM_TILES = H."""
    Q_FP8_BASE = 0
    S0 = _even(H)
    SCALE_BASE = S0
    M_BASE = S0 + 2
    L_BASE = S0 + 4
    O_BASE = S0 + 6
    S = _even(O_BASE + 2 * H)
    max_reg = S + 25
    assert max_reg < 64, f"MRF overflow: {max_reg + 1} regs needed for H={H} (max 64)"
    return dict(
        Q_FP8_BASE=Q_FP8_BASE,
        SCALE_BASE=SCALE_BASE,
        M_BASE=M_BASE,
        L_BASE=L_BASE,
        O_BASE=O_BASE,
        KT_BF16=S,
        KT_FP8=S + 2,
        VT_BF16=S + 4,
        VT_FP8=S + 6,
        T_SCORES=S + 8,
        T_SCALED=S + 10,
        T_TMAX=S + 12,
        T_MNEW=S + 14,
        T_EXPS=S + 16,
        T_EXPSFP8=S + 18,
        T_VC=S + 20,
        T_LDECAY=S + 22,
        T_LSUM=S + 24,
    )


# ── ISA helpers ───────────────────────────────────────────────────────────────


def _emit_load_imm32(rd: int, value: int, out: list) -> None:
    """lui + addi to materialise a 32-bit value in scalar register rd."""
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


# ── data layout helpers ───────────────────────────────────────────────────────


def _col_block(t: torch.Tensor) -> torch.Tensor:
    """Pack [rows, cols] bf16 into column-blocked DRAM layout.

    Each vload reads one contiguous [32, 16] bf16 block (1024 bytes).
    Blocks ordered: col 0:16, col 16:32, ... for rows 0:32; then same for
    rows 32:64; etc.
    """
    rows, cols = t.shape
    assert rows % TILE == 0 and cols % 16 == 0, (
        f"_col_block requires shape aligned to (32, 16), got {t.shape}"
    )
    parts = []
    for r in range(0, rows, TILE):
        for c in range(0, cols, 16):
            parts.append(t[r : r + TILE, c : c + 16].contiguous())
    return torch.cat(parts, dim=0)


# ── ISA generator ─────────────────────────────────────────────────────────────


def make_fused_attention_instructions(
    Q_ROWS: int,
    K_SEQ: int,
    HEAD_DIM: int,
    dram_q: int,
    dram_kt_base: int,
    dram_vt_base: int,
    dram_scale: int,
    dram_out: int,
) -> list:
    """Generate flash-attention ISA instructions.

    The Q-block and K-tile loops use hardware branch instructions.
    Inner loops over H (head-dim segments) remain Python-unrolled since
    they reference different MRF register indices per iteration.

    Scalar register map:
        x1=VMEM_Q  x2=VMEM_KT  x3=VMEM_VT  x4=VMEM_SCALE  x5=VMEM_OUT
        x6=dram_q_ptr (Q-block, advances by TILE_BYTES per qb)
        x7=dram_scale  x8=dram_out_ptr (Q-block, advances by TILE_BYTES per qb)
        x9=TILE_BYTES  x10=SCALE_BYTES
        x11=dram_kt_ptr (K-tile, advances by TILE_BYTES per k; reset per qb)
        x12=dram_vt_ptr (K-tile, advances by TILE_BYTES per k; reset per qb)
        x13=dram_kt_base (constant, used to reset x11 per qb)
        x14=dram_vt_base (constant, used to reset x12 per qb)
        x15=Q_BLOCKS  x16=qb counter
        x17=K_TILES   x18=k counter
    """
    assert Q_ROWS % TILE == 0, f"Q_ROWS={Q_ROWS} must be multiple of {TILE}"
    assert K_SEQ % TILE == 0, f"K_SEQ={K_SEQ} must be multiple of {TILE}"
    assert HEAD_DIM % TILE == 0, f"HEAD_DIM={HEAD_DIM} must be multiple of {TILE}"

    H = HEAD_DIM // TILE
    K_TILES = K_SEQ // TILE
    Q_BLOCKS = Q_ROWS // TILE
    TILE_BYTES = TILE * HEAD_DIM * BF16

    R = _reg_layout(H)
    Q_FP8_BASE = R["Q_FP8_BASE"]
    SCALE_BASE = R["SCALE_BASE"]
    M_BASE = R["M_BASE"]
    L_BASE = R["L_BASE"]
    O_BASE = R["O_BASE"]
    KT_BF16 = R["KT_BF16"]
    KT_FP8 = R["KT_FP8"]
    VT_BF16 = R["VT_BF16"]
    VT_FP8 = R["VT_FP8"]
    T_SCORES = R["T_SCORES"]
    T_SCALED = R["T_SCALED"]
    T_TMAX = R["T_TMAX"]
    T_MNEW = R["T_MNEW"]
    T_EXPS = R["T_EXPS"]
    T_EXPSFP8 = R["T_EXPSFP8"]
    T_VC = R["T_VC"]
    T_LDECAY = R["T_LDECAY"]
    T_LSUM = R["T_LSUM"]

    VMEM_Q = 0
    VMEM_KT = VMEM_Q + TILE_BYTES
    VMEM_VT = VMEM_KT + TILE_BYTES
    VMEM_SCALE = VMEM_VT + TILE_BYTES
    VMEM_OUT = VMEM_SCALE + SCALE_BYTES

    nop = Instruction("addi", ScalarArgs(rd=0, rs1=0, imm=0))
    insns: list = []

    # Prologue: VMEM addresses, constants, loop limits
    _emit_load_imm32(1, VMEM_Q, insns)
    _emit_load_imm32(2, VMEM_KT, insns)
    _emit_load_imm32(3, VMEM_VT, insns)
    _emit_load_imm32(4, VMEM_SCALE, insns)
    _emit_load_imm32(5, VMEM_OUT, insns)
    _emit_load_imm32(6, dram_q, insns)       # dram_q_ptr (initial)
    _emit_load_imm32(7, dram_scale, insns)
    _emit_load_imm32(8, dram_out, insns)     # dram_out_ptr (initial)
    _emit_load_imm32(9, TILE_BYTES, insns)
    insns.append(Instruction("addi", ScalarArgs(rd=10, rs1=0, imm=SCALE_BYTES)))
    _emit_load_imm32(13, dram_kt_base, insns)
    _emit_load_imm32(14, dram_vt_base, insns)
    _emit_load_imm32(15, Q_BLOCKS, insns)
    _emit_load_imm32(17, K_TILES, insns)
    insns.append(Instruction("addi", ScalarArgs(rd=16, rs1=0, imm=0)))  # qb = 0

    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    # Load scale once: DRAM → VMEM_SCALE → MRF
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=4, rs1=7, rs2=10, channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("vload", VectorArgs(vd=SCALE_BASE, rs1=4, imm12=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("vload", VectorArgs(vd=SCALE_BASE + 1, rs1=4, imm12=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))

    # qb-loop
    qb_loop_start = len(insns)
    # Load Q[qb] from DRAM → VMEM_Q
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=6, rs2=9, channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

    # Quantize Q segments (H iterations, Python-unrolled — different MRF regs per i)
    for i in range(H):
        insns.append(Instruction("vload", VectorArgs(vd=KT_BF16, rs1=1, imm12=i * 64)))
        insns.append(Instruction("delay", ScalarArgs(imm=34)))
        insns.append(Instruction("vload", VectorArgs(vd=KT_BF16 + 1, rs1=1, imm12=i * 64 + 32)))
        insns.append(Instruction("delay", ScalarArgs(imm=34)))
        insns.append(Instruction("vmatpush.acc.bf16.mxu0", MatrixArgs(vd=1, vs1=KT_BF16)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))
        insns.append(Instruction("vmatpop.fp8.acc.mxu0", MatrixArgs(vd=Q_FP8_BASE + i, vs1=1)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))

    # Initialize online-softmax state (fresh per Q-block)
    for vd in [M_BASE, M_BASE + 1]:
        insns.append(Instruction("vli.all", VectorArgs(vd=vd, imm=-100)))
        insns.append(Instruction("delay", ScalarArgs(imm=65)))
    for vd in [L_BASE, L_BASE + 1]:
        insns.append(Instruction("vli.all", VectorArgs(vd=vd, imm=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=65)))
    for j in range(H):
        for vd in [O_BASE + 2 * j, O_BASE + 2 * j + 1]:
            insns.append(Instruction("vli.all", VectorArgs(vd=vd, imm=0)))
            insns.append(Instruction("delay", ScalarArgs(imm=65)))

    # Reset K-tile pointers to base addresses for this Q-block
    insns.append(Instruction("addi", ScalarArgs(rd=11, rs1=13, imm=0)))  # dram_kt_ptr = kt_base
    insns.append(Instruction("addi", ScalarArgs(rd=12, rs1=14, imm=0)))  # dram_vt_ptr = vt_base
    insns.append(Instruction("addi", ScalarArgs(rd=18, rs1=0, imm=0)))   # k = 0

    # k-loop
    k_loop_start = len(insns)
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=11, rs2=9, channel=0)))
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=3, rs1=12, rs2=9, channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    # Q @ K^T: H iterations (Python-unrolled — different Q_fp8 and MRF regs per i)
    for i in range(H):
        insns.append(Instruction("vload", VectorArgs(vd=KT_BF16, rs1=2, imm12=i * 64)))
        insns.append(Instruction("delay", ScalarArgs(imm=34)))
        insns.append(Instruction("vload", VectorArgs(vd=KT_BF16 + 1, rs1=2, imm12=i * 64 + 32)))
        insns.append(Instruction("delay", ScalarArgs(imm=34)))
        insns.append(Instruction("vmatpush.acc.bf16.mxu0", MatrixArgs(vd=1, vs1=KT_BF16)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))
        insns.append(Instruction("vmatpop.fp8.acc.mxu0", MatrixArgs(vd=KT_FP8, vs1=1)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))
        insns.append(Instruction("vmatpush.weight.mxu0", MatrixArgs(vd=0, vs1=KT_FP8)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))
        if i == 0:
            insns.append(Instruction("vmatmul.mxu0", MatrixArgs(vd=0, vs1=Q_FP8_BASE, vs2=0)))
        else:
            insns.append(Instruction("vmatmul.acc.mxu0", MatrixArgs(vd=0, vs1=Q_FP8_BASE + i, vs2=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=96)))

    insns.append(Instruction("vmatpop.bf16.acc.mxu0", MatrixArgs(vd=T_SCORES, vs1=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=32)))

    # scaled = scores * scale
    insns.append(Instruction("vmul.bf16", VectorArgs(vd=T_SCALED, vs1=T_SCORES, vs2=SCALE_BASE)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))

    # tile_max = rowmax(scaled)
    insns.append(Instruction("vredmax.row.bf16", VectorArgs(vd=T_TMAX, vs1=T_SCALED)))
    insns.append(Instruction("delay", ScalarArgs(imm=34)))

    # m_new = max(m_prev, tile_max)
    insns.append(Instruction("vmaximum.bf16", VectorArgs(vd=T_MNEW, vs1=M_BASE, vs2=T_TMAX)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))

    # exp_diff = exp(m_prev - m_new), reuse T_TMAX
    insns.append(Instruction("vsub.bf16", VectorArgs(vd=T_TMAX, vs1=M_BASE, vs2=T_MNEW)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))
    insns.append(Instruction("vexp.bf16", VectorArgs(vd=T_TMAX, vs1=T_TMAX)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))

    # O[j] *= exp_diff (Python-unrolled over H output col-blocks)
    for j in range(H):
        insns.append(Instruction("vmul.bf16", VectorArgs(vd=O_BASE + 2 * j, vs1=O_BASE + 2 * j, vs2=T_TMAX)))
        insns.append(Instruction("delay", ScalarArgs(imm=66)))

    # exp_s = exp(scaled - m_new)
    insns.append(Instruction("vsub.bf16", VectorArgs(vd=T_EXPS, vs1=T_SCALED, vs2=T_MNEW)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))
    insns.append(Instruction("vexp.bf16", VectorArgs(vd=T_EXPS, vs1=T_EXPS)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))

    # Quantize exp_s → fp8
    insns.append(Instruction("vmatpush.acc.bf16.mxu0", MatrixArgs(vd=1, vs1=T_EXPS)))
    insns.append(Instruction("delay", ScalarArgs(imm=32)))
    insns.append(Instruction("vmatpop.fp8.acc.mxu0", MatrixArgs(vd=T_EXPSFP8, vs1=1)))
    insns.append(Instruction("delay", ScalarArgs(imm=32)))

    # exp_s @ V (Python-unrolled over H V col-blocks — different O regs per j)
    for j in range(H):
        insns.append(Instruction("vload", VectorArgs(vd=VT_BF16, rs1=3, imm12=j * 64)))
        insns.append(Instruction("delay", ScalarArgs(imm=34)))
        insns.append(Instruction("vload", VectorArgs(vd=VT_BF16 + 1, rs1=3, imm12=j * 64 + 32)))
        insns.append(Instruction("delay", ScalarArgs(imm=34)))
        insns.append(Instruction("vmatpush.acc.bf16.mxu0", MatrixArgs(vd=1, vs1=VT_BF16)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))
        insns.append(Instruction("vmatpop.fp8.acc.mxu0", MatrixArgs(vd=VT_FP8, vs1=1)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))
        insns.append(Instruction("vmatpush.weight.mxu0", MatrixArgs(vd=0, vs1=VT_FP8)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))
        insns.append(Instruction("vmatmul.mxu0", MatrixArgs(vd=0, vs1=T_EXPSFP8, vs2=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=96)))
        insns.append(Instruction("vmatpop.bf16.acc.mxu0", MatrixArgs(vd=T_VC, vs1=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))
        insns.append(Instruction("vadd.bf16", VectorArgs(vd=O_BASE + 2 * j, vs1=O_BASE + 2 * j, vs2=T_VC)))
        insns.append(Instruction("delay", ScalarArgs(imm=66)))

    # l = exp_diff * l + rowsum(exp_s)
    insns.append(Instruction("vmul.bf16", VectorArgs(vd=T_LDECAY, vs1=T_TMAX, vs2=L_BASE)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))
    insns.append(Instruction("vredsum.row.bf16", VectorArgs(vd=T_LSUM, vs1=T_EXPS)))
    insns.append(Instruction("delay", ScalarArgs(imm=39)))
    insns.append(Instruction("vadd.bf16", VectorArgs(vd=L_BASE, vs1=T_LDECAY, vs2=T_LSUM)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))

    # m_prev = m_new
    insns.append(Instruction("vmov", VectorArgs(vd=M_BASE, vs1=T_MNEW)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))
    insns.append(Instruction("vmov", VectorArgs(vd=M_BASE + 1, vs1=T_MNEW + 1)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))

    # Advance k pointers and counter
    insns.append(Instruction("add", ScalarArgs(rd=11, rs1=11, rs2=9)))   # dram_kt_ptr += TILE_BYTES
    insns.append(Instruction("add", ScalarArgs(rd=12, rs1=12, rs2=9)))   # dram_vt_ptr += TILE_BYTES
    insns.append(Instruction("addi", ScalarArgs(rd=18, rs1=18, imm=1)))
    k_blt_idx = len(insns)
    insns.append(Instruction("blt", ScalarArgs(rs1=18, rs2=17, imm=k_loop_start - k_blt_idx)))
    insns.append(nop)
    insns.append(nop)

    # Normalize: O /= l (Python-unrolled over H)
    insns.append(Instruction("vrecip.bf16", VectorArgs(vd=T_LDECAY, vs1=L_BASE)))
    insns.append(Instruction("delay", ScalarArgs(imm=66)))
    for j in range(H):
        insns.append(Instruction("vmul.bf16", VectorArgs(vd=O_BASE + 2 * j, vs1=O_BASE + 2 * j, vs2=T_LDECAY)))
        insns.append(Instruction("delay", ScalarArgs(imm=66)))

    # Store output O pairs → VMEM_OUT → DRAM (Python-unrolled over H)
    for j in range(H):
        insns.append(Instruction("vstore", VectorArgs(vd=O_BASE + 2 * j, rs1=5, imm12=j * 64)))
        insns.append(Instruction("delay", ScalarArgs(imm=34)))
        insns.append(Instruction("vstore", VectorArgs(vd=O_BASE + 2 * j + 1, rs1=5, imm12=j * 64 + 32)))
        insns.append(Instruction("delay", ScalarArgs(imm=34)))
    insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=8, rs1=5, rs2=9, channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

    # Advance qb pointers and counter
    insns.append(Instruction("add", ScalarArgs(rd=6, rs1=6, rs2=9)))    # dram_q_ptr += TILE_BYTES
    insns.append(Instruction("add", ScalarArgs(rd=8, rs1=8, rs2=9)))    # dram_out_ptr += TILE_BYTES
    insns.append(Instruction("addi", ScalarArgs(rd=16, rs1=16, imm=1)))
    qb_blt_idx = len(insns)
    insns.append(Instruction("blt", ScalarArgs(rs1=16, rs2=15, imm=qb_loop_start - qb_blt_idx)))
    insns.append(nop)
    insns.append(nop)

    return insns


# ── ISA-exact golden reference ────────────────────────────────────────────────


def fused_attention_golden(
    Q_raw: torch.Tensor,
    Ks_raw: list,
    Vs_mlir: list,
    scale_val: float,
) -> torch.Tensor:
    """Simulate ISA flash-attention exactly.

    Args:
        Q_raw:   [Q_ROWS, HEAD_DIM] bf16
        Ks_raw:  K_TILES × [TILE, HEAD_DIM] bf16  (un-transposed K)
        Vs_mlir: K_TILES × [HEAD_DIM, TILE] bf16  (MLIR head_dim-first V)
        scale_val: 1/sqrt(HEAD_DIM)

    Returns:
        [Q_BLOCKS * 2H * TILE, 16] bf16 — col-blocked, matching DRAM store order.
    """
    Q_ROWS, HEAD_DIM = Q_raw.shape
    H = HEAD_DIM // TILE
    H4 = TILE // 2  # = 16
    Q_BLOCKS = Q_ROWS // TILE
    scale_r = torch.full((TILE, H4), scale_val, dtype=torch.bfloat16)

    out_parts = []
    for qb in range(Q_BLOCKS):
        Q_block = Q_raw[qb * TILE : (qb + 1) * TILE, :]

        # Quantize Q to fp8 (matches ISA bf16-pair → acc → fp8)
        Q_fp8 = [Q_block[:, i * TILE : (i + 1) * TILE].to(torch.float8_e4m3fn) for i in range(H)]

        m = torch.full((TILE, H4), -100.0, dtype=torch.bfloat16)
        l = torch.zeros(TILE, H4, dtype=torch.bfloat16)
        O = [torch.zeros(TILE, TILE, dtype=torch.bfloat16) for _ in range(H)]

        for k_raw, v_mlir in zip(Ks_raw, Vs_mlir):
            KT = k_raw.T.contiguous()  # [HEAD_DIM, TILE]

            # Q @ K^T: accumulate in float16 (matches MXU accumulator)
            scores_f16 = torch.zeros(TILE, TILE, dtype=torch.float16)
            for i in range(H):
                KT_seg = KT[i * TILE : (i + 1) * TILE, :].to(torch.float8_e4m3fn)
                scores_f16 = scores_f16 + Q_fp8[i].to(torch.float16) @ KT_seg.to(torch.float16)
            scores = scores_f16.to(torch.bfloat16)

            sl = (scores[:, :H4] * scale_r).to(torch.bfloat16)
            sr = (scores[:, H4:] * scale_r).to(torch.bfloat16)
            rm_l = sl.max(dim=1, keepdim=True).values.expand(-1, H4).to(torch.bfloat16)
            rm_r = sr.max(dim=1, keepdim=True).values.expand(-1, H4).to(torch.bfloat16)
            tile_max = torch.maximum(rm_l, rm_r).to(torch.bfloat16)
            m_new = torch.maximum(m, tile_max).to(torch.bfloat16)
            exp_diff = torch.exp((m - m_new).to(torch.bfloat16)).to(torch.bfloat16)

            # exp_diff is [TILE, H4]=[32,16] broadcast; O[j] is [32,32].
            # ISA: pair vmul broadcasts exp_diff across both [32,16] halves of O[j].
            # exp_diff is [TILE, H4]=[32,16]; O[j] is [32,32].
            # ISA: pair vmul broadcasts exp_diff to both [32,16] halves of O[j].
            exp_diff_full = exp_diff.repeat(1, 2)  # [32,32], same value in both halves
            O = [(Oj * exp_diff_full).to(torch.bfloat16) for Oj in O]

            esl = torch.exp((sl - m_new).to(torch.bfloat16)).to(torch.bfloat16)
            esr = torch.exp((sr - m_new).to(torch.bfloat16)).to(torch.bfloat16)
            exp_s_fp8 = torch.cat([esl, esr], dim=1).to(torch.float8_e4m3fn)

            V_std = v_mlir.T.contiguous()  # [TILE, HEAD_DIM]
            for j in range(H):
                V_col = V_std[:, j * TILE : (j + 1) * TILE].to(torch.float8_e4m3fn)
                vc = (exp_s_fp8.to(torch.float16) @ V_col.to(torch.float16)).to(torch.bfloat16)
                O[j] = (O[j] + vc).to(torch.bfloat16)

            rs_l = esl.sum(dim=1, keepdim=True).expand(-1, H4).to(torch.bfloat16)
            rs_r = esr.sum(dim=1, keepdim=True).expand(-1, H4).to(torch.bfloat16)
            l = (
                (exp_diff * l).to(torch.bfloat16) + (rs_l + rs_r).to(torch.bfloat16)
            ).to(torch.bfloat16)
            m = m_new

        inv_l = (1.0 / l).to(torch.bfloat16)
        inv_l_full = inv_l.repeat(1, 2)  # [32,32], same value in both halves
        O = [(Oj * inv_l_full).to(torch.bfloat16) for Oj in O]

        # Col-blocked: O[0]_left, O[0]_right, O[1]_left, ..., O[H-1]_right
        block_parts = []
        for j in range(H):
            block_parts.append(O[j][:, :H4].contiguous())
            block_parts.append(O[j][:, H4:].contiguous())
        out_parts.append(torch.cat(block_parts, dim=0))

    return torch.cat(out_parts, dim=0)


# ── program factory ───────────────────────────────────────────────────────────


def _make_attn_program(Q_ROWS: int, K_SEQ: int, HEAD_DIM: int, seed: int):
    H = HEAD_DIM // TILE
    K_TILES = K_SEQ // TILE
    Q_BLOCKS = Q_ROWS // TILE
    TILE_BYTES = TILE * HEAD_DIM * BF16

    dram_q = 0x0000
    dram_kt_base = dram_q + Q_BLOCKS * TILE_BYTES
    dram_vt_base = dram_kt_base + K_TILES * TILE_BYTES
    dram_scale = dram_vt_base + K_TILES * TILE_BYTES
    dram_out = dram_scale + SCALE_BYTES

    torch.manual_seed(seed)
    scale_val = 1.0 / math.sqrt(float(HEAD_DIM))

    Q_raw = (torch.randn(Q_ROWS, HEAD_DIM) * 0.5).to(torch.bfloat16)
    Ks_raw = [(torch.randn(TILE, HEAD_DIM) * 0.5).to(torch.bfloat16) for _ in range(K_TILES)]
    Vs_mlir = [(torch.randn(HEAD_DIM, TILE) * 0.5).to(torch.bfloat16) for _ in range(K_TILES)]
    scale_data = torch.full((TILE, TILE // 2), scale_val, dtype=torch.bfloat16)

    regions = []
    for qb in range(Q_BLOCKS):
        Q_block = Q_raw[qb * TILE : (qb + 1) * TILE, :]
        regions.append((dram_q + qb * TILE_BYTES, _col_block(Q_block)))
    for k, k_raw in enumerate(Ks_raw):
        regions.append((dram_kt_base + k * TILE_BYTES, _col_block(k_raw.T.contiguous())))
    for k, v_mlir in enumerate(Vs_mlir):
        regions.append((dram_vt_base + k * TILE_BYTES, _col_block(v_mlir.T.contiguous())))
    regions.append((dram_scale, scale_data))

    expected = fused_attention_golden(Q_raw, Ks_raw, Vs_mlir, scale_val)

    insns = make_fused_attention_instructions(
        Q_ROWS=Q_ROWS,
        K_SEQ=K_SEQ,
        HEAD_DIM=HEAD_DIM,
        dram_q=dram_q,
        dram_kt_base=dram_kt_base,
        dram_vt_base=dram_vt_base,
        dram_scale=dram_scale,
        dram_out=dram_out,
    )
    return insns, regions, (dram_out, expected)


# ── Program classes ───────────────────────────────────────────────────────────

# HEAD_DIM=64 (H=2) — SmolVLA shapes
_fa_q32_k64_h64 = _make_attn_program(Q_ROWS=32, K_SEQ=64, HEAD_DIM=64, seed=10)
_fa_q32_k96_h64 = _make_attn_program(Q_ROWS=32, K_SEQ=96, HEAD_DIM=64, seed=11)
_fa_q32_k128_h64 = _make_attn_program(Q_ROWS=32, K_SEQ=128, HEAD_DIM=64, seed=12)

# HEAD_DIM=128 (H=4)
_fa_q32_k64_h128 = _make_attn_program(Q_ROWS=32, K_SEQ=64, HEAD_DIM=128, seed=20)
_fa_q32_k96_h128 = _make_attn_program(Q_ROWS=32, K_SEQ=96, HEAD_DIM=128, seed=21)

# HEAD_DIM=256 (H=8) — PI0/PaliGemma-3B
_fa_q32_k64_h256 = _make_attn_program(Q_ROWS=32, K_SEQ=64, HEAD_DIM=256, seed=30)
_fa_q32_k128_h256 = _make_attn_program(Q_ROWS=32, K_SEQ=128, HEAD_DIM=256, seed=31)

# Q_ROWS=64 (2 Q-blocks)
_fa_q64_k64_h64 = _make_attn_program(Q_ROWS=64, K_SEQ=64, HEAD_DIM=64, seed=40)
_fa_q64_k96_h64 = _make_attn_program(Q_ROWS=64, K_SEQ=96, HEAD_DIM=64, seed=41)

# Q_ROWS=64, HEAD_DIM=128 — multi-Q-block with larger HEAD_DIM, within cycle budget
# (Q64+H256 exceeds 100k default cycles due to 16KB DMA transfers per tile)
_fa_q64_k64_h128 = _make_attn_program(Q_ROWS=64, K_SEQ=64, HEAD_DIM=128, seed=50)


class ParameterizedFusedAttentionQ32K64H64Program(Program):
    """Flash attention: Q_ROWS=32, K_SEQ=64, HEAD_DIM=64 (SmolVLA shape)."""

    instructions: List[Instruction[Any]] = _fa_q32_k64_h64[0]
    memory_regions: List[Tuple[int, torch.Tensor]] = _fa_q32_k64_h64[1]
    golden_result: tuple[int, torch.Tensor] = _fa_q32_k64_h64[2]
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)


class ParameterizedFusedAttentionQ32K96H64Program(Program):
    """Flash attention: Q_ROWS=32, K_SEQ=96, HEAD_DIM=64 (3 K-tiles)."""

    instructions: List[Instruction[Any]] = _fa_q32_k96_h64[0]
    memory_regions: List[Tuple[int, torch.Tensor]] = _fa_q32_k96_h64[1]
    golden_result: tuple[int, torch.Tensor] = _fa_q32_k96_h64[2]
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)


class ParameterizedFusedAttentionQ32K128H64Program(Program):
    """Flash attention: Q_ROWS=32, K_SEQ=128, HEAD_DIM=64 (4 K-tiles)."""

    instructions: List[Instruction[Any]] = _fa_q32_k128_h64[0]
    memory_regions: List[Tuple[int, torch.Tensor]] = _fa_q32_k128_h64[1]
    golden_result: tuple[int, torch.Tensor] = _fa_q32_k128_h64[2]
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)


class ParameterizedFusedAttentionQ32K64H128Program(Program):
    """Flash attention: Q_ROWS=32, K_SEQ=64, HEAD_DIM=128 (H=4)."""

    instructions: List[Instruction[Any]] = _fa_q32_k64_h128[0]
    memory_regions: List[Tuple[int, torch.Tensor]] = _fa_q32_k64_h128[1]
    golden_result: tuple[int, torch.Tensor] = _fa_q32_k64_h128[2]
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)


class ParameterizedFusedAttentionQ32K96H128Program(Program):
    """Flash attention: Q_ROWS=32, K_SEQ=96, HEAD_DIM=128 (H=4, 3 K-tiles)."""

    instructions: List[Instruction[Any]] = _fa_q32_k96_h128[0]
    memory_regions: List[Tuple[int, torch.Tensor]] = _fa_q32_k96_h128[1]
    golden_result: tuple[int, torch.Tensor] = _fa_q32_k96_h128[2]
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)


class ParameterizedFusedAttentionQ32K64H256Program(Program):
    """Flash attention: Q_ROWS=32, K_SEQ=64, HEAD_DIM=256 (H=8, PI0/PaliGemma)."""

    instructions: List[Instruction[Any]] = _fa_q32_k64_h256[0]
    memory_regions: List[Tuple[int, torch.Tensor]] = _fa_q32_k64_h256[1]
    golden_result: tuple[int, torch.Tensor] = _fa_q32_k64_h256[2]
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)


class ParameterizedFusedAttentionQ32K128H256Program(Program):
    """Flash attention: Q_ROWS=32, K_SEQ=128, HEAD_DIM=256 (H=8, 4 K-tiles)."""

    instructions: List[Instruction[Any]] = _fa_q32_k128_h256[0]
    memory_regions: List[Tuple[int, torch.Tensor]] = _fa_q32_k128_h256[1]
    golden_result: tuple[int, torch.Tensor] = _fa_q32_k128_h256[2]
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)
    kernel_max_cycles: int = 300000  # H=8 with 4 K-tiles exceeds 100k default


class ParameterizedFusedAttentionQ64K64H64Program(Program):
    """Flash attention: Q_ROWS=64, K_SEQ=64, HEAD_DIM=64 (2 Q-blocks)."""

    instructions: List[Instruction[Any]] = _fa_q64_k64_h64[0]
    memory_regions: List[Tuple[int, torch.Tensor]] = _fa_q64_k64_h64[1]
    golden_result: tuple[int, torch.Tensor] = _fa_q64_k64_h64[2]
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)


class ParameterizedFusedAttentionQ64K96H64Program(Program):
    """Flash attention: Q_ROWS=64, K_SEQ=96, HEAD_DIM=64 (2 Q-blocks, 3 K-tiles)."""

    instructions: List[Instruction[Any]] = _fa_q64_k96_h64[0]
    memory_regions: List[Tuple[int, torch.Tensor]] = _fa_q64_k96_h64[1]
    golden_result: tuple[int, torch.Tensor] = _fa_q64_k96_h64[2]
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)


class ParameterizedFusedAttentionQ64K64H128Program(Program):
    """Flash attention: Q_ROWS=64, K_SEQ=64, HEAD_DIM=128 (2 Q-blocks, H=4)."""

    instructions: List[Instruction[Any]] = _fa_q64_k64_h128[0]
    memory_regions: List[Tuple[int, torch.Tensor]] = _fa_q64_k64_h128[1]
    golden_result: tuple[int, torch.Tensor] = _fa_q64_k64_h128[2]
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)
