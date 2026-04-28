"""Fused attention (flash SDPA) kernel with HEAD_DIM fixed at 64 (H=2).

Parameterized over:
    Q_ROWS = any multiple of 32  (processed as Q_ROWS//32 Q-blocks in sequence)
    K_SEQ  = any multiple of 32  (K_TILES = K_SEQ//32 flash-attention tiles)

Flash attention per Q-block (online softmax):
    m = -inf, l = 0, O = 0
    for k in K_TILES:
        S  = Q @ K^T * scale         [32,32]
        m' = max(m, rowmax(S))
        O  = exp(m-m') * O + exp(S-m') @ V
        l  = exp(m-m') * l + rowsum(exp(S-m'))
        m  = m'
    output = O / l

MRF register layout (H=2, S=scratch_start=10):

    Persistent (survive K-tile loop):
      v0  v1              Q_fp8[0..1]  (Q col-block fp8 activations)
      v2  v3              scale pair [32,16] broadcast
      v4  v5              m_prev pair
      v6  v7              l_prev pair
      v8  v9  v10  v11    O[0] pair, O[1] pair

    Per-tile scratch (S=12):
      v12 v13   KT_BF16 pair
      v14       KT_FP8
      v16 v17   VT_BF16 pair
      v18       VT_FP8
      v20 v21   T_SCORES
      v22 v23   T_SCALED
      v24 v25   T_TMAX / exp_diff (reused)
      v26 v27   T_MNEW
      v28 v29   T_EXPS
      v30       T_EXPSFP8
      v32 v33   T_VC
      v34 v35   T_LDECAY / inv_l
      v36 v37   T_LSUM   (max reg 37 < 64)

DRAM layouts (bf16, column-blocked; each vload = one [32,16] block = 1024 bytes):
    Q[qb]:   [32, 64] col-blocked → [4*32, 16]
    KT[k]:   K^T[64, 32] col-blocked → [4*32, 16]
    VT[k]:   V_std[32, 64] col-blocked → [4*32, 16]
    SCALE:   [32, 16] single block (1024 bytes, constant)
    OUT[qb]: same layout as Q[qb]

Scalar register map:
    x1=VMEM_Q  x2=VMEM_KT  x3=VMEM_VT  x4=VMEM_SCALE  x5=VMEM_OUT
    x6=DRAM_Q[qb]  x7=DRAM_SCALE  x8=DRAM_OUT[qb]
    x9=TILE_BYTES  x10=SCALE_BYTES
    x11=DRAM_KT[k]  x12=DRAM_VT[k]
"""

import math
from typing import List, Tuple

import torch

from npu_model.software.program import Program, ASM_FOLDER
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction

TILE = 32
BF16 = 2
SCALE_BYTES = TILE * (TILE // 2) * BF16  # [32,16] bf16 = 1024, constant


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


def _make_attn_program(Q_ROWS: int, K_SEQ: int, seed: int):
    HEAD_DIM = 64
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

    return regions, (dram_out, expected)


_fa_q32_k64  = _make_attn_program(Q_ROWS=32, K_SEQ=64,  seed=10)
_fa_q32_k96  = _make_attn_program(Q_ROWS=32, K_SEQ=96,  seed=11)
_fa_q32_k128 = _make_attn_program(Q_ROWS=32, K_SEQ=128, seed=12)
_fa_q64_k64  = _make_attn_program(Q_ROWS=64, K_SEQ=64,  seed=40)
_fa_q64_k96  = _make_attn_program(Q_ROWS=64, K_SEQ=96,  seed=41)


class ParameterizedFusedAttentionQ32K64Program(Program):
    """Flash attention: Q_ROWS=32, K_SEQ=64, HEAD_DIM=64."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_fused_attention_q32_k64.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _fa_q32_k64[0]
    golden_result: tuple[int, torch.Tensor] = _fa_q32_k64[1]
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)


class ParameterizedFusedAttentionQ32K96Program(Program):
    """Flash attention: Q_ROWS=32, K_SEQ=96, HEAD_DIM=64 (3 K-tiles)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_fused_attention_q32_k96.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _fa_q32_k96[0]
    golden_result: tuple[int, torch.Tensor] = _fa_q32_k96[1]
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)


class ParameterizedFusedAttentionQ32K128Program(Program):
    """Flash attention: Q_ROWS=32, K_SEQ=128, HEAD_DIM=64 (4 K-tiles)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_fused_attention_q32_k128.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _fa_q32_k128[0]
    golden_result: tuple[int, torch.Tensor] = _fa_q32_k128[1]
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)


class ParameterizedFusedAttentionQ64K64Program(Program):
    """Flash attention: Q_ROWS=64, K_SEQ=64, HEAD_DIM=64 (2 Q-blocks)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_fused_attention_q64_k64.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _fa_q64_k64[0]
    golden_result: tuple[int, torch.Tensor] = _fa_q64_k64[1]
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)


class ParameterizedFusedAttentionQ64K96Program(Program):
    """Flash attention: Q_ROWS=64, K_SEQ=96, HEAD_DIM=64 (2 Q-blocks, 3 K-tiles)."""

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'parameterized_fused_attention_q64_k96.S')
    memory_regions: List[Tuple[int, torch.Tensor]] = _fa_q64_k96[0]
    golden_result: tuple[int, torch.Tensor] = _fa_q64_k96[1]
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)
