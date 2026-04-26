"""Shape coverage tests derived from SaturnNPU MLIR variants.

Calls every make_*_instructions function with the tensor shapes found in
SaturnNPU/kernels/ and verifies that instruction generation succeeds (no
exception, non-empty list).  No simulation is performed.

Shape derivation rules
----------------------
- Every dimension is rounded up to the nearest multiple of 32 via _p32().
- 2D tensors (M, N): use as-is after padding.
- 3D tensors (B, M, N): flatten B into M where the kernel has no batch dim,
  i.e. M = _p32(B * M_orig), N = _p32(N_orig).
- Fixed-N=32 kernels (softmax, reduction_sum, rms_norm, gelu_tanh): M is the
  total number of 32-element rows to process; see per-kernel notes.
- Matmul shapes: read from operand tensors A(M, K) @ B^T(N, K) → C(M, N).
- Shapes that require unsupported features (non-bf16 dtypes, 4-D+ contractions,
  scalar broadcast) are noted and excluded.
"""

import math

import pytest

from npu_model.configs.programs.parameterized_batch_matmul import (
    make_batch_matmul_instructions,
)
from npu_model.configs.programs.parameterized_bias_add_cast import (
    make_bias_add_cast_instructions,
)
from npu_model.configs.programs.parameterized_elementwise_add import (
    make_elementwise_add_instructions,
)
from npu_model.configs.programs.parameterized_elementwise_div import (
    make_elementwise_div_instructions,
)
from npu_model.configs.programs.parameterized_elementwise_mul import (
    make_elementwise_mul_instructions,
)
from npu_model.configs.programs.parameterized_elementwise_sub import (
    make_elementwise_sub_instructions,
)
from npu_model.configs.programs.parameterized_fused_attention import (
    make_fused_attention_instructions,
)
from npu_model.configs.programs.parameterized_fused_matmul_bias import (
    make_fused_matmul_bias_instructions,
)
from npu_model.configs.programs.parameterized_fused_norm_scale import (
    make_fused_norm_scale_instructions,
)
from npu_model.configs.programs.parameterized_fused_silu_gate import (
    make_fused_silu_gate_instructions,
)
from npu_model.configs.programs.parameterized_gelu_tanh import (
    make_gelu_tanh_instructions,
)
from npu_model.configs.programs.parameterized_matmul import make_matmul_instructions
from npu_model.configs.programs.parameterized_reduction_sum import (
    make_reduction_sum_instructions,
)
from npu_model.configs.programs.parameterized_requant import make_requant_instructions
from npu_model.configs.programs.parameterized_rms_norm import make_rms_norm_instructions
from npu_model.configs.programs.parameterized_rope_frequency import (
    make_rope_frequency_instructions,
)
from npu_model.configs.programs.parameterized_silu import make_silu_instructions
from npu_model.configs.programs.parameterized_softmax import make_softmax_instructions


def _p32(n: int) -> int:
    """Round n up to the nearest multiple of 32."""
    return math.ceil(n / 32) * 32


def _check(insns) -> None:
    assert len(insns) > 0


# ── silu ──────────────────────────────────────────────────────────────────────
# SaturnNPU/kernels/silu/ — three 2D bf16 variants.
# make_silu_instructions(M, N, dram_x, dram_out)


@pytest.mark.parametrize(
    "variant,M,N",
    [
        ("241x2560", _p32(241), 2560),
        ("50x2048", _p32(50), 2048),
        ("50x720", _p32(50), _p32(720)),
    ],
)
def test_silu_mlir_shape(variant, M, N):
    _check(make_silu_instructions(M, N, dram_x=0, dram_out=0))


# ── gelu_tanh ─────────────────────────────────────────────────────────────────
# SaturnNPU/kernels/gelu_tanh/ — one 1024×3072 variant.
# Kernel operates on M×32 tiles (N=32 fixed).  For a 1024-row input, M=1024
# covers one column-tile pass; the caller iterates passes for wider inputs.
# make_gelu_tanh_instructions(M, dram_x, dram_k_coeff, dram_k_sqrt2pi, dram_k_half, dram_out)


@pytest.mark.parametrize(
    "variant,M",
    [
        ("1024x3072", 1024),
    ],
)
def test_gelu_tanh_mlir_shape(variant, M):
    _check(
        make_gelu_tanh_instructions(
            M, dram_x=0, dram_k_coeff=0, dram_k_sqrt2pi=0, dram_k_half=0, dram_out=0
        )
    )


# ── softmax ───────────────────────────────────────────────────────────────────
# SaturnNPU/kernels/linalg.softmax/ — three 3D variants (B×R×C).
# Kernel operates on M×32 tiles (N=32 fixed), one pass per 32-element column
# slice.  M = total rows = _p32(B × R); outer loop over column-tiles is
# managed by the compiler, not the kernel generator.
# make_softmax_instructions(M, dram_x, dram_out)


@pytest.mark.parametrize(
    "variant,M",
    [
        ("15x241x241", _p32(15 * 241)),
        ("15x291x291", _p32(15 * 291)),
        ("15x50x241", _p32(15 * 50)),
    ],
)
def test_softmax_mlir_shape(variant, M):
    _check(make_softmax_instructions(M, dram_x=0, dram_out=0))


# ── reduction_sum ─────────────────────────────────────────────────────────────
# SaturnNPU/kernels/reduction_sum/ — three variants reducing rows of 768/960/720
# columns to scalars.  Kernel reduces one 32-element column-tile per call;
# M = number of rows.
# make_reduction_sum_instructions(M, dram_x, dram_out)


@pytest.mark.parametrize(
    "variant,M",
    [
        ("1024_from_1024x768", 1024),
        ("241_from_241x960", _p32(241)),
        ("50_from_50x720", _p32(50)),
    ],
)
def test_reduction_sum_mlir_shape(variant, M):
    _check(make_reduction_sum_instructions(M, dram_x=0, dram_out=0))


# ── rms_norm ──────────────────────────────────────────────────────────────────
# SaturnNPU/kernels/rms_norm/ — three 1D variants (1024, 241, 50 elements).
# Kernel normalizes M×32 tiles (N=32 fixed); M = total_elements // 32 rounded
# up to a multiple of 32.
# make_rms_norm_instructions(M, dram_x, dram_inv_dim, dram_eps, dram_out)


@pytest.mark.parametrize(
    "variant,M",
    [
        ("1024", 1024 // 32),        # 32 rows of 32 = 1024 elements
        ("241", _p32(241 // 32)),    # 8 groups → M=32 (covers 256 > 241 elements)
        ("50", _p32(50 // 32)),      # 2 groups → M=32
    ],
)
def test_rms_norm_mlir_shape(variant, M):
    _check(
        make_rms_norm_instructions(M, dram_x=0, dram_inv_dim=0, dram_eps=0, dram_out=0)
    )


# ── rope_frequency ────────────────────────────────────────────────────────────
# SaturnNPU/kernels/rope_frequency/ — two 2D variants.
# make_rope_frequency_instructions(M, N, dram_x, dram_out)


@pytest.mark.parametrize(
    "variant,M,N",
    [
        ("241x960", _p32(241), 960),
        ("50x720", _p32(50), _p32(720)),
    ],
)
def test_rope_frequency_mlir_shape(variant, M, N):
    _check(make_rope_frequency_instructions(M, N, dram_x=0, dram_out=0))


# ── elementwise_add ───────────────────────────────────────────────────────────
# SaturnNPU/kernels/elementwise_add/ — 17 variants (2D, 3D, broadcast, 1D).
# 3D (B, R, C): flatten leading dims → M = _p32(B * R), N = _p32(C).
# 1D variants (960, 1024, 241, 50) require a different kernel; excluded here.
# make_elementwise_add_instructions(M, N, dram_a, dram_b, dram_c)


@pytest.mark.parametrize(
    "variant,M,N",
    [
        ("1024x3072", 1024, 3072),
        ("1024x768", 1024, 768),
        ("12x1024x64", _p32(12 * 1024), 64),
        ("12x64x1024", _p32(12 * 64), 1024),
        ("241x15x32", _p32(241 * 15), 32),
        ("241x5x32", _p32(241 * 5), 32),
        ("241x960", _p32(241), 960),
        ("291x15x32", _p32(291 * 15), 32),
        ("291x5x32", _p32(291 * 5), 32),
        ("32x32x768", _p32(32 * 32), 768),
        ("50x15x32", _p32(50 * 15), 32),
        ("50x32", _p32(50), 32),
        ("50x720", _p32(50), _p32(720)),
    ],
)
def test_elementwise_add_mlir_shape(variant, M, N):
    _check(make_elementwise_add_instructions(M, N, dram_a=0, dram_b=0, dram_c=0))


# ── elementwise_mul ───────────────────────────────────────────────────────────
# SaturnNPU/kernels/elementwise_mul/ — 15 variants.
# Broadcast and scalar-multiply variants (64×960 with scalar, 241×bf16 broadcast)
# are mapped to their full output shape.


@pytest.mark.parametrize(
    "variant,M,N",
    [
        ("1024x768", 1024, 768),
        ("48x960", _p32(48), 960),
        ("15x241x241", _p32(15 * 241), _p32(241)),
        ("15x291x291", _p32(15 * 291), _p32(291)),
        ("15x50x241", _p32(15 * 50), _p32(241)),
        ("241x15x32", _p32(241 * 15), 32),
        ("241x2560", _p32(241), 2560),
        ("241x5x32", _p32(241 * 5), 32),
        ("241x960", _p32(241), 960),
        ("291x15x32", _p32(291 * 15), 32),
        ("291x5x32", _p32(291 * 5), 32),
        ("50x15x32", _p32(50 * 15), 32),
        ("50x2048", _p32(50), 2048),
        ("50x720", _p32(50), _p32(720)),
        ("64x960", 64, 960),
    ],
)
def test_elementwise_mul_mlir_shape(variant, M, N):
    _check(make_elementwise_mul_instructions(M, N, dram_a=0, dram_b=0, dram_c=0))


# ── elementwise_div ───────────────────────────────────────────────────────────
# SaturnNPU/kernels/elementwise_div/ — 6 variants.
# Broadcast variants (A is 1D, B is 1D, output is 2D) are mapped to the
# full output shape.  Pure 1D variants are excluded.


@pytest.mark.parametrize(
    "variant,M,N",
    [
        ("241x32", _p32(241), 32),
        ("291x32", _p32(291), 32),
        ("50x32", _p32(50), 32),
    ],
)
def test_elementwise_div_mlir_shape(variant, M, N):
    _check(make_elementwise_div_instructions(M, N, dram_a=0, dram_b=0, dram_c=0))


# ── elementwise_sub ───────────────────────────────────────────────────────────
# SaturnNPU/kernels/elementwise_sub/ — 6 variants.


@pytest.mark.parametrize(
    "variant,M,N",
    [
        ("1024x768", 1024, 768),
        ("241x15x32", _p32(241 * 15), 32),
        ("241x5x32", _p32(241 * 5), 32),
        ("291x15x32", _p32(291 * 15), 32),
        ("291x5x32", _p32(291 * 5), 32),
        ("50x15x32", _p32(50 * 15), 32),
    ],
)
def test_elementwise_sub_mlir_shape(variant, M, N):
    _check(make_elementwise_sub_instructions(M, N, dram_a=0, dram_b=0, dram_c=0))


# ── matmul (fp8) ──────────────────────────────────────────────────────────────
# SaturnNPU/kernels/quantized_matmul_fp8/ — simple 2D variants.
# A(M, K) @ B^T(N, K) → C(M, N).  B is stored transposed in the MLIR.
# Variants with 3-D or higher-rank operands are handled in test_batch_matmul.
# make_matmul_instructions(M, K, N, dram_a, dram_b, dram_c)


@pytest.mark.parametrize(
    "variant,M,K,N",
    [
        # variant_0: A(1024,768), B(3072,768)
        ("1024x768x3072", 1024, 768, 3072),
        # variant_1: A(1024,768), B(768,768)
        ("1024x768x768", 1024, 768, 768),
        # variant_9: A(241,960), B(2560,960)
        ("241x960x2560", _p32(241), 960, 2560),
        # variant_10: A(241,960), B(320,960)
        ("241x960x320", _p32(241), 960, 320),
        # variant_11: A(241,960), B(960,960)
        ("241x960x960", _p32(241), 960, 960),
        # variant_13: A(50,720), B(2048,720)
        ("50x720x2048", _p32(50), _p32(720), 2048),
        # variant_14: A(50,720), B(320,720)
        ("50x720x320", _p32(50), _p32(720), 320),
        # variant_15: A(50,720), B(32,720)
        ("50x720x32", _p32(50), _p32(720), 32),
        # variant_16: A(50,32), B(720,32)
        ("50x32x720", _p32(50), 32, _p32(720)),
        # variant_17: A(50,720), B(960,720)
        ("50x720x960", _p32(50), _p32(720), 960),
        # variant_18: A(64,12288), B(960,12288)
        ("64x12288x960", 64, 12288, 960),
        # variant_20: A(32,), B(960,32) — vector-matrix, treat M=32
        ("32x32x960", 32, 32, 960),
    ],
)
def test_matmul_mlir_shape(variant, M, K, N):
    _check(make_matmul_instructions(M, K, N, dram_a=0, dram_b=0, dram_c=0))


# ── batch_matmul (fp8) ────────────────────────────────────────────────────────
# SaturnNPU/kernels/quantized_matmul_fp8/ — batched 3D variants.
# A(B, M, K) @ B(B, K, N) → C(B, M, N).
# make_batch_matmul_instructions(B, M, K, N, dram_a, dram_b, dram_c)


@pytest.mark.parametrize(
    "variant,B,M,K,N",
    [
        # variant_4: A(15,241,64), B(15,64,241)
        ("15x241x64x241", 15, _p32(241), 64, _p32(241)),
        # variant_5: A(15,291,64), B(15,64,291)
        ("15x291x64x291", 15, _p32(291), 64, _p32(291)),
        # variant_6: A(15,50,64), B(15,64,241)
        ("15x50x64x241", 15, _p32(50), 64, _p32(241)),
        # variant_8: A(15,241,241), B(15,241,64) → B=15, M=241, K=241, N=64
        ("15x241x241x64", 15, _p32(241), _p32(241), 64),
        # variant_12: A(15,50,241), B(15,241,64)
        ("15x50x241x64", 15, _p32(50), _p32(241), 64),
    ],
)
def test_batch_matmul_mlir_shape(variant, B, M, K, N):
    _check(
        make_batch_matmul_instructions(
            B, M, K, N, dram_a=0, dram_b=0, dram_c=0
        )
    )


# ── fused_silu_gate ───────────────────────────────────────────────────────────
# SaturnNPU/kernels/fused_silu_gate/op_a.mlir → 50×720 input tensor.
# make_fused_silu_gate_instructions(M, N, dram_x, dram_out)


@pytest.mark.parametrize(
    "variant,M,N",
    [
        ("50x720", _p32(50), _p32(720)),
    ],
)
def test_fused_silu_gate_mlir_shape(variant, M, N):
    _check(make_fused_silu_gate_instructions(M, N, dram_x=0, dram_out=0))


# ── fused_norm_scale ──────────────────────────────────────────────────────────
# SaturnNPU/kernels/fused_norm_scale/
# op_a = rms_norm on tensor<241xbf16>
# op_b = elementwise_mul on tensor<241x960xbf16>
# Kernel: make_fused_norm_scale_instructions(M, N, dram_var, dram_mat, dram_out)
# M = rows of the variance vector → _p32(241) = 256; N = feature width = 960.


@pytest.mark.parametrize(
    "variant,M,N",
    [
        ("241x960", _p32(241), 960),
    ],
)
def test_fused_norm_scale_mlir_shape(variant, M, N):
    _check(
        make_fused_norm_scale_instructions(M, N, dram_var=0, dram_mat=0, dram_out=0)
    )


# ── fused_matmul_bias ─────────────────────────────────────────────────────────
# SaturnNPU/kernels/fused_matmul_bias/ — K=32 fixed matmul + bias add.
# The MLIR op_a produces a complex 768×32×32 tensor via a tiled contraction;
# the simplest decomposition uses M=32, N=32 (single tile).
# make_fused_matmul_bias_instructions(M, N, dram_a, dram_b, dram_bias, dram_out)


@pytest.mark.parametrize(
    "variant,M,N",
    [
        ("32x32", 32, 32),
        ("32x768", 32, 768),
    ],
)
def test_fused_matmul_bias_mlir_shape(variant, M, N):
    _check(
        make_fused_matmul_bias_instructions(
            M, N, dram_a=0, dram_b=0, dram_bias=0, dram_out=0
        )
    )


# ── fused_attention ───────────────────────────────────────────────────────────
# SaturnNPU/kernels/iree_linalg_ext.attention/variant_0_12_1024_64_bf16.mlir
# Q(12,1024,64), K(12,1024,64), V(12,64,1024) — 12 heads each 1024×64.
# One kernel call handles one head: Q_ROWS=1024, K_SEQ=1024, HEAD_DIM=64.
# make_fused_attention_instructions(Q_ROWS, K_SEQ, HEAD_DIM, dram_q, dram_kt_base,
#                                   dram_vt_base, dram_scale, dram_out)


@pytest.mark.parametrize(
    "variant,Q_ROWS,K_SEQ,HEAD_DIM",
    [
        ("12x1024x64_one_head", 1024, 1024, 64),
    ],
)
def test_fused_attention_mlir_shape(variant, Q_ROWS, K_SEQ, HEAD_DIM):
    _check(
        make_fused_attention_instructions(
            Q_ROWS,
            K_SEQ,
            HEAD_DIM,
            dram_q=0,
            dram_kt_base=0,
            dram_vt_base=0,
            dram_scale=0,
            dram_out=0,
        )
    )


# ── requant ───────────────────────────────────────────────────────────────────
# No dedicated MLIR directory; requant (bf16→fp8) shapes are derived from
# type_conversion and quantized_matmul_fp8 output tensors.
# make_requant_instructions(M, N, dram_x, dram_out)


@pytest.mark.parametrize(
    "variant,M,N",
    [
        ("1024x768", 1024, 768),
        ("1024x3072", 1024, 3072),
        ("241x960", _p32(241), 960),
        ("241x2560", _p32(241), 2560),
        ("50x720", _p32(50), _p32(720)),
        ("50x2048", _p32(50), 2048),
    ],
)
def test_requant_mlir_shape(variant, M, N):
    _check(make_requant_instructions(M, N, dram_x=0, dram_out=0))


# ── bias_add_cast ─────────────────────────────────────────────────────────────
# No dedicated MLIR directory; operates on a fixed 32×32 tile.
# Shape is determined by the tile size, not by MLIR variants.
# make_bias_add_cast_instructions(dram_x, dram_bias, dram_out)


def test_bias_add_cast_mlir_shape():
    _check(make_bias_add_cast_instructions(dram_x=0, dram_bias=0, dram_out=0))
