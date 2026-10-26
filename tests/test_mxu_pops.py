"""Accumulator pushes and pops of both MXUs: BF16_RNE with the GPU's NaN, the scaled BF16 pops, which
round BF16_RNE(acc x 2^(e - 127)) once, and the FP32 push and pop of mxu1, which copy every bit
(docs/fp8-mxu.md, docs/nvfp4-mxu.md)."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from npu_model.configs import cvt
from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.configs.numerics import TorchNumerics
from npu_model.configs.isa_definition import (
    VMATPOP_BF16_ACC_NV, VMATPOP_BF16_SCALED_ACC_MXU1, VMATPOP_BF16_SCALED_ACC_NV, VMATPOP_FP32_ACC_MXU1,
    VMATPUSH_ACC_FP32_MXU1,
)
from npu_model.hardware.arch_state import ArchState
from npu_model.software import acc, e, m

# FP32 accumulator bits and BF16_RNE of each (cvt.rn.bf16.f32 on the RTX 5090).
UNSCALED = [
    (0x3F80_0000, 0x3F80), (0x3F80_8000, 0x3F80), (0x3F81_8000, 0x3F82), (0x8000_0000, 0x8000),
    (0x0000_8000, 0x0000), (0x0001_8000, 0x0002), (0x7F7F_FFFF, 0x7F80), (0xFF80_0000, 0xFF80),
    (0x7FC0_0000, 0x7FFF), (0xFFC0_0001, 0x7FFF), (0x7F80_0001, 0x7FFF), (0xFFFF_FFFF, 0x7FFF),
]

# (FP32 accumulator bits, e, BF16 bits of acc x 2^(e - 127) rounded once), worked by hand.
SCALED = [
    # 1.0 at every e of the e-register's range ends: 2^-127 is the BF16 subnormal 64 x 2^-133.
    (0x3F80_0000, 0, 0x0040), (0x3F80_0000, 1, 0x0080), (0x3F80_0000, 126, 0x3F00),
    (0x3F80_0000, 127, 0x3F80), (0x3F80_0000, 128, 0x4000), (0x3F80_0000, 254, 0x7F00),
    (0x3F80_0000, 255, 0x7F80),                                   # 2^128 overflows to Inf
    (0xBFC0_0000, 254, 0xFF40), (0xBFC0_0000, 255, 0xFF80),       # -1.5 x 2^127, -1.5 x 2^128
    (0x7F7F_FFFF, 126, 0x7F00), (0x7F7F_FFFF, 127, 0x7F80),       # FP32 max rounds up
    # Ties to even: 1 + 2^-8 and 1 + 3 x 2^-8.
    (0x3F80_8000, 126, 0x3F00), (0x3F81_8000, 128, 0x4002),
    # FP32 subnormal accumulators.
    (0x0000_0001, 255, 0x3500),                                   # 2^-149 x 2^128 = 2^-21
    (0x007F_FFFF, 255, 0x4080),                                   # 4 - 2^-21 rounds to 4
    (0x0040_0000, 127, 0x0040), (0x0040_0000, 0, 0x0000),
    (0x8000_0001, 127, 0x8000),                                   # a negative result that rounds to zero
    # BF16 subnormal results: ties to even at 1.5 and 2.5 x 2^-133, and one rounding of
    # 1.49999237 x 2^-133 (an FMUL to FP32 first would round it to the tie 1.5 x 2^-133, then 2).
    (0x0060_0000, 121, 0x0002), (0x00A0_0000, 121, 0x0002), (0x3CBF_FFC0, 0, 0x0001),
    # Specials keep their class at every e.
    (0x7F80_0000, 0, 0x7F80), (0xFF80_0000, 0, 0xFF80), (0x0000_0000, 255, 0x0000),
    (0x8000_0000, 255, 0x8000), (0x7FC0_0000, 0, 0x7FFF), (0xFFC0_0001, 255, 0x7FFF),
    (0x7F80_0001, 127, 0x7FFF),
]


@pytest.fixture
def state():
    s = ArchState(replace(DefaultHardwareConfig().arch_state_config, numerics="rtl"))
    yield s
    s.close()


def tile(values) -> np.ndarray:
    """A 32 x 32 FP32 tile (bits) that repeats values."""
    return np.resize(np.array(values, dtype=np.uint32), (32, 32))


def bf16_tile(state, reg) -> np.ndarray:
    return state.read_mrf_bf16_tile(reg).view(torch.int16).numpy().view(np.uint16)


def test_nv_bf16_pop_rounds_to_nearest_even_and_gives_the_gpu_nan(state):
    x = tile([x for x, _ in UNSCALED])
    state.acc_nv[1] = x.tolist()
    VMATPOP_BF16_ACC_NV(vd=m(4), vs2=acc(1)).exec(state)
    want = dict(UNSCALED)
    assert (bf16_tile(state, m(4)) == np.vectorize(want.get)(x)).all()


@pytest.mark.parametrize("mxu", ["mxu1", "nv"])
@pytest.mark.parametrize("x, scale, want", SCALED, ids=[f"{x:08x}-e{s}" for x, s, _ in SCALED])
def test_scaled_bf16_pop_rounds_the_exact_product_once(state, mxu, x, scale, want):
    if mxu == "mxu1":
        state.acc["mxu1"][1].view(torch.int32)[:] = torch.from_numpy(tile([x]).view(np.int32))
        pop = VMATPOP_BF16_SCALED_ACC_MXU1
    else:
        state.acc_nv[1] = tile([x]).tolist()
        pop = VMATPOP_BF16_SCALED_ACC_NV
    state.write_erf(3, scale)
    pop(vd=m(6), vs2=acc(1), es1=e(3)).exec(state)
    assert (bf16_tile(state, m(6)) == want).all()


def test_a_two_rounding_epilogue_differs():
    """A GPU epilogue (FMUL, then cvt.rn.bf16.f32) rounds 0x3CBFFFC0 x 2^-127 to the FP32 tie
    1.5 x 2^-133 and then to 0x0002; the pop gives 0x0001 (SCALED)."""
    x = np.array(0x3CBF_FFC0, np.uint32).view(np.float32)
    product = (x * np.float32(2.0 ** -127)).view(np.uint32)
    assert product == 0x0001_8000 and cvt.bf16_rne(product) == 0x0002


# FP32 bit patterns of every class, both signs: zero, subnormals, normals, the largest finite value,
# infinity, and quiet and signaling NaNs with payloads.
FP32_CLASSES = [
    0x0000_0000, 0x8000_0000, 0x0000_0001, 0x8000_0001, 0x007F_FFFF, 0x807F_FFFF, 0x0080_0000,
    0x8080_0000, 0x3F80_0000, 0xBF80_0000, 0x7F7F_FFFF, 0xFF7F_FFFF, 0x7F80_0000, 0xFF80_0000,
    0x7FC0_0000, 0xFFC0_0000, 0x7FC1_2345, 0xFFFF_FFFF, 0x7F80_0001, 0xFF80_0001, 0x7FBF_FFFF,
    0xFFBF_FFFF,
]


def test_fp32_push_and_pop_copy_every_bit_pattern(state):
    rng = np.random.default_rng(7)
    words = np.concatenate([np.array(FP32_CLASSES, np.uint32),
                            rng.integers(0, 1 << 32, 1024 - len(FP32_CLASSES), dtype=np.uint32)]).reshape(32, 32)
    for q in range(4):  # register q holds columns 8q .. 8q + 7
        state.mrf[8 + q].view(torch.int32)[:] = torch.from_numpy(words[:, 8 * q:8 * q + 8].ravel().view(np.int32))
    VMATPUSH_ACC_FP32_MXU1(vd=acc(1), vs1=m(8)).exec(state)
    assert (state.acc["mxu1"][1].view(torch.int32).numpy().view(np.uint32) == words).all()
    VMATPOP_FP32_ACC_MXU1(vd=m(12), vs2=acc(1)).exec(state)
    for q in range(4):
        assert torch.equal(state.mrf[12 + q], state.mrf[8 + q]), f"m{12 + q}"


def test_pytorch_numerics_pop_rounds_once_as_cvt():
    """The default numerics ("pytorch") give cvt.bf16_rne's bits at every e: the SCALED cases, the FP32
    classes and random words."""
    rng = np.random.default_rng(5)
    words = np.concatenate([np.array([x for x, _, _ in SCALED] + FP32_CLASSES, np.uint32),
                            rng.integers(0, 1 << 32, 4096, dtype=np.uint64).astype(np.uint32)])
    x = torch.from_numpy(words.view(np.int32)).view(torch.float32)
    for scale in range(256):
        got = TorchNumerics.f32_to_bf16(x, scale).view(torch.int16).numpy().view(np.uint16)
        assert (got == cvt.bf16_rne(words, scale)).all(), f"e = {scale}"
