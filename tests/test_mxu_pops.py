"""Accumulator pops of both MXUs: BF16_RNE with the GPU's NaN (docs/fp8-mxu.md, docs/nvfp4-mxu.md)."""
import numpy as np
import pytest
import torch

from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.configs.isa_definition import VMATPOP_BF16_ACC_NV
from npu_model.hardware.arch_state import ArchState
from npu_model.software import acc, m

# FP32 accumulator bits and BF16_RNE of each (cvt.rn.bf16.f32 on the RTX 5090).
UNSCALED = [
    (0x3F80_0000, 0x3F80), (0x3F80_8000, 0x3F80), (0x3F81_8000, 0x3F82), (0x8000_0000, 0x8000),
    (0x0000_8000, 0x0000), (0x0001_8000, 0x0002), (0x7F7F_FFFF, 0x7F80), (0xFF80_0000, 0xFF80),
    (0x7FC0_0000, 0x7FFF), (0xFFC0_0001, 0x7FFF), (0x7F80_0001, 0x7FFF), (0xFFFF_FFFF, 0x7FFF),
]


@pytest.fixture
def state():
    s = ArchState(DefaultHardwareConfig().arch_state_config)
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
