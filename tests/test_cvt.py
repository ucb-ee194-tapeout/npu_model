"""configs/cvt.py against RTX 5090 conversion outputs (baremetal/generators/fp8_silicon)."""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from npu_model.configs import cvt

VECTORS = Path(__file__).resolve().parents[2] / "baremetal/generators/fp8_silicon"

# (name, FP32 x, e, RTX 5090 result) from fp8_silicon/*.cvt.gz
E4M3 = [
    ("tie at 2^-10: to even 0", 0x3A80_0000, 127, 0x00),
    ("above 2^-10: 2^-9", 0x3A80_0001, 127, 0x01),
    ("-2^-10: -0", 0xBA80_0000, 127, 0x80),
    ("-0", 0x8000_0000, 127, 0x80),
    ("subnormal tie 1.5 x 2^-9: to even", 0x3B40_0000, 127, 0x02),
    ("tie 7.5 x 2^-9: to even 2^-6", 0x3C70_0000, 127, 0x08),
    ("tie at 464: 448", 0x43E8_0000, 127, 0x7E),
    ("above 464: 448", 0x43E8_0001, 127, 0x7E),
    ("+Inf: 448", 0x7F80_0000, 127, 0x7E),
    ("-Inf: -448", 0xFF80_0000, 127, 0xFE),
    ("-NaN: 0x7F", 0xFFC0_0001, 127, 0x7F),
    ("FP32 subnormal x 2^128", 0x007F_FFFF, 255, 0x48),
    ("-FP32 subnormal x 2^128", 0x807F_FFFF, 255, 0xC8),
    ("2^-127 x 2^128", 0x0040_0000, 255, 0x40),
    ("BF16 2^-133 x 2^128", 0x0001_0000, 255, 0x10),
    ("FLT_MAX x 2^-127", 0x7F7F_FFFF, 0, 0x40),
    ("-2^-149 x 2^-127: -0", 0x8000_0001, 0, 0x80),
]
BF16 = [
    ("subnormal tie: to even 0", 0x0000_8000, 127, 0x0000),
    ("above the subnormal tie", 0x0000_8001, 127, 0x0001),
    ("subnormal tie: to even 2", 0x0001_8000, 127, 0x0002),
    ("tie at 1 + 2^-8: to even 1", 0x3F80_8000, 127, 0x3F80),
    ("tie at 1 + 3 x 2^-8: to even", 0x3F81_8000, 127, 0x3F82),
    ("largest FP32 subnormal: -2^-126", 0x807F_FFFF, 127, 0x8080),
    ("below the overflow tie", 0x7F7F_7FFF, 127, 0x7F7F),
    ("overflow tie: Inf", 0x7F7F_8000, 127, 0x7F80),
    ("-0", 0x8000_0000, 127, 0x8000),
    ("NaN: 0x7FFF", 0x7FC0_0000, 127, 0x7FFF),
    ("-NaN: 0x7FFF", 0xFF80_0001, 127, 0x7FFF),
    ("NaN with a low payload: 0x7FFF", 0x7F80_FFFF, 127, 0x7FFF),
    ("2^-9 x 2^-124: smallest subnormal", 0x3B00_0000, 3, 0x0001),
    ("-2^-9 x 2^-127: -0", 0xBB00_0000, 0, 0x8000),
    ("2^128: Inf", 0x3F80_0000, 255, 0x7F80),
]
E4M3_TO_F32 = [(0x00, 0), (0x80, 0x8000_0000), (0x01, 0x3B00_0000), (0x07, 0x3C60_0000),
               (0x08, 0x3C80_0000), (0x7E, 0x43E0_0000), (0xFE, 0xC3E0_0000), (0x7F, 0x7FFF_FFFF),
               (0xFF, 0x7FFF_FFFF)]


@pytest.mark.parametrize("name,x,e,d", E4M3, ids=[c[0] for c in E4M3])
def test_e4m3_rn_satfinite(name, x, e, d):
    assert int(cvt.e4m3_rn_satfinite(x, e)) == d


@pytest.mark.parametrize("name,x,e,d", BF16, ids=[c[0] for c in BF16])
def test_bf16_rne(name, x, e, d):
    assert int(cvt.bf16_rne(x, e)) == d


@pytest.mark.parametrize("code,d", E4M3_TO_F32, ids=[f"{c[0]:#04x}" for c in E4M3_TO_F32])
def test_e4m3_to_f32(code, d):
    assert int(cvt.e4m3_to_f32(code)) == d


def test_scaled_value_rounds_once():
    # (1.5 - 2^-17) x 2^-133 is below a BF16 tie; FP32 first would round it onto the tie, then up
    assert int(cvt.bf16_rne(0x3CBF_FFC0, 0)) == 0x0001


def test_round_trips():
    codes = np.arange(256)
    finite = codes & 0x7F != 0x7F
    back = cvt.e4m3_rn_satfinite(cvt.e4m3_to_f32(codes))
    np.testing.assert_array_equal(back[finite], codes[finite])
    bf16 = np.arange(1 << 16, dtype=np.uint32)
    finite = bf16 & 0x7F80 != 0x7F80
    np.testing.assert_array_equal(cvt.bf16_rne(bf16 << 16)[finite], bf16[finite])


def _read_cvt(path):
    """name and columns of a cvt-vectors-v1 file."""
    raw = gzip.decompress(path.read_bytes())
    nl = raw.index(b"\n")
    header = json.loads(raw[:nl])
    columns, offset = {}, nl + 1
    for name, dtype in header["columns"]:
        columns[name] = np.frombuffer(raw, dtype, header["count"], offset)
        offset += columns[name].nbytes
    return header["name"], columns


@pytest.mark.parametrize("path", sorted(VECTORS.glob("*.cvt.gz")),
                         ids=lambda p: p.name.split(".")[0])
def test_silicon_vectors(path):
    name, columns = _read_cvt(path)
    rtx5090 = columns["rtx5090"]
    i = np.arange(len(rtx5090), dtype=np.uint32)
    if name == "vpack":
        model = cvt.e4m3_rn_satfinite((i & 0xFFFF) << 16, i >> 16)
    elif name == "vunpack":
        model = cvt.bf16_rne(cvt.e4m3_to_f32(i & 0xFF), i >> 8)
    elif name == "push":
        model = cvt.e4m3_to_f32(i)
    elif name == "pop":
        model = cvt.e4m3_rn_satfinite(columns["x"], columns["e"])
    else:  # runs: the first and the last input of every run
        x = np.concatenate([columns["x"], np.append(columns["x"][1:] - 1, 0xFFFF_FFFF)])
        model = {"e4m3_runs": cvt.e4m3_rn_satfinite, "bf16_runs": cvt.bf16_rne}[name](x)
        rtx5090 = np.tile(rtx5090, 2)
    np.testing.assert_array_equal(model, rtx5090)
